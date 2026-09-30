#!/usr/bin/env bash
# Turbo cell (ours): serve the trained harness-editor on vLLM, generate a per-issue SEARCH/REPLACE
# patch to H*, and run the frozen student on the patched harness. In-process executor + eval_server:5152.
# Set ADVISOR_CKPT to a .../global_step_N/policy dir.
# Usage: STUDENT=haiku ADVISOR_CKPT=/path/to/policy bash scripts/swe_smith/eval_turbo.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
# shellcheck source=/dev/null
source scripts/env.sh
: "${ADVISOR_CKPT:?set ADVISOR_CKPT to the editor checkpoint policy dir}"
: "${VLLM_PYTHON:=python}"   # any interpreter with vLLM (e.g. the SkyRL venv, or 'pip install vllm')
: "${ADVISOR_PORT:=8120}"
: "${DP:=8}"                                   # vLLM data-parallel size (GPUs)
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"   # vLLM ignores it; litellm requires it be set
mkdir -p results

echo "Serving editor: $ADVISOR_CKPT -> :$ADVISOR_PORT"
"$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$ADVISOR_CKPT" --served-model-name advisor --port "$ADVISOR_PORT" \
  --data-parallel-size "$DP" --dtype bfloat16 --gpu-memory-utilization 0.85 \
  --max-model-len 32768 > /tmp/turbo_advisor.log 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null' EXIT
for _ in $(seq 1 60); do
  grep -qi 'Application startup complete' /tmp/turbo_advisor.log 2>/dev/null && break
  sleep 10
done

python -m turbo_harness.eval.playbook_eval \
  --harness-dir "$ART/general" \
  --data_file "$DATA_TEST" \
  --playbook "$ART/playbook.json" \
  --patch-model openai/advisor --patch-api-base "http://127.0.0.1:${ADVISOR_PORT}/v1" \
  --student "$STUDENT_MODEL" --repo multi_repo \
  --num_runs "${NUM_RUNS:-3}" --max_workers "${MAX_WORKERS:-10}" \
  --out "results/swe_smith_${STUDENT}_turbo.json"
