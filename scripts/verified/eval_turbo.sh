#!/usr/bin/env bash
# Turbo cell (SWE-bench Verified): serve the trained editor on vLLM, patch H* per issue, run the
# frozen student; scored IN-PROCESS by swebench. Set ADVISOR_CKPT to a .../global_step_N/policy dir
# Usage: STUDENT=haiku ADVISOR_CKPT=/path/to/policy bash scripts/verified/eval_turbo.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
BENCH=verified
# shellcheck source=/dev/null
source scripts/env.sh
unset EVAL_SERVER_URL   # Verified scores in-process (swebench)
: "${ADVISOR_CKPT:?set ADVISOR_CKPT to the editor checkpoint policy dir}"
: "${VLLM_PYTHON:=python}"   # any interpreter with vLLM (e.g. the SkyRL venv, or 'pip install vllm')
: "${ADVISOR_PORT:=8120}"
: "${DP:=8}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
mkdir -p results

echo "Serving editor: $ADVISOR_CKPT -> :$ADVISOR_PORT"
"$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$ADVISOR_CKPT" --served-model-name advisor --port "$ADVISOR_PORT" \
  --data-parallel-size "$DP" --dtype bfloat16 --gpu-memory-utilization 0.85 \
  --max-model-len 32768 > /tmp/verified_turbo_advisor.log 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null' EXIT
for _ in $(seq 1 60); do
  grep -qi 'Application startup complete' /tmp/verified_turbo_advisor.log 2>/dev/null && break
  sleep 10
done

python -m turbo_harness.eval.playbook_eval \
  --harness-dir "$ART/general" \
  --data_file "$DATA_TEST" \
  --playbook "$ART/playbook.json" \
  --patch-model openai/advisor --patch-api-base "http://127.0.0.1:${ADVISOR_PORT}/v1" \
  --student "$STUDENT_MODEL" --repo "$REPO_FLAG" \
  --num_runs "${NUM_RUNS:-3}" --max_workers "${MAX_WORKERS:-10}" \
  --out "results/verified_${STUDENT}_turbo.json"
