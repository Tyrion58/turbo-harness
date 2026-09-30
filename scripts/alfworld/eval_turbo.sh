#!/usr/bin/env bash
# Turbo cell (ALFWorld): serve the trained editor on vLLM, patch H* per instance, frozen Qwen student
# runs the patched harness. Requires the STUDENT already served on $STUDENT_URL (serve_student.sh) and
# ALFWORLD_DATA. Set ADVISOR_CKPT to a .../global_step_N/policy dir.
# Usage: ALFWORLD_DATA=... ADVISOR_CKPT=/path/to/policy bash scripts/alfworld/eval_turbo.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
# shellcheck source=/dev/null
source scripts/alfworld/_env.sh
: "${ADVISOR_CKPT:?set ADVISOR_CKPT to the editor checkpoint policy dir}"
: "${VLLM_PYTHON:=python}"   # any interpreter with vLLM (e.g. the SkyRL venv, or 'pip install vllm')
: "${ADVISOR_PORT:=8120}"
: "${DP:=4}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
mkdir -p results

echo "Serving editor: $ADVISOR_CKPT -> :$ADVISOR_PORT"
"$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$ADVISOR_CKPT" --served-model-name advisor --port "$ADVISOR_PORT" \
  --data-parallel-size "$DP" --dtype bfloat16 --gpu-memory-utilization 0.85 \
  --max-model-len 32768 > /tmp/alfworld_advisor.log 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null' EXIT
for _ in $(seq 1 60); do
  grep -qi 'Application startup complete' /tmp/alfworld_advisor.log 2>/dev/null && break
  sleep 10
done

python -m turbo_harness.eval.alfworld_advisor_eval \
  --advisor-url "http://127.0.0.1:${ADVISOR_PORT}/v1" --advisor-model advisor \
  --harness-dir "$ART_ALF/general" \
  --playbook "$ART_ALF/playbook.json" \
  --split "$TEST_SPLIT" --start 0 --count "${COUNT:-150}" \
  --concurrency "${CONCURRENCY:-8}" --max-steps 50 \
  --out results/alfworld_turbo.jsonl
