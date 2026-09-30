#!/usr/bin/env bash
# Turbo cell (TB2.1): serve the trained editor on vLLM; it patches the winner harness (kira_auto_test)
# per task, and the Sonnet-4.5 student runs the patched harness via harbor. Set ADVISOR_CKPT to a
# .../global_step_N/policy dir.
# Usage: VERTEXAI_PROJECT=... TB2_ENGINE_DIR=... ADVISOR_CKPT=/path/to/policy bash scripts/tb2/eval_turbo.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
# shellcheck source=/dev/null
source scripts/tb2/_env.sh
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
  --max-model-len 32768 > /tmp/tb2_advisor.log 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null' EXIT
for _ in $(seq 1 60); do
  grep -qi 'Application startup complete' /tmp/tb2_advisor.log 2>/dev/null && break
  sleep 10
done

python -m turbo_harness.eval.tb2_advisor_eval \
  --advisor-url "http://127.0.0.1:${ADVISOR_PORT}/v1" --advisor-model advisor \
  --split turbo_harness/terminal_bench/data/test_tasks.json \
  --playbook artifacts/tb2/playbook.json \
  --student "$STUDENT_MODEL" \
  --count "${COUNT:-44}" \
  --out results/tb2_turbo.jsonl
