#!/usr/bin/env bash
# Serve the FROZEN Qwen3.5-9B ALFWorld student via vLLM (tool-calling enabled). Give it its own GPUs.
# Usage: CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/alfworld/serve_student.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
: "${VLLM_PYTHON:=SkyRL/.venv/bin/python}"
: "${STUDENT_MODEL_PATH:=Qwen/Qwen3.5-9B}"
: "${PORT:=8110}"
: "${DP:=4}"
"$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$STUDENT_MODEL_PATH" --served-model-name Qwen3.5-9B --port "$PORT" \
  --data-parallel-size "$DP" \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --max-model-len 32768 --dtype bfloat16 --gpu-memory-utilization 0.85
