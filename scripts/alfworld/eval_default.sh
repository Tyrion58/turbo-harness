#!/usr/bin/env bash
# Default cell (ALFWorld): unoptimized harness, frozen Qwen student. Requires the student served
# (scripts/alfworld/serve_student.sh) + ALFWORLD_DATA. Usage: ALFWORLD_DATA=... bash scripts/alfworld/eval_default.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
# shellcheck source=/dev/null
source scripts/alfworld/_env.sh
mkdir -p results

python -m turbo_harness.alfworld.runner \
  --harness-dir default \
  --split "$TEST_SPLIT" --start 0 --count "${COUNT:-150}" \
  --base-url "$STUDENT_URL" --model "$STUDENT_MODEL" \
  --concurrency "${CONCURRENCY:-8}" --max-steps 50 \
  --out results/alfworld_default.jsonl
