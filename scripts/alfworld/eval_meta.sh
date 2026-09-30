#!/usr/bin/env bash
# Meta-Harness cell (ALFWorld): the global harness H*, frozen Qwen student.
# Requires the student served + ALFWORLD_DATA. Usage: ALFWORLD_DATA=... bash scripts/alfworld/eval_meta.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
# shellcheck source=/dev/null
source scripts/alfworld/_env.sh
mkdir -p results

python -m turbo_harness.alfworld.runner \
  --harness-dir "$ART_ALF/general" \
  --split "$TEST_SPLIT" --start 0 --count "${COUNT:-150}" \
  --base-url "$STUDENT_URL" --model "$STUDENT_MODEL" \
  --concurrency "${CONCURRENCY:-8}" --max-steps 50 \
  --out results/alfworld_meta.jsonl
