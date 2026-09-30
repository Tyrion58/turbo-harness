#!/usr/bin/env bash
# Default cell: unoptimized mini-swe-agent DefaultAgent on SWE-smith-MR test (no harness opt).
# Requires BOTH servers (agent_server:8081 + eval_server:5152) and Docker. Results are printed.
# Usage: STUDENT=haiku bash scripts/swe_smith/eval_default.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
# shellcheck source=/dev/null
source scripts/env.sh

python -m turbo_harness.eval.baseline \
  --model "$STUDENT_MODEL" \
  --data_file "$DATA_TEST" \
  --num_runs "${NUM_RUNS:-3}" \
  --max_workers "${MAX_WORKERS:-10}"
