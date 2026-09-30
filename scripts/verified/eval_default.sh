#!/usr/bin/env bash
# Default cell (SWE-bench Verified): unoptimized mini-swe-agent, scored IN-PROCESS by swebench.
# No servers needed. Usage: STUDENT=haiku bash scripts/verified/eval_default.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
BENCH=verified
# shellcheck source=/dev/null
source scripts/env.sh
unset EVAL_SERVER_URL   # Verified scores in-process (swebench); do NOT route to eval_server
mkdir -p results

python -m turbo_harness.eval.harness_eval \
  --harness default \
  --data_file "$DATA_TEST" \
  --student "$STUDENT_MODEL" \
  --repo "$REPO_FLAG" \
  --num_runs "${NUM_RUNS:-3}" \
  --max_workers "${MAX_WORKERS:-10}" \
  --out "results/verified_${STUDENT}_default.json"
