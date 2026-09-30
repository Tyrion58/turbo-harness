#!/usr/bin/env bash
# Meta-Harness cell (SWE-bench Verified): the global harness H*, in-process swebench scoring.
# Usage: STUDENT=haiku bash scripts/verified/eval_meta.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
BENCH=verified
# shellcheck source=/dev/null
source scripts/env.sh
unset EVAL_SERVER_URL   # Verified scores in-process (swebench)
mkdir -p results

python -m turbo_harness.eval.harness_eval \
  --harness "$ART/general" \
  --data_file "$DATA_TEST" \
  --student "$STUDENT_MODEL" \
  --repo "$REPO_FLAG" \
  --num_runs "${NUM_RUNS:-3}" \
  --max_workers "${MAX_WORKERS:-10}" \
  --out "results/verified_${STUDENT}_meta.json"
