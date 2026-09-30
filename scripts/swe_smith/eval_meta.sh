#!/usr/bin/env bash
# Meta-Harness cell: run the global harness H* (no advisor) on SWE-smith-MR test.
# In-process executor + eval_server:5152 + Docker. Usage: STUDENT=haiku bash scripts/swe_smith/eval_meta.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
# shellcheck source=/dev/null
source scripts/env.sh
mkdir -p results

python -m turbo_harness.eval.harness_eval \
  --harness "$ART/general" \
  --data_file "$DATA_TEST" \
  --student "$STUDENT_MODEL" \
  --repo multi_repo \
  --num_runs "${NUM_RUNS:-3}" \
  --max_workers "${MAX_WORKERS:-10}" \
  --out "results/swe_smith_${STUDENT}_meta.json"
