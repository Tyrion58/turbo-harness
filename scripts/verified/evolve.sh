#!/usr/bin/env bash
# (Optional) Stage A (SWE-bench Verified): regenerate the global harness H* via Meta-Harness search
# on the full 251-issue train split. Verified scores IN-PROCESS (swebench), so no eval_server; needs
# frontier API (proposer) + Docker (with the Verified task images available locally).
# Usage: STUDENT=haiku bash scripts/verified/evolve.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
BENCH=verified
# shellcheck source=/dev/null
source scripts/env.sh
unset EVAL_SERVER_URL   # Verified scores in-process (swebench)

python -m turbo_harness.meta_harness \
  --repo verified \
  --student "$STUDENT_MODEL" \
  --proposer-model "$TURBO_FRONTIER_MODEL" \
  --iterations 5 --trials 2 --concurrency "${CONCURRENCY:-10}" \
  --run-name "${RUN:-verified_${STUDENT}}" \
  --fresh
