#!/usr/bin/env bash
# (Optional) Stage A: regenerate the global harness H* via Meta-Harness evolution. The shipped
# artifacts/swe_smith/<student>/general/ was produced this way. Needs frontier API (proposer) +
# eval_server:5152 + Docker. Writes logs to experiments/logs/meta_harness/<run> and candidate
# harnesses to artifacts/multi_repo/<student_slug>/<run> (meta_harness's own layout).
# Usage: STUDENT=haiku bash scripts/swe_smith/evolve.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
# shellcheck source=/dev/null
source scripts/env.sh

python -m turbo_harness.meta_harness \
  --repo multi_repo \
  --student "$STUDENT_MODEL" \
  --proposer-model "$TURBO_FRONTIER_MODEL" \
  --search-subset 0-49 \
  --iterations 5 --trials 2 --concurrency "${CONCURRENCY:-10}" \
  --run-name "${RUN:-swe_smith_${STUDENT}}" \
  --fresh
