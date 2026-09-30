#!/usr/bin/env bash
# (Optional) Stage B: rebuild the playbook from a completed Stage-A evolution run. The shipped
# artifacts/swe_smith/<student>/playbook.json was produced this way. Paths follow meta_harness's
# output layout; set RUN (and ART_SUBDIR if your student slug differs) to match your evolve run.
# Usage: STUDENT=haiku RUN=swe_smith_haiku bash scripts/swe_smith/build_playbook.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
# shellcheck source=/dev/null
source scripts/env.sh
RUN="${RUN:-swe_smith_${STUDENT}}"
LOGS="experiments/logs/meta_harness/${RUN}"
ART_SUBDIR="${ART_SUBDIR:-artifacts/multi_repo/${STUDENT}/${RUN}}"

python -m turbo_harness.playbook.run_pipeline \
  --evolution-log "${LOGS}/evolution_summary.jsonl" \
  --trajectory-dir "${LOGS}/trajectories" \
  --artifacts-dir "${ART_SUBDIR}" \
  --data-file "$DATA_TRAIN" \
  --output-dir playbook_output/ \
  --run-name "${RUN}" \
  --reflector-model "vertex_ai/${TURBO_FRONTIER_MODEL}" \
  --curator-model "vertex_ai/${TURBO_FRONTIER_MODEL}"
