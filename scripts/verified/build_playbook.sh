#!/usr/bin/env bash
# (Optional) Stage B (SWE-bench Verified): rebuild the playbook from a completed Stage-A run.
# Paths follow meta_harness's output layout; set RUN / ART_SUBDIR to match your evolve run.
# Usage: STUDENT=haiku RUN=verified_haiku bash scripts/verified/build_playbook.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
BENCH=verified
# shellcheck source=/dev/null
source scripts/env.sh
RUN="${RUN:-verified_${STUDENT}}"
LOGS="experiments/logs/meta_harness/${RUN}"
ART_SUBDIR="${ART_SUBDIR:-artifacts/verified/${STUDENT}/${RUN}}"

python -m turbo_harness.playbook.run_pipeline \
  --evolution-log "${LOGS}/evolution_summary.jsonl" \
  --trajectory-dir "${LOGS}/trajectories" \
  --artifacts-dir "${ART_SUBDIR}" \
  --data-file "$DATA_TRAIN" \
  --output-dir playbook_output/ \
  --run-name "${RUN}" \
  --reflector-model "vertex_ai/${TURBO_FRONTIER_MODEL}" \
  --curator-model "vertex_ai/${TURBO_FRONTIER_MODEL}"
