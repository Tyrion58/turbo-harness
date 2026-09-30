#!/usr/bin/env bash
# Stage C (SWE-bench Verified): GRPO editor training. Thin wrapper that sets BENCH=verified and
# reuses the SWE trainer; env.sh then points ART/SWE_HARNESS_DIR at the Verified general harness.
#
# Build the Verified RL dataset first (note the Verified splits/artifacts + a distinct output):
#   python -m turbo_harness.rl.build_rl_dataset \
#     --data-file data/swe_smith/train_verified.json \
#     --harness-dir artifacts/verified/haiku/general \
#     --playbook   artifacts/verified/haiku/playbook.json \
#     --output     data/rl_verified/train.parquet
# then run this with DATA_DIR pointing at that parquet's directory (trainer reads
# ${DATA_DIR}/train.parquet):
#   STUDENT=haiku RUN_NAME=harness_advisor_grpo_qwen35_verified \
#     DATA_DIR=data/rl_verified bash scripts/verified/train_rl.sh
exec env BENCH=verified bash "$(dirname "$0")/../swe_smith/train_rl.sh" "$@"
