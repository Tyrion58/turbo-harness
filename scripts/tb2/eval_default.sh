#!/usr/bin/env bash
# Default cell (TB2.1): a human-designed baseline harness (Terminus-Kira), Sonnet-4.5 student via harbor.
# Usage: VERTEXAI_PROJECT=... TB2_ENGINE_DIR=... bash scripts/tb2/eval_default.sh
# For the Terminus-2 baseline instead: AGENT=agents.baseline_terminus2:AgentHarness bash scripts/tb2/eval_default.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
# shellcheck source=/dev/null
source scripts/tb2/_env.sh

python -m turbo_harness.terminal_bench.eval \
  --split test \
  --agent "${AGENT:-agents.baseline_kira:AgentHarness}" \
  --model "$STUDENT_MODEL" \
  --concurrency "${CONCURRENCY:-32}"
