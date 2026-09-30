#!/usr/bin/env bash
# Meta-Harness cell (TB2.1): the evolution winner harness (agents/kira_auto_test.py), Sonnet-4.5 student.
# Usage: VERTEXAI_PROJECT=... TB2_ENGINE_DIR=... bash scripts/tb2/eval_meta.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
# shellcheck source=/dev/null
source scripts/tb2/_env.sh

python -m turbo_harness.terminal_bench.eval \
  --split test \
  --agent agents.kira_auto_test:AgentHarness \
  --model "$STUDENT_MODEL" \
  --concurrency "${CONCURRENCY:-32}"
