#!/usr/bin/env bash
# Shared environment for turbo-harness SWE pipelines. `source` this from the other scripts.
# Values come from your shell / .env with sensible defaults (see .env.example). Override as needed.

: "${VERTEXAI_PROJECT:?set VERTEXAI_PROJECT (copy .env.example -> .env and edit)}"
: "${VERTEXAI_LOCATION:=us-east5}"
: "${EVAL_SERVER_URL:=http://localhost:5152}"
: "${AGENT_SERVER_URL:=http://localhost:8081}"
export VERTEXAI_PROJECT VERTEXAI_LOCATION EVAL_SERVER_URL AGENT_SERVER_URL
export PYTHONUNBUFFERED=1

# Frontier model for Stage A/B (Meta-Harness proposer + playbook curation).
: "${TURBO_FRONTIER_MODEL:=claude-sonnet-4-5}"
export TURBO_FRONTIER_MODEL

# Execution ("student") model, selected by STUDENT={haiku|gemini37}.
STUDENT="${STUDENT:-haiku}"
case "$STUDENT" in
  haiku)    : "${STUDENT_MODEL:=vertex_ai/claude-haiku-4-5}" ;;
  gemini37) : "${STUDENT_MODEL:=vertex_ai/gemini-3.7-flash}" ;;
  *) echo "unknown STUDENT='$STUDENT' (use: haiku | gemini37)"; return 1 2>/dev/null || exit 1 ;;
esac
export STUDENT STUDENT_MODEL

# Coding benchmark selector: BENCH={smith|verified}. Sets harness/data paths + the --repo flag.
BENCH="${BENCH:-smith}"
case "$BENCH" in
  smith)
    export REPO_FLAG="multi_repo"
    export ART="artifacts/swe_smith/${STUDENT}"            # holds general/ and playbook.json
    export DATA_TEST="data/swe_smith/test_multi_repo.json"
    export DATA_TRAIN="data/swe_smith/train_multi_repo.json"
    export DATA_VAL="" ;;
  verified)
    export REPO_FLAG="verified"
    export ART="artifacts/verified/${STUDENT}"
    export DATA_TEST="data/swe_smith/test_verified.json"
    export DATA_TRAIN="data/swe_smith/train_verified.json"
    export DATA_VAL="data/swe_smith/val_verified.json" ;;
  *) echo "unknown BENCH='$BENCH' (use: smith | verified)"; return 1 2>/dev/null || exit 1 ;;
esac
export BENCH

# NOTE (node-specific, rootless Docker): you may also need to export, e.g.
#   export DOCKER_HOST="unix:///run/user/$(id -u)/docker.sock"
#   export MSWEA_DOCKER_EXECUTABLE="$HOME/bin/docker"
# See docs/DEPENDENCIES.md.
