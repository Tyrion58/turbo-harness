#!/usr/bin/env bash
# Shared ALFWorld settings. ALFWorld runs a FROZEN Qwen3.5-9B student served via vLLM (tool-calling)
# over the alfworld game env. `source` this from the other ALFWorld scripts.
# Set ALFWORLD_DATA to your ALFWorld game-data download (see docs/DEPENDENCIES.md).
: "${ALFWORLD_DATA:?set ALFWORLD_DATA to your ALFWorld game-data dir (see docs/DEPENDENCIES.md)}"
export ALFWORLD_DATA
# Interpreter that can import `alfworld` + TextWorld (a dedicated worker venv, or your current one):
: "${ALFWORLD_WORKER_PY:=$(command -v python)}"; export ALFWORLD_WORKER_PY
# Frozen student endpoint (serve with scripts/alfworld/serve_student.sh):
: "${STUDENT_URL:=http://127.0.0.1:8110/v1}"; export STUDENT_URL
: "${STUDENT_MODEL:=Qwen3.5-9B}"; export STUDENT_MODEL
export ART_ALF="artifacts/alfworld"                      # general/ (harness.py) + playbook.json
export TEST_SPLIT="${TEST_SPLIT:-harness_r1_eval}"       # 150 held-out (valid_unseen) test games
export PYTHONUNBUFFERED=1
