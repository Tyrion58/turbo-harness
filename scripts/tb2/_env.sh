#!/usr/bin/env bash
# Shared TB2.1 settings. TB2 drives the harbor engine (terminal-bench) with a FROZEN Claude Sonnet 4.5
# student on Vertex. Needs Docker + tmux on the host. See docs/DEPENDENCIES.md.
: "${VERTEXAI_PROJECT:?set VERTEXAI_PROJECT (Sonnet student via Vertex; see .env.example)}"
: "${VERTEXAI_LOCATION:=global}"
export VERTEXAI_PROJECT VERTEXAI_LOCATION
# harbor engine: set TB2_ENGINE_DIR to a terminal_bench_2 project (harbor in its .venv), OR set
# HARBOR_BIN to a `harbor` on PATH (from `pip install -e '.[tb2]'`).
if [ -z "${TB2_ENGINE_DIR:-}" ] && [ -z "${HARBOR_BIN:-}" ]; then
  echo "set TB2_ENGINE_DIR (harbor project dir) or HARBOR_BIN (harbor on PATH) — see docs/DEPENDENCIES.md" >&2
  exit 1
fi
export TB2_ENGINE_DIR="${TB2_ENGINE_DIR:-}"
: "${STUDENT_MODEL:=vertex_ai/claude-sonnet-4-5}"; export STUDENT_MODEL
export PYTHONUNBUFFERED=1
