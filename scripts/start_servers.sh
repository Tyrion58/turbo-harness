#!/usr/bin/env bash
# Start the SWE agent + eval servers (Default eval needs both; Meta/Turbo need eval_server).
# Requires the dev venv active and a working Docker. Backgrounds both; Ctrl-C stops them.
set -u
cd "$(dirname "$0")/.." || exit 1
: "${AGENT_PORT:=8081}"
: "${EVAL_PORT:=5152}"
echo "eval_server  -> :$EVAL_PORT"
python -m turbo_harness.infra.eval_server  --host 0.0.0.0 --port "$EVAL_PORT" &
echo "agent_server -> :$AGENT_PORT"
python -m turbo_harness.infra.agent_server --host 0.0.0.0 --port "$AGENT_PORT" &
wait
