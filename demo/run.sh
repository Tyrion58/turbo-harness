#!/usr/bin/env bash
# One-command runner for the Turbo Harness live demo.
#
# It (a) serves the trained editor on vLLM (unless you pass a frontier ADVISOR_MODEL, or one is
# already up), waits for it, then (b) runs demo/run_demo.py. The demo scores the single instance
# IN-PROCESS (local Docker), so no eval server is needed. The editor server THIS script starts is
# stopped on exit (set KEEP_SERVERS=1 to leave it up for repeated runs).
#
# Everything is configured by env vars (all overridable):
#   PYTHON        python with the project deps        (default: python)
#   VLLM_PYTHON   python with vLLM, to serve editor   (default: python)
#   EDITOR_CKPT   /path/to/<run>/global_step_N/policy (serve the trained editor)   -- OR --
#   ADVISOR_MODEL a frontier editor id, e.g. vertex_ai/claude-opus-4-6 (skips local serving)
#   STUDENT       execution model     (default: vertex_ai/claude-haiku-4-5)
#   INSTANCE      instance_id         (default: first in DATA_FILE)
#   WITH_DEFAULT  1=also run Default for contrast (default: 1)
#   SPEED         pause multiplier; 0=none, ~1.2 for recording (default: 0)
#   HARNESS_DIR   default: artifacts/swe_smith/haiku/general
#   PLAYBOOK      default: artifacts/swe_smith/haiku/playbook.json
#   DATA_FILE     default: data/swe_smith/test_multi_repo.json
#   REPO_FLAG     multi_repo | verified   (default: multi_repo)
#   GPU           GPU index for the editor (default: 0)  -- set to a FREE gpu
#   EDITOR_PORT   default 8120
#   OUT           if set, also tee the run's output here (for demo/render.sh)
#   KEEP_SERVERS  1 = don't stop servers on exit
# Rootless-Docker users: also export DOCKER_HOST / MSWEA_DOCKER_EXECUTABLE before running.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"

: "${PYTHON:=python}"
: "${VLLM_PYTHON:=python}"
: "${EDITOR_CKPT:=}"
: "${ADVISOR_MODEL:=}"
: "${STUDENT:=vertex_ai/claude-haiku-4-5}"
: "${INSTANCE:=}"
: "${WITH_DEFAULT:=1}"
: "${SPEED:=0}"
: "${HARNESS_DIR:=$REPO/artifacts/swe_smith/haiku/general}"
: "${PLAYBOOK:=$REPO/artifacts/swe_smith/haiku/playbook.json}"
: "${DATA_FILE:=$REPO/data/swe_smith/test_multi_repo.json}"
: "${REPO_FLAG:=multi_repo}"
: "${GPU:=0}"
: "${EDITOR_PORT:=8120}"
: "${KEEP_SERVERS:=0}"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

EDITOR_PID=""
cleanup() {
  if [ "$KEEP_SERVERS" = 1 ]; then
    echo "[demo] KEEP_SERVERS=1 -> leaving editor up (:$EDITOR_PORT)"; return
  fi
  [ -n "$EDITOR_PID" ] && { echo "[demo] stopping editor (pid $EDITOR_PID)"; kill "$EDITOR_PID" 2>/dev/null; }
}
trap cleanup EXIT
port_open() { ss -ltn 2>/dev/null | grep -q ":$1\b"; }

# --- editor sanity: need a checkpoint to serve OR a frontier model ---
if [ -z "$ADVISOR_MODEL" ] && [ -z "$EDITOR_CKPT" ]; then
  echo "[demo] ERROR: set EDITOR_CKPT=/path/to/global_step_N/policy (serve the trained editor)" >&2
  echo "               OR ADVISOR_MODEL=<frontier id, e.g. vertex_ai/claude-opus-4-6> (API editor)" >&2
  exit 1
fi

# --- scoring: the demo scores this single instance IN-PROCESS (local Docker) ---
# run_demo.py unsets EVAL_SERVER_URL itself, so no eval server is started or needed.
echo "[demo] scoring in-process (local Docker) -- no eval server needed"

# --- editor: serve the trained checkpoint, unless a frontier ADVISOR_MODEL is given ---
EDITOR_ARGS=()
if [ -n "$ADVISOR_MODEL" ]; then
  echo "[demo] editor = frontier model $ADVISOR_MODEL (no local serving)"
  EDITOR_ARGS=(--advisor-model "$ADVISOR_MODEL")
else
  if port_open "$EDITOR_PORT"; then
    echo "[demo] editor already served on :$EDITOR_PORT (reusing)"
  else
    [ -f "$EDITOR_CKPT/config.json" ] || { echo "[demo] ERROR: no checkpoint at $EDITOR_CKPT" >&2; exit 1; }
    echo "[demo] serving editor from $EDITOR_CKPT on GPU $GPU :$EDITOR_PORT (loads in ~2-4 min) ..."
    CUDA_VISIBLE_DEVICES="$GPU" "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
      --model "$EDITOR_CKPT" --served-model-name advisor --port "$EDITOR_PORT" \
      --dtype bfloat16 --gpu-memory-utilization 0.85 --max-model-len 32768 >/tmp/demo_editor_vllm.log 2>&1 &
    EDITOR_PID=$!
    for _ in $(seq 1 90); do grep -qi 'Application startup complete' /tmp/demo_editor_vllm.log 2>/dev/null && break; sleep 5; done
    grep -qi 'Application startup complete' /tmp/demo_editor_vllm.log 2>/dev/null \
      || { echo "[demo] ERROR: editor did not come up (see /tmp/demo_editor_vllm.log)" >&2; exit 1; }
  fi
  EDITOR_ARGS=(--advisor-url "http://127.0.0.1:$EDITOR_PORT/v1" --advisor-model openai/advisor)
fi

# --- assemble + run ---
ARGS=(--student "$STUDENT" --repo "$REPO_FLAG" --harness-dir "$HARNESS_DIR"
      --playbook "$PLAYBOOK" --data-file "$DATA_FILE" --speed "$SPEED" "${EDITOR_ARGS[@]}")
[ -n "$INSTANCE" ] && ARGS+=(--instance "$INSTANCE")
[ "$WITH_DEFAULT" = 1 ] && ARGS+=(--with-default)

echo "[demo] run_demo.py ${ARGS[*]}"
echo
if [ -n "${OUT:-}" ]; then
  "$PYTHON" "$REPO/demo/run_demo.py" "${ARGS[@]}" | tee "$OUT"
  echo "[demo] captured to $OUT  (render with: bash demo/render.sh $OUT)"
else
  "$PYTHON" "$REPO/demo/run_demo.py" "${ARGS[@]}"
fi
