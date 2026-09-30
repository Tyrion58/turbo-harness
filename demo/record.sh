#!/usr/bin/env bash
# Record demo/run_demo.py to an animated GIF (assets/demo.gif) for the README.
# Needs: asciinema (recorder) + agg (asciinema GIF generator).
#   pip install asciinema
#   agg:  https://github.com/asciinema/agg  (cargo install --git https://github.com/asciinema/agg, or a release binary)
# Run this on the node where the runtime is up (eval_server + Docker + served editor + Vertex creds).
set -u
cd "$(dirname "$0")/.." || exit 1
# Editor: set ADVISOR_URL for a served vLLM editor, OR put --advisor-model <frontier> in DEMO_ARGS.
if [ -n "${ADVISOR_URL:-}" ]; then EDITOR_ARG="--advisor-url $ADVISOR_URL"; else EDITOR_ARG=""; fi
CAST="${CAST:-/tmp/turbo_demo.cast}"
OUT="${OUT:-assets/demo.gif}"
mkdir -p "$(dirname "$OUT")"

# Record the real run. Add DEMO_ARGS="--with-default" for a Default-vs-Turbo contrast;
# --speed pads the pauses so the GIF is readable.
asciinema rec "$CAST" --overwrite --cols 100 --rows 32 \
  -c "python demo/run_demo.py $EDITOR_ARG --speed 1.2 ${DEMO_ARGS:-}"

# Convert the cast to a GIF.
agg --font-size 16 --theme asciinema "$CAST" "$OUT"
echo "wrote $OUT   (embed in README with:  ![demo]($OUT) )"
