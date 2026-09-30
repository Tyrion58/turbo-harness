#!/usr/bin/env bash
# Render a captured demo run into assets/demo.{svg,png} for the README.
# Capture a run first:  OUT=/tmp/run.txt bash demo/run.sh   (then:)  bash demo/render.sh /tmp/run.txt
# Needs a PYTHON with `rich` (SVG); the PNG additionally needs `cairosvg` (pip install cairosvg).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
IN="${1:?usage: bash demo/render.sh <captured-run.txt>}"
: "${PYTHON:=python}"
: "${SVG:=assets/demo.svg}"
: "${PNG:=assets/demo.png}"
: "${WIDTH:=104}"
: "${TITLE:=Turbo Harness — example run}"
mkdir -p "$(dirname "$SVG")"

"$PYTHON" - "$IN" "$SVG" "$WIDTH" "$TITLE" <<'PY'
import sys
from pathlib import Path
from rich.console import Console
from rich.text import Text
inp, svg, width, title = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
con = Console(record=True, width=width)
con.print(Text.from_ansi(Path(inp).read_text().strip("\n")))
con.save_svg(svg, title=title)
print("wrote", svg)
PY

if "$PYTHON" -c "import cairosvg" 2>/dev/null; then
  "$PYTHON" -c "import cairosvg; cairosvg.svg2png(url='$SVG', write_to='$PNG', output_width=1300); print('wrote $PNG')"
else
  echo "[render] cairosvg not installed -> SVG only. For a PNG:  $PYTHON -m pip install cairosvg  then re-run."
fi
