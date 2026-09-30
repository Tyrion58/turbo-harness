#!/usr/bin/env python3
"""Turn a captured demo run into an animated GIF for the README.

Capture a run first, then render:
    OUT=/tmp/run.txt bash demo/run.sh          # capture one real run's ANSI output
    python demo/make_gif.py /tmp/run.txt       # -> assets/demo.gif

It replays the *real* captured output as a top-down screencast (no re-run), holding on
the editor "generating patch" and student "running" steps to mimic real latency.

Deps (same family as render.sh, plus Pillow):  pip install rich cairosvg pillow
Env overrides: OUT (out path), COLS (terminal width), PNG_W (px), TITLE.
"""
import io
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.text import Text
import cairosvg
from PIL import Image

SRC = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: python demo/make_gif.py <captured-run.txt>")
OUT = os.environ.get("OUT", str(Path(__file__).resolve().parent.parent / "assets" / "demo.gif"))
COLS = int(os.environ.get("COLS", "106"))     # terminal columns (match render.sh; avoid wrapping)
PNG_W = int(os.environ.get("PNG_W", "960"))   # output width in px
TITLE = os.environ.get("TITLE", "Turbo Harness — example run")
CHUNK = int(os.environ.get("CHUNK", "3"))     # lines revealed per frame

lines = Path(SRC).read_text().rstrip("\n").split("\n")
N = len(lines)

# reveal plan: (lines_shown, duration_ms); hold on the slow steps, long hold at the end
steps, i = [], 0
while i < N:
    j = min(N, i + CHUNK)
    newly = "\n".join(lines[i:j])
    dur = 300
    if "generating patch" in newly:
        dur = 1700
    elif "running ..." in newly or "running Default" in newly:
        dur = 1500
    steps.append((j, dur))
    i = j
steps[-1] = (N, 3200)


def frame(nshown):
    body = lines[:nshown] + [""] * (N - nshown)   # pad -> constant canvas size across frames
    con = Console(record=True, width=COLS, file=io.StringIO())
    con.print(Text.from_ansi("\n".join(body)))
    svg = con.export_svg(title=TITLE)
    png = cairosvg.svg2png(bytestring=svg.encode(), output_width=PNG_W)
    return Image.open(io.BytesIO(png)).convert("RGB")


imgs = [frame(n) for (n, _) in steps]
pal = [im.convert("P", palette=Image.ADAPTIVE, colors=128) for im in imgs]
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
pal[0].save(OUT, save_all=True, append_images=pal[1:], duration=[d for (_, d) in steps],
            loop=0, optimize=True, disposal=1)
print(f"wrote {OUT}  ({len(imgs)} frames, {Path(OUT).stat().st_size / 1024:.0f} KB, {imgs[0].size[0]}x{imgs[0].size[1]})")
