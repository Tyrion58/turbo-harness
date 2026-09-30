"""Build a difficulty/category-stratified train/test split for TB2 (deterministic).

The original random seed-42 split was difficulty-imbalanced (test 40% hard vs train 27%,
and — worse — within-bucket unlucky), giving a 50%/27% baseline gap. This partitions the
same 89-task pool so difficulty is balanced exactly (hard 15/15, etc.) and category is
balanced approximately, deterministically. Writes to turbo_harness/terminal_bench/data/.

  python -m turbo_harness.terminal_bench.make_split
"""

from __future__ import annotations

import glob
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
CACHE = Path(os.path.expanduser("~/.cache/harbor"))
# source pool = the 89 tasks we use (union of the original reference split)
SRC = (
    HERE.parents[1]
    / "meta-harness"
    / "reference_examples"
    / "terminal_bench_2"
    / "data"
)
DIFF_ORDER = {"easy": 0, "medium": 1, "hard": 2}


def task_meta() -> dict[str, tuple[str, str]]:
    meta: dict[str, tuple[str, str]] = {}
    for f in glob.glob(str(CACHE / "**" / "task.toml"), recursive=True):
        name = os.path.basename(os.path.dirname(f))
        if name in meta:
            continue
        t = open(f, errors="ignore").read()
        diff = (re.search(r'difficulty\s*=\s*"([^"]+)"', t) or [None, None])[1]
        cat = (re.search(r'category\s*=\s*"([^"]+)"', t) or [None, None])[1]
        meta[name] = (diff or "unknown", cat or "unknown")
    return meta


def build_split(
    pool: list[str], meta: dict[str, tuple[str, str]]
) -> tuple[list[str], list[str]]:
    """Stratify by difficulty (exact) and category (approx), deterministic round-robin."""
    by_diff: dict[str, list[str]] = defaultdict(list)
    for t in pool:
        by_diff[meta.get(t, ("unknown", "unknown"))[0]].append(t)
    train, test, toggle = [], [], 0
    for d in sorted(by_diff, key=lambda x: DIFF_ORDER.get(x, 9)):
        # sort within difficulty by category then name → alternation balances categories too
        for t in sorted(by_diff[d], key=lambda t: (meta.get(t, ("", ""))[1], t)):
            (train if toggle % 2 == 0 else test).append(t)
            toggle += 1
    return sorted(train), sorted(test)


def main() -> None:
    pool = sorted(
        set(
            json.loads((SRC / "train_tasks.json").read_text())
            + json.loads((SRC / "test_tasks.json").read_text())
        )
    )
    meta = task_meta()
    train, test = build_split(pool, meta)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train_tasks.json").write_text(json.dumps(train, indent=1) + "\n")
    (OUT / "test_tasks.json").write_text(json.dumps(test, indent=1) + "\n")
    print(f"wrote stratified split -> {OUT}  (train {len(train)} / test {len(test)})")
    for split, ids in [("train", train), ("test", test)]:
        diffs = Counter(meta.get(t, ("?", "?"))[0] for t in ids)
        print(
            f"  {split}: "
            + " ".join(f"{k}={diffs.get(k, 0)}" for k in ["easy", "medium", "hard"])
        )


if __name__ == "__main__":
    main()
