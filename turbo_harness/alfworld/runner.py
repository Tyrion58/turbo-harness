"""ALFWorld episode runner (runs in the alfworld-worker venv, py3.9).

Runs a full-scaffold harness (`<harness_dir>/harness.py` exposing run_episode(env, llm, max_steps))
over a contiguous range of games in a split, concurrently via a process pool (each game gets its own
TextWorld env in its own process = safe). Writes per-game results JSONL and playbook-pipeline trajectories.

  python -m turbo_harness.alfworld.runner --harness-dir <dir|default> \
      --split harness_r1_train --start 0 --count 100 --concurrency 24 --max-steps 50 \
      --out results.jsonl --traj-dir <dir> --harness-name iter1_foo
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ALF_DATA = Path(os.environ.get("ALFWORLD_DATA", "data/alfworld_data"))
DEFAULT_HARNESS_DIR = REPO / "turbo_harness/alfworld/default_harness"
TARGET_URL = "http://127.0.0.1:8110/v1"
TARGET_MODEL = "Qwen3.5-9B"

_HARNESS_CACHE = {}


def _load_harness(harness_dir: str):
    if harness_dir in _HARNESS_CACHE:
        return _HARNESS_CACHE[harness_dir]
    hp = Path(harness_dir) / "harness.py"
    spec = importlib.util.spec_from_file_location(
        f"alfharness_{abs(hash(harness_dir)) % 10**8}", hp
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _HARNESS_CACHE[harness_dir] = mod
    return mod


def _games(split: str) -> list[str]:
    data = json.load(open(ALF_DATA / f"{split}.json"))
    out = []
    for v in data.values():
        out.extend(v)
    return out


def _run_one(
    idx: int,
    game_relpath: str,
    harness_dir: str,
    max_steps: int,
    base_url: str,
    model: str,
    trusted_score: bool = False,
) -> dict:
    from turbo_harness.alfworld.api import LLMClient
    from turbo_harness.alfworld.env import AlfEnv, make_scoring_facade

    real_env = None
    try:
        harness = _load_harness(harness_dir)
        real_env = AlfEnv(game_relpath)
        # trusted_score=True (RL / policy-patched harness): hand the harness a FACADE that cannot
        # touch the outcome, and read the reward from the REAL env below — so policy code cannot fake
        # its reward. Honest/trusted harnesses (trusted-off) get the real env and self-report, and
        # final_won()==their reported won, so those flows are unchanged.
        run_env = make_scoring_facade(real_env) if trusted_score else real_env
        llm = LLMClient(base_url=base_url, model=model, max_tokens=4096)
        res = harness.run_episode(run_env, llm, max_steps)
        won = bool(real_env.final_won()) if trusted_score else bool(res.get("won", False))
        return {
            "index": idx,
            "won": won,
            "steps": int(res.get("steps", 0) or 0),
            "status": str(res.get("status", "?")),
            "messages": res.get("messages", []),
        }
    except BaseException as e:  # noqa: BLE001 - a bad harness must not kill the pool worker
        return {
            "index": idx,
            "won": False,
            "steps": 0,
            "status": f"HARNESS_ERR:{type(e).__name__}:{str(e)[:120]}",
            "messages": [],
            "trace": traceback.format_exc()[:1500],
        }
    finally:
        if real_env is not None:
            real_env.close()


def main():
    ap = argparse.ArgumentParser(description="ALFWorld full-scaffold runner")
    ap.add_argument("--harness-dir", default="default")
    ap.add_argument("--harness-name", default=None)
    ap.add_argument("--split", default="harness_r1_train")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--base-url", default=TARGET_URL)
    ap.add_argument("--model", default=TARGET_MODEL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--traj-dir", default=None)
    ap.add_argument(
        "--trusted-score",
        action="store_true",
        help="Score from the env (AlfEnv.final_won) not the harness dict — for policy-patched harnesses.",
    )
    args = ap.parse_args()

    harness_dir = (
        str(DEFAULT_HARNESS_DIR) if args.harness_dir == "default" else args.harness_dir
    )
    harness_name = args.harness_name or (
        "default" if args.harness_dir == "default" else Path(harness_dir).name
    )
    games = _games(args.split)
    indices = list(range(args.start, min(args.start + args.count, len(games))))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traj_dir = Path(args.traj_dir) if args.traj_dir else None
    if traj_dir:
        traj_dir.mkdir(parents=True, exist_ok=True)

    n_done = n_won = 0
    with (
        open(out_path, "w") as outf,
        ProcessPoolExecutor(max_workers=args.concurrency) as pool,
    ):
        futs = {
            pool.submit(
                _run_one,
                i,
                games[i],
                harness_dir,
                args.max_steps,
                args.base_url,
                args.model,
                args.trusted_score,
            ): i
            for i in indices
        }
        for f in as_completed(futs):
            r = f.result()
            n_done += 1
            n_won += int(r["won"])
            outf.write(
                json.dumps({k: v for k, v in r.items() if k != "messages"}) + "\n"
            )
            outf.flush()
            if traj_dir:
                msgs = [
                    {
                        "role": (
                            "assistant"
                            if m.get("role") == "agent"
                            else m.get("role", "?")
                        ),
                        "content": str(m.get("content", ""))[:2000],
                    }
                    for m in r.get("messages", [])
                ]
                rec = {
                    "instance_id": f"{args.split}_{r['index']}",
                    "harness": harness_name,
                    "resolved": bool(r["won"]),
                    "status": r["status"],
                    "steps": r["steps"],
                    "messages": msgs,
                }
                (traj_dir / f"{args.split}_{r['index']}.jsonl").write_text(
                    json.dumps(rec)
                )
            if n_done % 10 == 0:
                print(f"[{n_done}/{len(indices)}] won={n_won}", flush=True)
    print(
        f"DONE {n_won}/{len(indices)} = {100 * n_won / max(1, len(indices)):.1f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
