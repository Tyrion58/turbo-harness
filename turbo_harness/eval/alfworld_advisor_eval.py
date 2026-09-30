"""Evaluate the RL-trained ALFWorld PATCH advisor (Turbo Harness) on the held-out test set.

Mirrors the ScienceWorld patch-advisor eval + deployment: for each test game, build the SAME prompt
used in RL training (FULL general-harness source + playbook + task) -> query the served advisor ->
extract SEARCH/REPLACE edits -> apply to a copy of the general harness -> validate. If the patch
applies AND validates, run the PATCHED harness; otherwise FALL BACK to the base general harness (the
realistic deployment — never run a broken harness). The outcome comes from the env (trusted_score),
not the harness dict. Reports pass rate + per-task-type vs the default and general-harness baselines.

Run in the dev venv (imports the training prompt for fidelity); needs the ADVISOR served (OpenAI
endpoint) and the STUDENT served on :8110 with tool-calling enabled. task_descs are dumped via the
alfworld worker venv.

  PYTHONPATH=$PWD .venv/bin/python -m turbo_harness.eval.alfworld_advisor_eval \
    --advisor-url http://127.0.0.1:8120/v1 --advisor-model advisor \
    --harness-dir artifacts/alfworld/qwen3-5-9b/alf_stage1_sonnet/general \
    --split harness_r1_eval \
    --playbook experiments/logs/alfworld_meta_harness/alf_stage1_sonnet/playbook.json \
    --count 150 --concurrency 8
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from turbo_harness.playbook.memory_advisor import format_playbook_context  # noqa: E402
from turbo_harness.playbook.schemas import Playbook  # noqa: E402
from turbo_harness.patch_advisor import _extract_edits, apply_patch  # noqa: E402
from turbo_harness.eval import alfworld_executor as ALFX  # noqa: E402
from turbo_harness.alfworld.runner import _games  # noqa: E402
from turbo_harness.rl.build_alfworld_rl_dataset import (  # noqa: E402
    DEFAULT_HARNESS_DIR,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    _dump_task_descs,
    _load_harness_code,
)

# Qwen3.5 sampling best-practice (thinking, precise): temp 0.6, top_p 0.95, top_k 20. NEVER temp 0
# (greedy -> endless repetition). The patch task terminates within max_tokens; extraction handles think.
DEFAULT_TEMPERATURE = 0.6
DEFAULT_MAX_TOKENS = 16384  # must match RL training max_generate_length, else the trained
# advisor's (longer) patches truncate at eval -> base-fallback -> Turbo understated.

# Baselines to contextualize the Turbo number (fill from the Sonnet ladder before/after a run).
DEFAULT_HARNESS_RATE = 38.0   # default harness held-out pass rate (%)
GENERAL_HARNESS_RATE = 90.0   # meta-harness general harness held-out pass rate (%)

# ALFWorld task-type prefix (first segment of the game dir) -> short label, for per-type aggregation.
_ALF_TYPE = {
    "pick_and_place_simple": "pick_and_place",
    "look_at_obj_in_light": "look_at",
    "pick_clean_then_place_in_recep": "clean",
    "pick_heat_then_place_in_recep": "heat",
    "pick_cool_then_place_in_recep": "cool",
    "pick_two_obj_and_place": "pick_two",
}


def _task_type(game_relpath: str) -> str:
    parts = game_relpath.split("/")
    seg = parts[2] if len(parts) > 2 else (parts[-1] if parts else game_relpath)
    prefix = seg.split("-", 1)[0]
    return _ALF_TYPE.get(prefix, prefix)


def _advisor_raw(url, model, harness_code, playbook_ctx, task_desc,
                 max_tokens=DEFAULT_MAX_TOKENS, temperature=DEFAULT_TEMPERATURE):
    user = USER_TEMPLATE.format(harness_code=harness_code,
                                playbook=playbook_ctx or "(none)", task_desc=task_desc)
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        "temperature": temperature, "max_tokens": max_tokens, "seed": 0,
        "top_p": 0.95, "top_k": 20,
    }).encode()
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=600) as r:
            d = json.loads(r.read())
        return d["choices"][0]["message"].get("content") or ""
    except Exception as e:  # noqa: BLE001
        return f"(advisor_error:{type(e).__name__})"


def _prepare_harness(raw, base_dir, out_dir):
    """Apply the advisor's patch; fall back to the base harness on empty/failed/invalid patch.

    Returns (harness_dir_to_run, num_edits, status)."""
    if raw.startswith("(advisor_error"):
        return base_dir, 0, "advisor_error->base"
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[1]
    edits = _extract_edits(raw)
    if not edits:
        return base_dir, 0, "no_patch->base"  # NO_PATCH_NEEDED or nothing parseable
    success, _ = apply_patch(edits, base_dir, out_dir)
    if not success:
        return base_dir, len(edits), "apply_failed->base"
    ok, msg = ALFX.validate_harness(str(out_dir))
    if not ok:
        return base_dir, len(edits), f"invalid->base:{msg[:40]}"
    safe, smsg = ALFX.reward_safe(str(out_dir))  # same reward-integrity guard as the RL env
    if not safe:
        return base_dir, len(edits), f"unsafe->base:{smsg[:40]}"
    return str(out_dir), len(edits), "patched"


def main():
    ap = argparse.ArgumentParser(description="Eval RL patch-advisor on ALFWorld test")
    ap.add_argument("--advisor-url", required=True)
    ap.add_argument("--advisor-model", default="advisor")
    ap.add_argument("--harness-dir", default=DEFAULT_HARNESS_DIR,
                    help="Base general harness the advisor patches (must match training).")
    ap.add_argument("--split", default="harness_r1_eval", help="split NAME (ALF_DATA/<split>.json)")
    ap.add_argument("--playbook",
                    default="artifacts/alfworld/playbook.json")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=150)
    ap.add_argument("--concurrency", type=int, default=8, help="parallel student rollouts")
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--advisor-workers", type=int, default=16)
    ap.add_argument("--out", default="results/alfworld_advisor.jsonl")
    ap.add_argument("--wandb-project", default="turbo-harness",
                    help="unified wandb project; set to 'none' to disable eval logging")
    ap.add_argument("--wandb-group", default="alfworld", help="wandb group (domain)")
    ap.add_argument("--wandb-run-name", default=None, help="wandb run name (default: --out stem)")
    args = ap.parse_args()

    run = None
    if args.wandb_project and args.wandb_project.lower() != "none":
        try:
            import wandb
            run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or Path(args.out).stem,
                group=args.wandb_group,
                job_type="eval",
                tags=[args.wandb_group, "patch", "eval"],
                config={"split": args.split, "harness_dir": args.harness_dir,
                        "advisor_model": args.advisor_model, "count": args.count,
                        "max_steps": args.max_steps, "method": "patch-advisor"},
            )
        except Exception as e:  # noqa: BLE001
            print(f"(wandb disabled: {type(e).__name__}: {e})")
            run = None

    games = _games(args.split)
    idxs = list(range(args.start, min(args.start + args.count, len(games))))
    harness_code = _load_harness_code(args.harness_dir)
    if args.playbook and not Path(args.playbook).exists():
        raise FileNotFoundError(
            f"--playbook {args.playbook} not found (a silent empty playbook would gut the "
            f"method). Pass --playbook '' to intentionally run a no-playbook ablation."
        )
    playbook_ctx = (
        format_playbook_context(Playbook.load(args.playbook)) if args.playbook else ""
    )
    print(f"dumping task_descs for {len(games)} games (worker venv)...", flush=True)
    descs = _dump_task_descs(args.split)

    # 1) advisor patch (raw output) per game, concurrent
    print(f"querying advisor for {len(idxs)} test games...", flush=True)

    def q(i):
        return i, _advisor_raw(args.advisor_url, args.advisor_model, harness_code,
                               playbook_ctx, descs.get(i, ""))

    raws = {}
    with ThreadPoolExecutor(max_workers=args.advisor_workers) as pool:
        for i, raw in pool.map(q, idxs):
            raws[i] = raw

    # 2) apply/validate each patch into a per-game dir (fall back to base on failure)
    tmp_root = Path(tempfile.mkdtemp(prefix="alf_patch_eval_"))
    prep = {}  # i -> (harness_dir, num_edits, status)
    for i in idxs:
        prep[i] = _prepare_harness(raws[i], args.harness_dir, tmp_root / f"inst_{i}")
    n_patched = sum(1 for v in prep.values() if v[2] == "patched")
    print(f"  patches applied+validated: {n_patched}/{len(idxs)} (rest fall back to base harness)")

    # 3) run each (patched or base) harness on its game; trusted env outcome
    def run_one(i):
        hdir, nedits, pstatus = prep[i]
        try:
            results, statuses, _ = ALFX.eval_harness(
                hdir, [i], split=args.split, student="Qwen3.5-9B",
                trials=1, concurrency=1, max_steps=args.max_steps,
                harness_name=f"patcheval_{i}", trusted_score=True, cleanup=True,
            )
            won = float(results.get(i, [0.0])[0])
            rstatus = statuses[0] if statuses else "?"
            steps = int(ALFX._LAST_STEPS.get(i, 0))
        except Exception as e:  # noqa: BLE001
            won, rstatus, steps = 0.0, f"eval_error:{type(e).__name__}", 0
        return {"index": i, "game_relpath": games[i], "task_type": _task_type(games[i]),
                "won": int(won >= 1.0), "num_edits": nedits,
                "patch_status": pstatus, "run_status": rstatus, "steps": steps}

    print("running (patched/base) harnesses on the student...", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for n, row in enumerate(pool.map(run_one, idxs), 1):
            rows.append(row)
            if n % 10 == 0:
                m = 100 * sum(r["won"] for r in rows) / len(rows)
                print(f"[{n}/{len(idxs)}] running pass_rate={m:.1f}%", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    shutil.rmtree(tmp_root, ignore_errors=True)

    # 4) aggregate
    by = collections.defaultdict(list)
    for r in rows:
        by[r["task_type"]].append(int(r["won"]))
    allv = [v for vs in by.values() for v in vs]
    rate = 100 * sum(allv) / max(1, len(allv))
    print(f"\n{'task type':24s} {'pass%':>8s} {'n':>5s}")
    for t in sorted(by):
        print(f"  {t:22s} {100 * sum(by[t]) / len(by[t]):7.0f} {len(by[t]):5d}")
    _st = [r["steps"] for r in rows if r.get("steps")]
    avg_steps = sum(_st) / len(_st) if _st else 0.0
    print(f"\n=== TURBO HARNESS (RL patch advisor) TEST: {rate:.1f}% pass avg_steps={avg_steps:.1f} "
          f"over {len(allv)} games ({n_patched} patched, {len(idxs) - n_patched} base-fallback) ===")
    print(f"   baselines: default harness {DEFAULT_HARNESS_RATE:.1f}% | "
          f"general (meta-harness) harness {GENERAL_HARNESS_RATE:.1f}%")

    if run is not None:
        run.log({
            "eval/pass_rate": rate,
            "eval/apply_rate": n_patched / max(1, len(idxs)),
            "eval/n_patched": n_patched,
            "eval/n_games": len(allv),
            "eval/delta_vs_general": rate - GENERAL_HARNESS_RATE,
            "eval/delta_vs_default": rate - DEFAULT_HARNESS_RATE,
            **{f"task/{t}": 100 * sum(by[t]) / len(by[t]) for t in by},
        })
        run.summary["pass_rate"] = rate
        run.finish()


if __name__ == "__main__":
    main()
