"""Evaluate the RL-trained Terminal-Bench-2 PATCH advisor (Turbo Harness) on the held-out TEST split.

Mirrors the ScienceWorld patch-advisor eval: for each test task, build the SAME prompt used in RL
training (FULL winner-harness source + the TB2 task instruction) -> query the served advisor ->
extract SEARCH/REPLACE edits -> apply to a copy of the winner harness -> validate + reward-safe. If
the patch applies AND validates, run the PATCHED harness via harbor + Sonnet; otherwise FALL BACK to
the base (unpatched) winner harness (the realistic deployment — never run a broken harness). Reward =
harbor's trusted verifier reward.txt (binary 0/1). Reports the pass RATE vs the base winner (62.2%
train) baseline.

`--mode base` runs the unpatched winner on every task (to measure the base rate under identical
conditions for a clean Δ). Needs the ADVISOR served (OpenAI endpoint) for `--mode advisor`.

  PYTHONPATH=$PWD .venv/bin/python -m turbo_harness.eval.tb2_advisor_eval \
    --advisor-url http://127.0.0.1:8120/v1 --advisor-model advisor \
    --split turbo_harness/terminal_bench/data/test_tasks.json \
    --concurrency 8 --out /tmp/tb2_test_advisor.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from turbo_harness.patch_advisor import _extract_edits  # noqa: E402
from turbo_harness.eval import tb2_executor as TBX  # noqa: E402
from turbo_harness.playbook.memory_advisor import format_playbook_context  # noqa: E402
from turbo_harness.playbook.schemas import Playbook  # noqa: E402
from turbo_harness.rl.build_tb2_rl_dataset import (  # noqa: E402
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    WINNER_AGENT,
    _task_instructions,
)

# Qwen3.5 sampling best-practice (thinking, precise): temp 0.6, top_p 0.95, top_k 20. NEVER temp 0
# (greedy -> endless repetition). Match RL training max_generate_length (8192).
DEFAULT_TEMPERATURE = 0.6
DEFAULT_MAX_TOKENS = 8192
BASE_HARNESS_DIR = str(TBX.BASE_HARNESS_DIR)  # unpatched winner (harbor-runnable as-is)


def _advisor_raw(
    url,
    model,
    harness_code,
    task,
    task_desc,
    playbook="(none)",
    max_tokens=DEFAULT_MAX_TOKENS,
    temperature=DEFAULT_TEMPERATURE,
):
    """Served (vLLM/OpenAI) advisor endpoint (the RL-Qwen)."""
    user = USER_TEMPLATE.format(
        harness_code=harness_code, playbook=playbook, task=task, task_desc=task_desc
    )
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": 0,
            "top_p": 0.95,
            "top_k": 20,
        }
    ).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=900) as r:
            d = json.loads(r.read())
        return d["choices"][0]["message"].get("content") or ""
    except Exception as e:  # noqa: BLE001
        return f"(advisor_error:{type(e).__name__})"


# Reconstruct the ORIGINAL (pre-playbook) prompt for the no-playbook control (removes exactly the
# playbook additions), so the control matches the exact advisor role the RL-Qwen was trained in.
_PB_PARA = (
    "You are given a LEARNED PLAYBOOK of scaffold strategies and ANTI-PATTERNS distilled "
    "from prior harness experiments on this benchmark. Apply its strategies when they fit "
    "THIS task's kind, and HEED its anti-patterns. IMPORTANT: the base harness already solves "
    "MOST tasks — if it already handles this task's kind and no specific failure mode applies, "
    "output NO_PATCH_NEEDED rather than making speculative edits. Over-patching an "
    "already-working harness (especially rewriting the completion gate, stall logic, or core "
    "loop) is the most common way to REGRESS.\n\n"
)
_PB_LINE = "## Learned playbook (scaffold strategies + anti-patterns)\n{playbook}\n\n"
ORIG_SYSTEM = SYSTEM_PROMPT.replace(_PB_PARA, "")
ORIG_USER = USER_TEMPLATE.replace(_PB_LINE, "")


def _advisor_vertex(
    model,
    harness_code,
    task,
    task_desc,
    playbook="(none)",
    orig_prompt=False,
    max_tokens=DEFAULT_MAX_TOKENS,
    temperature=DEFAULT_TEMPERATURE,
):
    """Frontier advisor via litellm/Vertex (e.g. opus-4.6) — used for the Stage-3 gate + control.
    orig_prompt=True uses the pre-playbook prompt (the no-playbook control that matches RL-Qwen)."""
    import litellm

    litellm.drop_params = True
    for var, attr in (
        ("VERTEXAI_PROJECT", "vertex_project"),
        ("VERTEXAI_LOCATION", "vertex_location"),
    ):
        if os.environ.get(var):
            setattr(litellm, attr, os.environ[var])
    system = ORIG_SYSTEM if orig_prompt else SYSTEM_PROMPT
    tmpl = ORIG_USER if orig_prompt else USER_TEMPLATE
    user = tmpl.format(
        harness_code=harness_code, playbook=playbook, task=task, task_desc=task_desc
    )
    try:
        resp = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:  # noqa: BLE001
        return f"(advisor_error:{type(e).__name__})"


def _prepare_harness(raw, out_dir):
    """Apply the advisor's patch into a fresh harness copy; fall back to the base winner on
    empty/failed/invalid/unsafe patch. Returns (harness_dir_to_run, num_edits, status)."""
    if raw.startswith("(advisor_error"):
        return BASE_HARNESS_DIR, 0, "advisor_error->base"
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[1]
    if "NO_PATCH_NEEDED" in raw and not _extract_edits(raw):
        return BASE_HARNESS_DIR, 0, "no_patch->base"
    edits = _extract_edits(raw)
    if not edits:
        return BASE_HARNESS_DIR, 0, "no_patch->base"
    TBX.make_harness_copy(out_dir)
    agent_path = Path(out_dir) / "agents" / TBX.AGENT_FILE
    content = agent_path.read_text()
    applied = 0
    for search, replace in edits:
        if search in content:
            content = content.replace(search, replace, 1)
            applied += 1
    if applied == 0:
        return BASE_HARNESS_DIR, len(edits), "apply_failed->base"
    agent_path.write_text(content)
    ok, msg = TBX.validate_harness(str(out_dir))
    if not ok:
        return BASE_HARNESS_DIR, len(edits), f"invalid->base:{msg[:40]}"
    safe, smsg = TBX.reward_safe(str(out_dir))
    if not safe:
        return BASE_HARNESS_DIR, len(edits), f"unsafe->base:{smsg[:40]}"
    return str(out_dir), len(edits), "patched"


def main():
    ap = argparse.ArgumentParser(
        description="Eval RL patch-advisor on Terminal-Bench-2 test"
    )
    ap.add_argument(
        "--advisor-url", default=None, help="OpenAI endpoint of the served advisor"
    )
    ap.add_argument("--advisor-model", default="advisor")
    ap.add_argument(
        "--advisor-backend",
        choices=["endpoint", "vertex"],
        default="endpoint",
        help="endpoint=served vLLM (RL-Qwen); vertex=litellm frontier advisor (gate)",
    )
    ap.add_argument(
        "--playbook",
        default=str(REPO / "artifacts/tb2/playbook.json"),
        help="playbook.json (Stage 2); pass '' for a no-playbook ablation.",
    )
    ap.add_argument(
        "--winner",
        default=str(WINNER_AGENT),
        help="Base winner agent file the advisor patches (must match training).",
    )
    ap.add_argument(
        "--split", default=str(REPO / "turbo_harness/terminal_bench/data/test_tasks.json")
    )
    ap.add_argument(
        "--mode",
        choices=["advisor", "base"],
        default="advisor",
        help="advisor: patch per task (fallback base); base: run unpatched winner on all.",
    )
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=999)
    ap.add_argument(
        "--concurrency", type=int, default=8, help="parallel harbor rollouts"
    )
    ap.add_argument("--max-turns", type=int, default=300)
    ap.add_argument("--advisor-workers", type=int, default=16)
    ap.add_argument("--student", default="vertex_ai/claude-sonnet-4-5")
    ap.add_argument("--out", default="/tmp/tb2_test_advisor.jsonl")
    ap.add_argument("--wandb-project", default="turbo-harness")
    ap.add_argument("--wandb-group", default="tb2")
    ap.add_argument("--wandb-run-name", default=None)
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
                tags=[args.wandb_group, "patch", "eval", args.mode],
                config={
                    "split": args.split,
                    "winner": args.winner,
                    "mode": args.mode,
                    "advisor_model": args.advisor_model,
                    "student": args.student,
                    "max_turns": args.max_turns,
                    "method": "patch-advisor",
                },
            )
        except Exception as e:  # noqa: BLE001
            print(f"(wandb disabled: {type(e).__name__}: {e})")
            run = None

    tasks = json.loads(Path(args.split).read_text())
    tasks = tasks[args.start : args.start + args.count]
    harness_code = Path(args.winner).read_text().strip()
    descs = _task_instructions()
    if args.playbook and not Path(args.playbook).exists():
        raise FileNotFoundError(
            f"--playbook {args.playbook} not found; pass --playbook '' for a no-playbook ablation."
        )
    playbook_ctx = (
        format_playbook_context(Playbook.load(args.playbook))
        if args.playbook
        else "(none)"
    )
    print(
        f"{len(tasks)} test tasks | mode={args.mode} | backend={args.advisor_backend} | "
        f"playbook={'yes' if args.playbook else 'NO'} | "
        f"{sum(1 for t in tasks if t in descs)}/{len(tasks)} have cached instructions",
        flush=True,
    )

    tmp_root = Path(tempfile.mkdtemp(prefix="tb2_patch_eval_"))
    prep = {}  # task -> (harness_dir, num_edits, status)

    if args.mode == "base":
        for t in tasks:
            prep[t] = (BASE_HARNESS_DIR, 0, "base")
    else:
        if args.advisor_backend == "endpoint":
            assert args.advisor_url, "--advisor-url required for endpoint backend"
        print(
            f"querying advisor ({args.advisor_backend}:{args.advisor_model}) "
            f"for {len(tasks)} tasks...",
            flush=True,
        )

        def q(t):
            desc = descs.get(
                t, "(no cached task description — infer the task type from the id)"
            )
            if args.advisor_backend == "vertex":
                raw = _advisor_vertex(
                    args.advisor_model, harness_code, t, desc, playbook=playbook_ctx
                )
            else:
                raw = _advisor_raw(
                    args.advisor_url,
                    args.advisor_model,
                    harness_code,
                    t,
                    desc,
                    playbook=playbook_ctx,
                )
            return t, raw

        raws = {}
        with ThreadPoolExecutor(max_workers=args.advisor_workers) as pool:
            for t, raw in pool.map(q, tasks):
                raws[t] = raw
        for i, t in enumerate(tasks):
            prep[t] = _prepare_harness(raws[t], tmp_root / f"task_{i}")
        n_patched = sum(1 for v in prep.values() if v[2] == "patched")
        print(
            f"  patches applied+validated: {n_patched}/{len(tasks)} (rest fall back to base)",
            flush=True,
        )

    def run_one(t):
        hdir, nedits, pstatus = prep[t]
        metrics = None
        try:
            reward, rstatus, metrics = TBX.eval_harness(
                hdir,
                t,
                student=args.student,
                max_turns=args.max_turns,
                return_metrics=True,
            )
        except Exception as e:  # noqa: BLE001
            reward, rstatus = 0.0, f"eval_error:{type(e).__name__}"
        row = {
            "task": t,
            "reward": float(reward),
            "num_edits": nedits,
            "patch_status": pstatus,
            "run_status": rstatus,
        }
        if metrics:  # step/token efficiency (from harbor trajectory.json)
            row["n_turns"] = metrics.get("n_turns")
            row["prompt_tokens"] = metrics.get("prompt_tokens")
            row["completion_tokens"] = metrics.get("completion_tokens")
            row["cached_tokens"] = metrics.get("cached_tokens")
            row["cost_usd"] = metrics.get("cost_usd")
        return row

    print("running (patched/base) harnesses via harbor + student...", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for n, row in enumerate(pool.map(run_one, tasks), 1):
            rows.append(row)
            m = sum(r["reward"] for r in rows) / len(rows)
            print(
                f"[{n}/{len(tasks)}] {row['task']:40s} r={row['reward']:.0f} "
                f"{row['patch_status']:20s} {row['run_status']:12s} running_rate={m * 100:.1f}%",
                flush=True,
            )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    shutil.rmtree(tmp_root, ignore_errors=True)

    solved = sum(1 for r in rows if r["reward"] >= 1.0)
    rate = 100.0 * solved / max(1, len(rows))
    n_patched = sum(1 for v in prep.values() if v[2] == "patched")
    print(
        f"\n=== TB2 {'TURBO HARNESS (patch advisor)' if args.mode == 'advisor' else 'BASE winner'} "
        f"TEST: {rate:.1f}% ({solved}/{len(rows)}) ==="
    )
    if args.mode == "advisor":
        # rate on the tasks the advisor actually patched vs the base-fallback tasks
        pat = [r for r in rows if r["patch_status"] == "patched"]
        fb = [r for r in rows if r["patch_status"] != "patched"]
        pr = 100.0 * sum(1 for r in pat if r["reward"] >= 1) / max(1, len(pat))
        fr = 100.0 * sum(1 for r in fb if r["reward"] >= 1) / max(1, len(fb))
        print(
            f"    patched tasks: {len(pat)} @ {pr:.1f}% | base-fallback tasks: {len(fb)} @ {fr:.1f}%"
        )
    print("    baseline: base winner harness 62.2% (train) ")

    if run is not None:
        run.log(
            {
                "eval/pass_rate": rate,
                "eval/n_solved": solved,
                "eval/n_tasks": len(rows),
                "eval/n_patched": n_patched,
                "eval/mode_advisor": int(args.mode == "advisor"),
            }
        )
        run.summary["pass_rate"] = rate
        run.finish()


if __name__ == "__main__":
    main()
