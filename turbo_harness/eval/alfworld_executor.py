"""ALFWorld executor (full-scaffold): score ONE harness (a `harness.py` dir exposing
run_episode, or 'default') with the frozen student over a set of ALFWorld games.

Runs our own runner (`turbo_harness.alfworld.runner`) in the alfworld-worker venv — a process
pool over games, no AgentBench. Metric = ALFWorld pass rate (won ∈ {0,1}). The runner also
emits playbook-pipeline trajectory JSONL, so `turbo_harness/playbook/*` (Stage 2) consumes it unchanged.

The student is served separately (frozen Qwen3.5-9B on an OpenAI-compatible vLLM endpoint).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import os
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

WORKER_PY = Path(os.environ.get("ALFWORLD_WORKER_PY", sys.executable))
TARGET_URL = "http://127.0.0.1:8110/v1"
TARGET_MODEL = "Qwen3.5-9B"
DEFAULT_SPLIT = "harness_r1_train"
_OUT_ROOT = Path(os.environ.get("ALFWORLD_OUT_ROOT", "logs/alfworld_runs"))


def validate_harness(harness_dir) -> tuple[bool, str]:
    """Import <harness_dir>/harness.py in the worker venv and check run_episode(env, llm, max_steps)."""
    if harness_dir == "default":
        return True, "default"
    code = (
        "import importlib.util, inspect, sys\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        f"spec=importlib.util.spec_from_file_location('h', {str(Path(harness_dir) / 'harness.py')!r})\n"
        "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "assert hasattr(m,'run_episode'), 'no run_episode'\n"
        "ps=list(inspect.signature(m.run_episode).parameters)\n"
        "assert len(ps)>=3, f'run_episode needs (env, llm, max_steps); got {ps}'\n"
        "print('OK')\n"
    )
    try:
        r = subprocess.run(
            [str(WORKER_PY), "-c", code], capture_output=True, text=True, timeout=120
        )
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    out = (r.stdout + r.stderr).strip()
    if r.returncode == 0 and out.endswith("OK"):
        return True, "import OK"
    return False, out.splitlines()[-1][:200] if out else "unknown import error"


# Tokens a POLICY-PATCHED harness has no legitimate reason to contain, but which could reach the
# trusted outcome object (via reflection over the scoring facade's method closures) to inflate reward.
# Trusted base harnesses (general/default) contain NONE of these, so reward_safe never rejects them.
_REWARD_HACK_TOKENS = (
    "_last_won", "final_won", "__closure__", "cell_contents",
    "__globals__", "__subclasses__", "get_referents", "gc.get_objects",
)


def reward_safe(harness_dir) -> tuple[bool, str]:
    """Defense-in-depth for reward integrity on the RL/eval PATCH path: reject a patched harness that
    references internals it could use to reach and inflate the trusted outcome (the scoring facade
    already blocks the simple hacks; this catches deeper reflection). Not used for trusted flows."""
    if harness_dir == "default":
        return True, "default"
    try:
        src = (Path(harness_dir) / "harness.py").read_text()
    except OSError as e:  # noqa: BLE001
        return False, f"read_error:{type(e).__name__}"
    hits = [t for t in _REWARD_HACK_TOKENS if t in src]
    if hits:
        return False, "reward_hack_tokens:" + ",".join(hits)
    return True, "reward_safe"


def _run_batch(
    harness,
    start,
    count,
    split,
    student,
    max_steps,
    concurrency,
    out_jsonl,
    traj_dir,
    harness_name,
    trusted_score=False,
) -> int:
    cmd = [
        str(WORKER_PY),
        "-m",
        "turbo_harness.alfworld.runner",
        "--harness-dir",
        str(harness),
        "--harness-name",
        harness_name,
        "--split",
        split,
        "--start",
        str(start),
        "--count",
        str(count),
        "--concurrency",
        str(concurrency),
        "--max-steps",
        str(max_steps),
        "--base-url",
        TARGET_URL,
        "--model",
        student,
        "--out",
        str(out_jsonl),
    ]
    if traj_dir:
        cmd += ["--traj-dir", str(traj_dir)]
    if trusted_score:
        cmd += ["--trusted-score"]
    log_path = Path(out_jsonl).with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as lf:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO),
            stdout=lf,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=max(1800, count * 120),
        )
    return proc.returncode


_LAST_STEPS: dict = {}  # {index: steps} side-channel so callers can record an efficiency metric


def eval_harness(
    harness,
    indices,
    split=DEFAULT_SPLIT,
    student=TARGET_MODEL,
    trials=1,
    concurrency=24,
    max_steps=50,
    trajectory_dir=None,
    harness_name=None,
    trusted_score=False,
    cleanup=False,
):
    """Evaluate a harness ('default' or a dir) over a CONTIGUOUS set of game indices.

    Returns ({idx: [reward_per_trial]}, statuses, resolved_flags) — reward = 1.0 if won else 0.0.
    trusted_score=True reads the env outcome (not the harness dict) — use for policy-patched harnesses.
    cleanup=True removes the per-run output dir after reading — use for RL/eval rollouts (thousands
    of them) so `mh_runs/` does not leak dirs/inodes.
    """
    indices = sorted(indices)
    start, count = indices[0], len(indices)
    assert indices == list(range(start, start + count)), (
        "ALFWorld eval needs a contiguous subset"
    )
    harness_name = harness_name or (
        "default" if harness == "default" else Path(harness).name
    )

    results = {i: [] for i in indices}
    statuses, resolved_flags = [], []
    for rep in range(trials):
        run_id = f"{harness_name}_{split}_{start}-{start + count}_{rep}_{uuid.uuid4().hex[:6]}"
        out_jsonl = _OUT_ROOT / run_id / "results.jsonl"
        rc = _run_batch(
            harness,
            start,
            count,
            split,
            student,
            max_steps,
            concurrency,
            out_jsonl,
            trajectory_dir,
            harness_name,
            trusted_score,
        )
        got = {}
        if out_jsonl.exists():
            for line in open(out_jsonl):
                if line.strip():
                    r = json.loads(line)
                    got[r["index"]] = (
                        1.0 if r.get("won") else 0.0,
                        r.get("status", "?"),
                    )
                    _LAST_STEPS[r["index"]] = int(r.get("steps", 0) or 0)
        for i in indices:
            reward, status = got.get(i, (0.0, f"MISSING(rc={rc})"))
            results[i].append(reward)
            statuses.append(status)
            resolved_flags.append(bool(reward >= 1.0))
        if cleanup:
            shutil.rmtree((_OUT_ROOT / run_id), ignore_errors=True)
    return results, statuses, resolved_flags


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="ALFWorld full-scaffold executor")
    ap.add_argument("--harness", default="default")
    ap.add_argument("--split", default=DEFAULT_SPLIT)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--traj-dir", default=None)
    args = ap.parse_args()
    if args.harness != "default":
        print("validate:", validate_harness(args.harness))
    res, st, rf = eval_harness(
        args.harness,
        list(range(args.start, args.start + args.count)),
        split=args.split,
        concurrency=args.concurrency,
        trajectory_dir=args.traj_dir,
    )
    n = sum(rf)
    print(f"pass {n}/{len(rf)} = {100 * n / max(1, len(rf)):.1f}%")
