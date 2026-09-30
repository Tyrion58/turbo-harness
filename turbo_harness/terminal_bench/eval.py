"""Terminal-Bench-2 (TB2) evaluation wrapper — OUR code.

Pins the standard GLM-on-TB2 evaluation config and a train/test split, then drives the
upstream harbor harness (``meta-harness/reference_examples/terminal_bench_2``) as the engine.
This replaces ad-hoc ``run_eval.sh`` flag invocations so evals are reproducible and the
evolution/RL pipeline can call the SAME config programmatically (see ``run_split``).

Defaults reproduce the recommended/faithful baseline:
  stock KIRA agent, native per-task timeouts (no multiplier), NO step cap, temperature 0.7.
Both bounding levers are exposed as flags (``--max-turns``, ``--timeout-multiplier``) for
RL/dev, but default OFF so the wrapper matches the reported baseline out of the box.

Also injects our fixed tmux override (``docker-compose-tmux.yaml``) which prefers each
container's own tmux — fixing the host-tmux glibc mismatch that broke old-base (bullseye)
task images. See [[tb2-setup-and-status]].

Usage:
  python -m turbo_harness.terminal_bench.eval --split train
  python -m turbo_harness.terminal_bench.eval --split test --concurrency 32
  python -m turbo_harness.terminal_bench.eval --split train -i qemu-startup -i qemu-alpine-ssh
  python -m turbo_harness.terminal_bench.eval --split train --max-turns 40 --timeout-multiplier 0.5
  python -m turbo_harness.terminal_bench.eval --split train --dry-run   # print the harbor cmd only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent  # our TB2 module (agents/, prompt-templates/ live here)
TB2 = Path(os.environ.get("TB2_ENGINE_DIR", str(REPO / "meta-harness" / "reference_examples" / "terminal_bench_2")))
HARBOR_BIN = Path(os.environ.get("HARBOR_BIN", str(TB2 / ".venv" / "bin" / "harbor")))  # or set HARBOR_BIN to the harbor on your PATH (tb2 extra)
SPLIT_DIR = HERE / "data"  # our stratified split (see make_split.py)
# TB2.1 = the fixed task set (26 tasks patched: procps/deps/timeouts/reward-hacks). It is NOT in
# the harbor registry, so we pin it via a local registry that points each task at the 2-1 repo
# (harbor-framework/terminal-bench-2-1) at a fixed commit; harbor reads it via --registry-path.
REGISTRY_PATH = SPLIT_DIR / "registry_tb21.json"
TMUX_OVERRIDE = HERE / "docker-compose-tmux.yaml"
RESULTS_DIR = REPO / "experiments" / "results" / "tb2"

# ── Pinned standard config (override via CLI/env) ───────────────────────────────
# Executor/student = Sonnet 4.6 on Vertex — a native fit for the Claude-shaped KIRA/Terminus-2
# harnesses, and deliberately WEAKER than Sonnet 5. Sonnet 5 ~saturated TB2: its residual was
# capability-bound (hard ML/systems tasks), leaving only ~2-4 harness-fixable tasks -> no Turbo
# headroom. A weaker student fails more tasks in ways a better scaffold CAN fix. History:
# Opus-4.6 (too $) -> local GLM-5.3-Flash (latency/weak/thrashy) -> Sonnet 5 (too strong,
# saturated) -> Sonnet 4.6 (+ TB2.1 to remove the broken-task confound). The GLM-only hacks
# (reasoning_effort via chat_template_kwargs, api_base) are gated to openai/ models; Vertex
# Anthropic routes via VERTEXAI_PROJECT/LOCATION + ADC.
DEFAULT_MODEL = os.environ.get("TB2_MODEL", "vertex_ai/claude-sonnet-4-5")
DEFAULT_VERTEX_PROJECT = os.environ.get("VERTEXAI_PROJECT", "your-gcp-project")
DEFAULT_VERTEX_LOCATION = os.environ.get("VERTEXAI_LOCATION", "global")  # us-east5 is quota-starved
DEFAULT_API_BASE = os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")  # only for openai/ (local GLM)
DEFAULT_AGENT = "agents.baseline_kira:AgentHarness"
DEFAULT_DATASET = "terminal-bench@2.1"  # fixed task set; resolved via REGISTRY_PATH (see above)
DEFAULT_CONCURRENCY = 32  # probe: Vertex/your-gcp-project-global handles >=48 concurrent Sonnet
# reqs with 0 rate-limits, so the real limit is host docker resources (~32 validated). Push to
# ~45 (all train tasks at once) only if host CPU/IO on command-heavy tasks stays healthy.
DEFAULT_TEMPERATURE = 1.0
# reasoning_effort ONLY applies to the local GLM (openai/) path (delivered via chat_template_kwargs);
# native Anthropic/Vertex models (Sonnet) ignore it and use extended thinking instead.
DEFAULT_REASONING_EFFORT = os.environ.get("TB2_REASONING_EFFORT", "low")
# Per-run turn cap: bounds runaway loops on hard tasks (300 > the highest observed solve, 241).
DEFAULT_MAX_TURNS = 300
# Sonnet 5 context; max_output also bounds KIRA's tool-call path (baseline_kira). Kept in-window.
DEFAULT_MODEL_INFO = {
    "max_input_tokens": 180000,
    "max_output_tokens": 16384,
    "input_cost_per_token": 0,
    "output_cost_per_token": 0,
}


def _uv() -> str:
    cand = shutil.which("uv")
    if cand and Path(cand).exists():
        return cand
    sys.exit("error: `uv` not found on PATH (install uv; see docs/DEPENDENCIES.md)")


def load_tasks(split: str) -> list[str]:
    if split == "all":  # union of train+test = the full pool
        return sorted(set(load_tasks("train") + load_tasks("test")))
    f = SPLIT_DIR / f"{split}_tasks.json"
    if not f.exists():
        sys.exit(f"error: split file not found: {f}")
    return json.loads(f.read_text())


def build_cmd(args, tasks: list[str]) -> list[str]:
    is_openai = args.model.startswith("openai/")  # local vLLM (GLM) needs the extra hacks below
    cmd = [
        str(HARBOR_BIN),
        "run",
        "--agent",
        args.agent,
        "-d",
        args.dataset,
        "--registry-path",
        str(REGISTRY_PATH),
        "-m",
        args.model,
        "-e",
        "docker",
        "-n",
        str(args.concurrency),
        "--n-attempts",
        str(args.n_attempts),
        "--cpus",
        "ignore",
        "--memory",
        "ignore",
        "--extra-docker-compose",
        str(TMUX_OVERRIDE),  # our fixed tmux override
        "--ak",
        f"temperature={args.temperature}",
        "--ak",
        f"model_info={json.dumps(DEFAULT_MODEL_INFO, separators=(',', ':'))}",
    ]
    if is_openai:
        # local GLM (openai/ vLLM): reasoning effort must go via chat_template_kwargs (litellm
        # drops the top-level reasoning_effort under drop_params), plus an explicit api_base.
        # Native Anthropic on Vertex (Sonnet) uses extended thinking + ADC, so skip all of this.
        cmd += [
            "--ak",
            f"reasoning_effort={args.reasoning_effort}",
            "--ak",
            "llm_call_kwargs="
            + json.dumps(
                {"extra_body": {"chat_template_kwargs": {"reasoning_effort": args.reasoning_effort}}},
                separators=(",", ":"),
            ),
            "--ak",
            f"api_base={args.api_base}",
        ]
    if args.max_turns:  # 0 (or None) = off / native
        cmd += ["--ak", f"max_turns={args.max_turns}"]
    if args.timeout_multiplier is not None:  # time cap (default: native timeouts)
        cmd += ["--agent-timeout-multiplier", str(args.timeout_multiplier)]
    for t in tasks:
        cmd += ["-i", t]
    return cmd


def _subprocess_env(args) -> dict:
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)  # let `uv run` use the TB2 project venv cleanly
    # cwd=HERE (our tree) makes `agents.baseline_kira` resolve to OUR copy; harbor comes
    # from the reference project's venv (HARBOR_BIN executable), not from cwd.
    env["PYTHONPATH"] = os.pathsep.join([str(HERE), env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
    env["PATH"] = os.pathsep.join([str(Path(_uv()).parent), env.get("PATH", "")])
    if args.model.startswith("openai/"):  # local GLM vLLM endpoint
        env["OPENAI_API_BASE"] = args.api_base
        env.setdefault("OPENAI_API_KEY", "dummy")
    else:  # Vertex Anthropic (Sonnet 5) — litellm reads these + ADC
        env["VERTEXAI_PROJECT"] = DEFAULT_VERTEX_PROJECT
        env["VERTEXAI_LOCATION"] = DEFAULT_VERTEX_LOCATION
    return env


def summarize(job_dir: Path) -> dict:
    """Bucket trials into solved / model-failure / infra(tmux)-failure and compute pass rates."""
    solved, model_fail, infra_fail = [], [], []
    for trial in sorted(job_dir.glob("*/")):
        res = trial / "result.json"
        if not res.exists():
            continue
        d = json.loads(res.read_text())
        exc = d.get("exception_info") or {}
        exc_type = exc.get("exception_type", "") if isinstance(exc, dict) else ""
        exc_msg = exc.get("exception_message", "") if isinstance(exc, dict) else ""
        rf = trial / "verifier" / "reward.txt"
        reward = None
        if rf.exists():
            try:
                reward = float(rf.read_text().strip())
            except ValueError:
                reward = None
        name = trial.name
        if exc_type == "RuntimeError" and "Failed to start tmux session" in exc_msg:
            infra_fail.append(
                name
            )  # our-bug tmux/glibc failure (should be 0 with the fix)
        elif reward is not None and reward >= 1.0:
            solved.append(name)
        else:
            model_fail.append(name)  # reward 0 / timeout / other non-infra failure
    n = len(solved) + len(model_fail) + len(infra_fail)
    runnable = len(solved) + len(model_fail)
    return {
        "job_dir": str(job_dir),
        "n_tasks": n,
        "solved": len(solved),
        "model_failure": len(model_fail),
        "infra_tmux_failure": len(infra_fail),
        "raw_pass_rate": round(len(solved) / n, 4) if n else None,
        "fair_pass_rate": round(len(solved) / runnable, 4) if runnable else None,
        "infra_failed_tasks": infra_fail,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="TB2 evaluation wrapper (Sonnet-5 on Vertex by default; pinned config)."
    )
    ap.add_argument(
        "--split",
        choices=["train", "test", "all"],
        help="which split to run (reads data/{split}_tasks.json; 'all' = full 89-task pool)",
    )
    ap.add_argument(
        "-i",
        "--task",
        action="append",
        default=None,
        help="run specific task id(s) instead of a full split (repeatable)",
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--api-base", default=DEFAULT_API_BASE)
    ap.add_argument("--agent", default=DEFAULT_AGENT)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("-n", "--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--n-attempts", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument(
        "--reasoning-effort",
        default=DEFAULT_REASONING_EFFORT,
        help="GLM reasoning depth: low|high|max (default low = light/fast turns)",
    )
    ap.add_argument(
        "--max-turns",
        type=int,
        default=DEFAULT_MAX_TURNS,
        help=f"per-run turn cap (default {DEFAULT_MAX_TURNS}; bounds low-reasoning thrashing). 0=off",
    )
    ap.add_argument(
        "--timeout-multiplier",
        type=float,
        default=None,
        help="scale per-task timeouts (default: native = recommended)",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="print the harbor command and exit"
    )
    args = ap.parse_args()

    if not args.task and not args.split:
        ap.error("provide --split {train,test} or one or more -i <task>")
    tasks = args.task or load_tasks(args.split)
    cmd = build_cmd(args, tasks)
    jobs_dir = HERE / "jobs"
    # include PID so concurrent eval runs (e.g. a model x harness matrix) never collide on a
    # same-second timestamp -> harbor FileExistsError on the shared jobs dir.
    job_name = f"tb2_{args.split or 'subset'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    cmd += ["--jobs-dir", str(jobs_dir), "--job-name", job_name]

    print(
        f"[tb2.eval] model={args.model} agent={args.agent} split={args.split or '(subset)'} "
        f"tasks={len(tasks)} concurrency={args.concurrency} n_attempts={args.n_attempts}"
    )
    print(
        f"[tb2.eval] step_cap={args.max_turns or 'off'} timeout_multiplier={args.timeout_multiplier or 'native'} "
        f"tmux_override={TMUX_OVERRIDE.name}"
    )
    if args.dry_run:
        print("[tb2.eval] DRY RUN — harbor command:")
        print("  (cd %s && %s)" % (HERE, " ".join(cmd)))
        return

    proc = subprocess.run(cmd, cwd=HERE, env=_subprocess_env(args))

    job = jobs_dir / job_name
    if not job.exists():
        sys.exit(f"error: expected job dir not found: {job}")
    summary = summarize(job)
    print("\n[tb2.eval] ===== RESULT =====")
    print(json.dumps(summary, indent=2))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.split or "subset"
    out = RESULTS_DIR / f"{tag}_{args.model.split('/')[-1]}_{stamp}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[tb2.eval] summary saved -> {out}")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
