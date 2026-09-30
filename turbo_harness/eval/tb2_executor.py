"""TB2 executor for the RL patch-advisor: run ONE terminal-bench-2 task with a (policy-patched)
harness via harbor, student = Sonnet 5 on Vertex. Reward = harbor's verifier reward.txt (binary
0/1), read from the trusted verifier — `tests/test.sh` runs AFTER the agent phase, inside the task
container, and OVERWRITES reward.txt with the real check, so the policy-patched agent cannot fake its
own reward. Mirrors turbo_harness/terminal_bench/eval.py's harbor invocation, for a single task.

`make_harness_copy` builds a self-contained, harbor-runnable copy of the harness context (the agents
package + anthropic_caching.py + prompt-templates/ + tmux override); the RL env patches the winner
agent file inside that copy. `reward_safe` is defense-in-depth against a patch that tries to tamper
with the verifier path.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Reuse the pinned harbor config from the eval wrapper (single source of truth).
from turbo_harness.terminal_bench.eval import (  # noqa: E402
    HARBOR_BIN,
    DEFAULT_MODEL,
    DEFAULT_MODEL_INFO,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TURNS,
    DEFAULT_DATASET,
    DEFAULT_VERTEX_PROJECT,
    DEFAULT_VERTEX_LOCATION,
    REGISTRY_PATH,
)

BASE_HARNESS_DIR = REPO / "turbo_harness" / "terminal_bench"
AGENT_FILE = (
    "kira_auto_test.py"  # meta-harness WINNER (TB2.1 iter10, 48.9%); the policy patches THIS file
)
AGENT_IMPORT = "agents.kira_auto_test:AgentHarness"
# Minimal terminal_bench context harbor needs to import + run the (patched) agent.
_COPY = [
    "agents",
    "anthropic_caching.py",
    "prompt-templates",
    "docker-compose-tmux.yaml",
]

# A patched agent has no legitimate reason to reference the verifier path; harbor overwrites
# reward.txt post-agent so this is belt-and-suspenders, but reject the obvious tamper attempts.
_REWARD_HACK_TOKENS = (
    "reward.txt",
    "/logs/verifier",
    "tests/test.sh",
    "verifier/reward",
)


def make_harness_copy(out_dir) -> Path:
    """Copy the minimal terminal_bench harness context into out_dir (harbor-runnable)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in _COPY:
        src = BASE_HARNESS_DIR / item
        if not src.exists():
            continue
        dst = out_dir / item
        if src.is_dir():
            shutil.copytree(
                src,
                dst,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                dirs_exist_ok=True,
            )
        else:
            shutil.copy2(src, dst)
    return out_dir


def _harbor_python() -> Path:
    return Path(HARBOR_BIN).parent / "python"


def validate_harness(harness_dir) -> tuple[bool, str]:
    """Import the (patched) agent module in the harbor venv + check it exposes AgentHarness."""
    agent_path = Path(harness_dir) / "agents" / AGENT_FILE
    code = (
        "import importlib.util,sys\n"
        f"sys.path.insert(0,{str(harness_dir)!r})\n"
        f"spec=importlib.util.spec_from_file_location('rl_agent',{str(agent_path)!r})\n"
        "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "assert hasattr(m,'AgentHarness'),'no AgentHarness'\n"
        "print('OK')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(harness_dir), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    try:
        r = subprocess.run(
            [str(_harbor_python()), "-c", code],
            capture_output=True,
            text=True,
            timeout=180,
            env=env,
        )
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    out = (r.stdout + r.stderr).strip()
    if r.returncode == 0 and out.endswith("OK"):
        return True, "import OK"
    return False, (out.splitlines()[-1][:200] if out else "unknown import error")


def reward_safe(harness_dir) -> tuple[bool, str]:
    """Reject a patched agent that references the verifier/reward path (reward-hack defense)."""
    try:
        src = (Path(harness_dir) / "agents" / AGENT_FILE).read_text()
    except OSError as e:  # noqa: BLE001
        return False, f"read_error:{type(e).__name__}"
    hits = [t for t in _REWARD_HACK_TOKENS if t in src]
    if hits:
        return False, "reward_hack_tokens:" + ",".join(hits)
    return True, "reward_safe"


def eval_harness(
    harness_dir,
    task_id,
    student=None,
    max_turns=None,
    timeout=None,
    temperature=None,
    return_metrics=False,
):
    """Run ONE TB2 task with the (patched) harness via harbor + Sonnet.
    Returns (reward: float 0/1, status). Reward comes from the trusted verifier reward.txt.
    `temperature` (or env TB2_TEMPERATURE) overrides the default sampling temp — set 0 for a
    deterministic base-vs-patched comparison that isolates the patch's causal effect from run noise."""
    student = student or os.environ.get("TB2_REWARD_STUDENT", DEFAULT_MODEL)
    max_turns = int(max_turns or DEFAULT_MAX_TURNS)
    temperature = float(
        temperature
        if temperature is not None
        else os.environ.get("TB2_TEMPERATURE", DEFAULT_TEMPERATURE)
    )
    reap_orphan_containers()  # reap zombie harbor containers (>75min) so they don't choke the host
    uid = f"tb2rl_{task_id}_{uuid.uuid4().hex[:8]}"
    jobs_dir = Path(os.environ.get("TB2_RL_JOBS_DIR", "/tmp/tb2rl_jobs"))
    jobs_dir.mkdir(parents=True, exist_ok=True)
    tmux = Path(harness_dir) / "docker-compose-tmux.yaml"

    cmd = [
        str(HARBOR_BIN),
        "run",
        "--agent",
        AGENT_IMPORT,
        "-d",
        DEFAULT_DATASET,
        "--registry-path",
        str(REGISTRY_PATH),
        "-m",
        student,
        "-e",
        "docker",
        "-n",
        "1",
        "--n-attempts",
        "1",
        "--cpus",
        "ignore",
        "--memory",
        "ignore",
        "--extra-docker-compose",
        str(tmux),
        "--ak",
        f"temperature={temperature}",
        "--ak",
        f"model_info={json.dumps(DEFAULT_MODEL_INFO, separators=(',', ':'))}",
        "--ak",
        f"max_turns={max_turns}",
        "-i",
        task_id,
        "--jobs-dir",
        str(jobs_dir),
        "--job-name",
        uid,
    ]

    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(harness_dir), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    uv = shutil.which("uv")
    if uv:
        env["PATH"] = os.pathsep.join([str(Path(uv).parent), env.get("PATH", "")])
    env["VERTEXAI_PROJECT"] = os.environ.get("VERTEXAI_PROJECT", DEFAULT_VERTEX_PROJECT)
    env["VERTEXAI_LOCATION"] = os.environ.get(
        "VERTEXAI_LOCATION", DEFAULT_VERTEX_LOCATION
    )

    # Per-task cap = harbor's NATIVE per-task agent.timeout_sec (matches eval.py + meta_harness.py,
    # for train/eval consistency). Measured on the TB2.1 baseline matrix: agents finish/give-up in
    # <=35 min (2085s max), well under every task's native cap (900-12000s) — NONE ran to their
    # native timeout. A fixed sub-native cut (the old 1800s) therefore only ever ZERO-REWARDS legit
    # slow solves (e.g. train-fasttext @2085s, distribution-search @1392s) that eval scores natively,
    # creating a train↔eval mismatch that would mistrain the advisor. So the subprocess timeout below
    # is just a generous HANG backstop (default 7200s); harbor's native timeout is the effective cap.
    to = int(timeout or os.environ.get("TB2_RL_TIMEOUT", "7200"))
    _t0 = time.time()
    try:
        subprocess.run(
            cmd,
            cwd=str(harness_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=to,
        )
    except subprocess.TimeoutExpired:
        _cleanup(jobs_dir / uid)
        print(f"[tb2] {task_id}: harbor_timeout after {to}s (stuck task)", flush=True)
        return (0.0, "harbor_timeout", None) if return_metrics else (0.0, "harbor_timeout")
    except Exception as e:  # noqa: BLE001
        _cleanup(jobs_dir / uid)
        st = f"harbor_error:{type(e).__name__}"
        return (0.0, st, None) if return_metrics else (0.0, st)

    job = jobs_dir / uid
    reward, status = 0.0, "no_result"
    metrics = None
    for trial in sorted(job.glob("*/")) if job.exists() else []:
        rf = trial / "verifier" / "reward.txt"
        if rf.exists():
            try:
                v = float(rf.read_text().strip())
                reward = 1.0 if v >= 1.0 else 0.0
                status = "solved" if reward >= 1.0 else "failed"
            except ValueError:
                status = "reward_parse_error"
        elif (trial / "result.json").exists():
            status = "no_reward"  # finished but verifier wrote nothing (treat as fail)
        metrics = _parse_trajectory_metrics(trial)  # turns/tokens/cost, BEFORE cleanup
        break
    _cleanup(job)
    print(
        f"[tb2] {task_id}: {status} in {time.time() - _t0:.0f}s", flush=True
    )  # task-time distribution
    if return_metrics:
        return reward, status, metrics
    return reward, status


def _parse_trajectory_metrics(trial_dir):
    """Extract agent turns + token/cost from harbor's agent/trajectory.json (called BEFORE
    _cleanup deletes the trial). turns = number of agent-sourced steps; tokens/cost come from
    the trajectory's final_metrics block. Returns None if the file is missing/unparseable."""
    tj = Path(trial_dir) / "agent" / "trajectory.json"
    if not tj.exists():
        return None
    try:
        d = json.loads(tj.read_text())
    except Exception:  # noqa: BLE001
        return None
    steps = d.get("steps") or []
    n_turns = sum(
        1 for s in steps if isinstance(s, dict) and s.get("source") == "agent"
    )
    fm = d.get("final_metrics") or {}
    return {
        "n_turns": n_turns,
        "prompt_tokens": fm.get("total_prompt_tokens"),
        "completion_tokens": fm.get("total_completion_tokens"),
        "cached_tokens": fm.get("total_cached_tokens"),
        "cost_usd": fm.get("total_cost_usd"),
    }


def _cleanup(path):
    shutil.rmtree(path, ignore_errors=True)


_LAST_REAP = [0.0]


def reap_orphan_containers(max_age_s=7800, throttle_s=120):
    """Force-remove harbor task containers (image alexgshaw/*) older than max_age_s.

    Harbor leaves task containers RUNNING when a task times out or its process is killed; over a
    long RL run / repeated evals these pile up as zombies that choke the host and cause false
    `no_reward`/`harbor_timeout` failures in later runs. Under the NATIVE per-task cap a live rollout
    can legitimately run up to the 7200s subprocess backstop (harbor native caps reach 3600-12000s),
    so the reap threshold (default 7800s = backstop + buffer) must sit ABOVE it: a container older
    than that is DEFINITELY an orphan — never a live task — safe even while other rollouts (incl.
    concurrent same-task GRPO samples) run. Throttled to once per throttle_s so it's cheap per call."""
    now = time.time()
    if now - _LAST_REAP[0] < throttle_s:
        return
    _LAST_REAP[0] = now
    try:
        r = subprocess.run(
            ["docker", "ps", "--format", "{{.ID}}\t{{.Image}}\t{{.CreatedAt}}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:  # noqa: BLE001
        return
    rm = []
    for line in r.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 3 or "alexgshaw" not in parts[1]:
            continue
        try:  # CreatedAt e.g. "2026-09-15 18:00:00 +0000 UTC"
            dt = datetime.strptime(parts[2].rsplit(" ", 1)[0], "%Y-%m-%d %H:%M:%S %z")
            if now - dt.timestamp() > max_age_s:
                rm.append(parts[0])
        except (ValueError, IndexError):
            continue
    if rm:
        try:
            subprocess.run(
                ["docker", "rm", "-f", *rm], capture_output=True, timeout=120
            )
        except Exception:  # noqa: BLE001
            pass
