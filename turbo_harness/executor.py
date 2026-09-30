"""Executor: run a harness (default or a proposer-produced artifact) with the student model
on one SWE-smith instance, in Docker, and score resolution.

A harness artifact is a dir with `harness.py` exposing `build_agent(model, env, step_limit, cost_limit)`.
Passing `--harness default` runs the vanilla mini-swe-agent (baseline).

Run from repo root:
  VERTEXAI_PROJECT=your-gcp-project VERTEXAI_LOCATION=us-east5 \
  python -m turbo_harness.executor --idx 0 --harness general --rep 0
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
MSA_SRC = REPO / "turbo_harness/infra/mini-swe-agent/src"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(MSA_SRC))

from minisweagent.agents.default import AgentConfig, DefaultAgent  # noqa: E402
from minisweagent.environments.docker import DockerEnvironment  # noqa: E402
from minisweagent.models.litellm_model import LitellmModel  # noqa: E402

from turbo_harness.infra.scoring import compute_score  # noqa: E402

DEFAULT_YAML = MSA_SRC / "minisweagent/config/default.yaml"
ARTIFACTS = REPO / "artifacts"


def _load_build_agent(harness_dir):
    hp = Path(harness_dir) / "harness.py"
    spec = importlib.util.spec_from_file_location(f"harness_{Path(harness_dir).name}", hp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_agent


def _default_agent(model, env, step_limit, cost_limit):
    cd = yaml.safe_load(open(DEFAULT_YAML))["agent"]
    cfg = AgentConfig(
        system_template=cd["system_template"], instance_template=cd["instance_template"],
        action_observation_template=cd["action_observation_template"],
        format_error_template=cd["format_error_template"],
        step_limit=step_limit, cost_limit=cost_limit,
    )
    return DefaultAgent(model, env, config_class=lambda **k: cfg)


def _truncate_message(msg, max_len=2000):
    """Truncate a message's content for trajectory logging."""
    content = msg.get("content", "")
    if len(content) > max_len:
        content = content[:max_len] + f"\n... ({len(content) - max_len} chars truncated)"
    return {"role": msg["role"], "content": content}


def _save_trajectory(agent, result, trajectory_dir):
    """Save agent trajectory (step-by-step messages) to a JSONL file."""
    if trajectory_dir is None:
        return
    trajectory_dir = Path(trajectory_dir)
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    instance_id = result["instance_id"]
    safe_id = instance_id.replace("/", "_")
    harness_name = result.get("harness", "default")
    if "/" in str(harness_name):
        harness_name = Path(harness_name).name
    path = trajectory_dir / f"{safe_id}.jsonl"

    messages = [_truncate_message(m) for m in getattr(agent, "messages", [])]
    entry = {
        "instance_id": instance_id,
        "harness": result.get("harness", "default"),
        "resolved": result["resolved"],
        "status": result["status"],
        "steps": result["steps"],
        "agent_cost": result.get("agent_cost", 0),
        "messages": messages,
    }
    with open(path, "w") as f:
        f.write(json.dumps(entry) + "\n")


def run_instance(gt, student_model, harness="default", repo="dvc",
                 step_limit=40, cost_limit=3.0, trajectory_dir=None, score=True):
    """Run one harness on one instance -> resolution result dict.

    When score=False, skip patch scoring (compute_score) and return the raw patch so a
    TRUSTED parent process can score it. This isolates policy-executed harness code from
    the reward computation (it cannot monkeypatch compute_score in a process it never runs in).
    """
    instance_id, image, problem = gt["instance_id"], gt["image_name"], gt["problem_statement"]
    if gt.get("benchmark") == "verified":
        # Verified images are ~3GB and Docker-Hub rate-limited; load from the local tar cache
        # (or pull-once-then-cache) so `docker run` never has to auto-pull and hit the limit.
        # A wedged rootless daemon can make this fail — error THIS instance, don't crash the run.
        from turbo_harness.infra.image_cache import ensure_image
        try:
            ensure_image(image)
        except Exception as e:
            return {"instance_id": instance_id, "harness": harness, "resolved": False,
                    "status": "ERROR:image_unavailable", "steps": 0, "agent_cost": 0,
                    "patch": "", "patch_len": 0, "solve_s": 0,
                    "info": f"ensure_image failed: {str(e)[:200]}"}
    env = DockerEnvironment(image=image, cwd="/testbed", timeout=60,
                            run_args=["--rm", "--network", "host"])
    if gt.get("benchmark") == "verified":
        # SWE-bench Verified images are already checked out at base_commit and have no per-instance
        # branch to fetch — reset to base_commit to clear any residual state, then proceed identically.
        bc = gt.get("base_commit", "")
        if bc:
            # git checkout rewrites the whole working tree; on rootless fuse-overlayfs under heavy
            # concurrent load this can be slow, so allow well beyond the default 60s exec timeout.
            env.execute(f"git checkout -f {bc} 2>/dev/null || git reset --hard {bc} 2>/dev/null || true",
                        cwd="/testbed", timeout=int(os.environ.get("VERIFIED_CHECKOUT_TIMEOUT", "300")))
        env.execute("git rev-parse HEAD", cwd="/testbed")
    else:
        fetch_result = env.execute("git fetch --all", cwd="/testbed", timeout=120)
        if fetch_result["returncode"] != 0:
            try:
                env.cleanup()
            except Exception:
                pass
            return {"instance_id": instance_id, "harness": harness, "resolved": False,
                    "status": "ERROR:git_fetch_failed", "steps": 0, "agent_cost": 0,
                    "patch_len": 0, "solve_s": 0, "info": f"git fetch failed: {fetch_result['output'][:200]}"}
        checkout_result = env.execute(f"git checkout {instance_id}", cwd="/testbed")
        if checkout_result["returncode"] != 0:
            try:
                env.cleanup()
            except Exception:
                pass
            return {"instance_id": instance_id, "harness": harness, "resolved": False,
                    "status": "ERROR:git_checkout_failed", "steps": 0, "agent_cost": 0,
                    "patch_len": 0, "solve_s": 0, "info": f"git checkout failed: {checkout_result['output'][:200]}"}
        env.execute("git rev-parse HEAD", cwd="/testbed")

    model = LitellmModel(model_name=student_model)
    try:
        if harness == "default":
            agent = _default_agent(model, env, step_limit, cost_limit)
        else:
            harness_dir = harness if Path(harness).exists() else (ARTIFACTS / repo / harness)
            agent = _load_build_agent(harness_dir)(model, env, step_limit, cost_limit)

        uname = env.execute("uname -srvm")["output"].strip().split()
        sysv = {"system": uname[0], "release": uname[1] if len(uname) > 1 else "",
                "version": uname[2] if len(uname) > 2 else "",
                "machine": uname[3] if len(uname) > 3 else "x86_64"}
        t0 = time.time()
        try:
            status, _ = agent.run(problem, **sysv)
        except Exception as e:
            status = f"ERROR:{type(e).__name__}"
        dt = time.time() - t0
        if status == "Submitted":
            try:
                env.execute("git add -A", cwd="/testbed")
                patch = env.execute("git diff --cached", cwd="/testbed")["output"]
            except Exception:
                patch = ""
        else:
            patch = ""
    finally:
        try:
            env.cleanup()
        except Exception:
            pass

    if score:
        reward, run_id, info = compute_score(patch, gt)
        if run_id in ("ERROR", "TIMEOUT"):
            status = f"SCORE_ERROR:{info}"
        resolved = reward >= 1.0
    else:
        resolved, info = None, ""
    result = {"instance_id": instance_id, "harness": harness, "resolved": resolved,
              "status": status, "steps": model.n_calls, "agent_cost": model.cost,
              "patch": patch, "patch_len": len(patch), "solve_s": dt, "info": info}

    _save_trajectory(agent, result, trajectory_dir)
    return result


def _load_gt(repo, idx):
    data = REPO / f"data/swe_smith/train_{ {'tenacity':'jd__tenacity','dvc':'iterative__dvc'}.get(repo, repo) }.json"
    rows = [json.loads(line) for line in open(data) if line.strip()]
    return json.loads(rows[idx]["reward_spec"]["ground_truth_json"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--harness", default="default", help="'default' or an artifact name/dir (e.g. 'general')")
    ap.add_argument("--repo", default="dvc")
    ap.add_argument("--rep", type=int, default=0, help="trial index (for multi-trial)")
    ap.add_argument("--student", default="vertex_ai/claude-haiku-4-5")
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--cost-limit", type=float, default=3.0)
    ap.add_argument("--out-dir", default=str(REPO / "experiments/logs/harness_eval"))
    args = ap.parse_args()

    gt = _load_gt(args.repo, args.idx)
    r = run_instance(gt, args.student, harness=args.harness, repo=args.repo,
                     step_limit=args.max_steps, cost_limit=args.cost_limit)
    r["idx"] = args.idx
    r["rep"] = args.rep
    print(f"[{args.harness}] idx={args.idx} rep={args.rep} RESOLVED={r['resolved']} "
          f"steps={r['steps']} cost=${r['agent_cost']:.4f} status={r['status']}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.harness}_idx{args.idx}_r{args.rep}.json").write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
