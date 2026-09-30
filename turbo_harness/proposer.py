"""Phase A proposer: a frontier model (Claude Code) proposes a GENERAL harness for a repo.

Modeled on the meta-harness proposer: prime Claude Code with a SKILL prior + the baseline scaffold +
a sample of training issues (with baseline outcomes), and have it WRITE a real harness artifact
(scaffold code), not reword prompts. Output artifact:
  artifacts/<repo>/general/{harness.py, RULES_MEMORY.md, SUMMARY.md}

Run from repo root:
  VERTEXAI_PROJECT=your-gcp-project VERTEXAI_LOCATION=us-east5 \
  python -m turbo_harness.proposer --repo tenacity --n-samples 8
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# strip [1m] so the spawned `claude` materializer/proposer works on Vertex
for _k in ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
           "ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_SMALL_FAST_MODEL"):
    if _k in os.environ:
        os.environ[_k] = os.environ[_k].replace("[1m]", "")

REPO = Path(__file__).resolve().parents[1]
MSA_SRC = REPO / "turbo_harness/infra/mini-swe-agent/src"
from turbo_harness import claude_wrapper as cw  # noqa: E402

HERE = Path(__file__).resolve().parent
SKILL = HERE / "PROPOSER_SKILL.md"
ARTIFACTS = REPO / "artifacts"
DEFAULT_AGENT_SRC = MSA_SRC / "minisweagent/agents/default.py"
DEFAULT_YAML = MSA_SRC / "minisweagent/config/default.yaml"


def load_baseline_outcomes(repo):
    """idx -> resolved(bool), parsed from baseline run logs (if present)."""
    out = {}
    # Try repo-specific baseline logs first, then fallback to generic
    candidates = [
        REPO / f"experiments/logs/baseline_{repo_file(repo).replace('__', '_')}",
        REPO / "experiments/logs/baseline_passrate_gemini",
        REPO / "experiments/logs/baseline_dvc_haiku",
    ]
    for d in candidates:
        if not d.exists():
            continue
        for f in d.glob("run_*.log"):
            m = re.search(r"run_(\d+)", f.name)
            r = re.search(r"RESOLVED=(\w+)", f.read_text())
            if m and r:
                out[int(m.group(1))] = (r.group(1) == "True")
        # Also try idx*.json files (from baseline_passrate.py)
        for f in d.glob("idx*.json"):
            import json as _json
            try:
                data = _json.loads(f.read_text())
                out[data["idx"]] = data["resolved"]
            except Exception:
                pass
        if out:
            break
    return out


def sample_training_issues(repo, n, seed_idxs=None):
    data = REPO / f"data/swe_smith/train_{repo_file(repo)}.json"
    rows = [json.loads(line) for line in open(data) if line.strip()]
    outcomes = load_baseline_outcomes(repo)
    # prefer issues with known baseline outcomes; take a spread
    idxs = seed_idxs or list(range(0, len(rows), max(1, len(rows) // n)))[:n]
    out = []
    for i in idxs:
        gt = json.loads(rows[i]["reward_spec"]["ground_truth_json"])
        out.append({"idx": i, "problem": gt["problem_statement"],
                    "resolved": outcomes.get(i)})
    return out


def repo_file(repo):
    return {"tenacity": "jd__tenacity", "dvc": "iterative__dvc"}.get(repo, repo)


def build_prompt(repo, sampled, student):
    skill = SKILL.read_text()
    blocks = []
    for s in sampled:
        tag = {True: "baseline PASS", False: "baseline FAIL", None: "baseline unknown"}[s["resolved"]]
        blocks.append(f"### training issue idx {s['idx']}  ({tag})\n{s['problem'][:1500]}")
    issues = "\n\n".join(blocks)
    return (
        f"{skill}\n\n"
        f"====================\n## TASK\n"
        f"Repository: {repo}. Propose the GENERAL harness for solving bug-fix issues in this repo.\n\n"
        f"### Base model that will RUN this harness\n"
        f"The harness will be executed by the STUDENT model **{student}** — a comparatively small / "
        f"low-capability model on a tight step budget. Design the harness for ITS capability: keep it "
        f"lean and efficient (get to a correct source edit in few steps); do NOT impose heavy, "
        f"multi-stage process (e.g. mandatory reproduction scripts + full test-suite runs) that a weak "
        f"model executes poorly and that burns its step budget before it can fix the bug.\n\n"
        f"### Baseline scaffold to study (READ THESE FIRST)\n"
        f"- DefaultAgent source (overridable methods): {DEFAULT_AGENT_SRC}\n"
        f"- Baseline config (templates + limits): {DEFAULT_YAML}\n\n"
        f"### Sample of training issues (with baseline outcome)\n{issues}\n\n"
        f"### Deliverable\n"
        f"Write harness.py, RULES_MEMORY.md, SUMMARY.md into the current directory, per the skill's "
        f"output contract. harness.py MUST expose build_agent(model, env, step_limit, cost_limit)."
    )


def validate_artifact(out_dir):
    """Validate harness.py: import check + template rendering + lint."""
    import subprocess
    hp = Path(out_dir) / "harness.py"
    if not hp.exists():
        return False, "harness.py missing"

    check_script = f"""
import sys, os
sys.path.insert(0, {str(MSA_SRC)!r})
os.environ.setdefault('VERTEXAI_PROJECT', 'test')
os.environ.setdefault('VERTEXAI_LOCATION', 'us-east5')
import importlib.util, inspect

# Step 1: Import and check build_agent exists
spec = importlib.util.spec_from_file_location('h', {str(hp)!r})
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
assert hasattr(mod, 'build_agent') and callable(mod.build_agent), 'build_agent missing'
params = list(inspect.signature(mod.build_agent).parameters)

# Step 2: Try to build agent with a dummy model/env to catch template errors
from unittest.mock import MagicMock
mock_model = MagicMock()
mock_model.n_calls = 0
mock_model.cost = 0
mock_model.get_template_vars = lambda: {{}}
mock_env = MagicMock()
mock_env.get_template_vars = lambda: {{}}
try:
    agent = mod.build_agent(mock_model, mock_env, 40, 3.0)
    # Step 3: Try rendering templates (catches Jinja UndefinedError/SyntaxError)
    agent.extra_template_vars = {{'task': 'test problem', 'system': 'Linux', 'release': '5.0', 'version': '1', 'machine': 'x86_64', 'step_limit': 40, 'cost_limit': 3.0}}
    if hasattr(agent, 'render_template') and hasattr(agent, 'config'):
        agent.render_template(agent.config.system_template)
        agent.render_template(agent.config.instance_template)
    print('OK:' + ','.join(params))
except Exception as e:
    print(f'TEMPLATE_ERROR:{{type(e).__name__}}: {{e}}')
    sys.exit(1)
"""
    result = subprocess.run(
        [sys.executable, "-c", check_script],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode == 0 and "OK:" in result.stdout:
        params = result.stdout.strip().split("OK:")[1]
        return True, f"OK (build_agent params: [{params}])"
    err = (result.stderr or result.stdout).strip()[-500:]
    return False, f"validation failed: {err}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="dvc")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--model", default="sonnet", help="proposer (frontier) model")
    ap.add_argument("--student", default="vertex_ai/claude-haiku-4-5",
                    help="the base/student model that will RUN the harness")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    out_dir = ARTIFACTS / args.repo / "general"
    if args.validate_only:
        ok, msg = validate_artifact(out_dir)
        print(("VALID: " if ok else "INVALID: ") + msg)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    sampled = sample_training_issues(args.repo, args.n_samples)
    print(f"Proposing general harness for {args.repo} from {len(sampled)} training issues "
          f"(idxs {[s['idx'] for s in sampled]}) -> {out_dir}")

    res = cw.run(
        build_prompt(args.repo, sampled, args.student),
        model=args.model,
        allowed_tools=["Read", "Glob", "Grep", "Write", "Edit"],
        cwd=str(out_dir),
        log_dir=str(HERE / "logs" / "proposer"),
        name=f"propose-{args.repo}",
        progress=False,
        timeout_seconds=1800,
        effort="high",
    )
    print(f"proposer session: exit={res.exit_code} cost=${res.cost_usd:.4f} "
          f"{len(res.tool_calls)} tool-calls {res.duration_seconds:.0f}s")
    for f in ("harness.py", "RULES_MEMORY.md", "SUMMARY.md"):
        p = out_dir / f
        print(f"  {'OK ' if p.exists() else 'MISSING'} {f}"
              + (f" ({len(p.read_text())} chars)" if p.exists() else ""))
    ok, msg = validate_artifact(out_dir)
    print(("VALID: " if ok else "INVALID: ") + msg)


if __name__ == "__main__":
    main()
