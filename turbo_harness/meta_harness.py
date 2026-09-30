"""Meta-harness evolution loop for SWE-smith.

Iteratively optimizes a GENERAL harness on a subset of training issues:
  Phase 0: evaluate baseline (default harness) on the search subset
  Phase 1..N: propose → validate → smoke test → eval → update frontier
  Output: best harness to artifacts/<repo>/general/

Follows the meta-harness pattern from terminal_bench_2/meta_harness.py,
adapted for SWE-smith (mini-swe-agent + compute_score via eval_server).

Run from repo root (requires eval_server running on port 5152):
  EVAL_SERVER_URL=http://localhost:5152 \
  VERTEXAI_PROJECT=your-gcp-project VERTEXAI_LOCATION=us-east5 \
  python -m turbo_harness.meta_harness \
      --repo dvc --iterations 5 --search-subset 0-29 --trials 2 --concurrency 5
"""

import argparse
import json
import os
import shutil
import signal
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

for _k in ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL",
           "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL",
           "ANTHROPIC_SMALL_FAST_MODEL"):
    if _k in os.environ:
        os.environ[_k] = os.environ[_k].replace("[1m]", "")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "turbo_harness/infra/mini-swe-agent/src"))
from turbo_harness import claude_wrapper as cw  # noqa: E402

from turbo_harness import executor as EX  # noqa: E402
from turbo_harness.proposer import (  # noqa: E402
    SKILL, DEFAULT_AGENT_SRC, DEFAULT_YAML, repo_file, validate_artifact,
)

ARTIFACTS = REPO / "artifacts"

_USE_COLOR = sys.stdout.isatty()
_interrupted = False


def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _bold(t): return _c("1", t)
def _dim(t): return _c("2", t)
def _green(t): return _c("32", t)
def _red(t): return _c("31", t)
def _yellow(t): return _c("33", t)
def _cyan(t): return _c("36", t)


def _ts():
    return _dim(datetime.now().strftime("[%H:%M:%S]"))


def _elapsed(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def _rate_str(rate):
    s = f"{rate:.1%}"
    if rate >= 0.5:
        return _green(s)
    elif rate >= 0.25:
        return _yellow(s)
    return _red(s)


def _handle_signal(signum, frame):
    global _interrupted
    _interrupted = True
    print("\nInterrupted, finishing current step...", flush=True)


# ── Data loading ─────────────────────────────────────────────

def load_issues(repo, split="train"):
    data = REPO / f"data/swe_smith/{split}_{repo_file(repo)}.json"
    rows = [json.loads(line) for line in open(data) if line.strip()]
    return [json.loads(r["reward_spec"]["ground_truth_json"]) for r in rows]


def parse_subset(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


# ── Evaluation ───────────────────────────────────────────────

def eval_harness(harness, indices, issues, student, trials, concurrency,
                 repo, step_limit, cost_limit, trajectory_dir=None):
    """Evaluate a harness on a set of issues × trials.

    Returns: {idx: [resolved_bool_per_trial, ...]}
    """
    jobs = [(idx, rep) for idx in indices for rep in range(trials)]
    results = defaultdict(list)
    statuses = []

    def run_one(idx, rep):
        gt = issues[idx]
        r = EX.run_instance(gt, student, harness=harness, repo=repo,
                            step_limit=step_limit, cost_limit=cost_limit,
                            trajectory_dir=trajectory_dir)
        return idx, r["resolved"], r["status"], r.get("agent_cost", 0)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {pool.submit(run_one, idx, rep): (idx, rep) for idx, rep in jobs}
        for f in as_completed(futs):
            idx, rep = futs[f]
            try:
                idx_r, resolved, status, cost = f.result()
                results[idx_r].append(resolved)
                statuses.append(status)
            except Exception as e:
                results[idx].append(False)
                statuses.append(f"HARNESS_ERR:{type(e).__name__}")

    return dict(results), statuses


def compute_pass_rates(task_results):
    """Compute per-issue pass rates and flat average.

    Returns: (per_issue_dict, flat_avg)
    """
    per_issue = {}
    total_passes = 0
    total_trials = 0
    for idx, trials in task_results.items():
        per_issue[idx] = sum(trials) / len(trials) if trials else 0.0
        total_passes += sum(trials)
        total_trials += len(trials)
    avg = total_passes / total_trials if total_trials else 0.0
    return per_issue, avg


# ── Validation & smoke test ──────────────────────────────────

def validate_harness(harness_dir):
    return validate_artifact(harness_dir)


def _find_smoke_test_idx(issues):
    """Pick the shortest issue for smoke testing (fast, cheap)."""
    best_idx, best_len = 0, float("inf")
    for i, gt in enumerate(issues):
        plen = len(gt.get("problem_statement", ""))
        if 0 < plen < best_len:
            best_idx, best_len = i, plen
    return best_idx


def smoke_test(harness_dir, issues, student, repo, step_limit, cost_limit):
    """Run 1 issue, 1 trial to verify harness doesn't crash."""
    idx = _find_smoke_test_idx(issues)
    gt = issues[idx]
    try:
        r = EX.run_instance(gt, student, harness=str(harness_dir), repo=repo,
                            step_limit=step_limit, cost_limit=cost_limit)
        if "ERROR" in r.get("status", ""):
            return False, f"agent error: {r['status']}"
        return True, f"OK (resolved={r['resolved']}, steps={r['steps']}, cost=${r.get('agent_cost',0):.3f})"
    except Exception as e:
        return False, f"crash: {type(e).__name__}: {e}"


# ── Proposer ─────────────────────────────────────────────────

PROPOSER_TOOLS = ["Read", "Glob", "Grep", "Agent", "Write", "Edit", "Bash"]


def render_task_prompt(repo, student, iteration, logs_dir, pending_eval_path, run_artifacts):
    """Build the prompt for the proposer. Like TB2, the proposer reads
    evolution_summary.jsonl, frontier.json, and agent trajectories."""
    return (
        f"Run iteration {iteration} of the harness evolution loop for SWE-smith.\n\n"
        f"## Context\n"
        f"- Repository: {repo}\n"
        f"- Student model (frozen, runs the harness): **{student}**\n"
        f"- Baseline scaffold: DefaultAgent from mini-swe-agent\n"
        f"  - Source: {DEFAULT_AGENT_SRC}\n"
        f"  - Config (templates + limits): {DEFAULT_YAML}\n\n"
        f"## Evolution state (READ THESE FILES)\n"
        f"- `{logs_dir / 'evolution_summary.jsonl'}` — past iterations and results\n"
        f"- `{logs_dir / 'frontier.json'}` — current best harness per issue + overall best\n"
        f"- Previous harness artifacts: `{run_artifacts}/iter*/` — read harness.py from prior iterations\n\n"
        f"## Agent trajectories (CRITICAL — deep-read these)\n"
        f"- `{logs_dir / 'trajectories'}/` — step-by-step agent execution logs (JSONL files)\n"
        f"  - Subdirs: `baseline/`, `iter1/`, `iter2/`, etc.\n"
        f"  - Each file = one issue's full trajectory: agent messages (thoughts + commands + outputs)\n"
        f"  - **Deep-read FAILED trajectories** to understand WHY the agent fails.\n"
        f"  - **Also read SUCCESSFUL trajectories** to understand what works well.\n"
        f"  - This is the MOST IMPORTANT step — design harness changes that target specific failure modes.\n\n"
        f"## Output\n"
        f"- Write candidate harness artifacts into: `{run_artifacts}/`\n"
        f"  - Each candidate in its own subdirectory (e.g. `iter{iteration}_<name>/`)\n"
        f"  - Each dir must contain: harness.py, RULES_MEMORY.md, SUMMARY.md\n"
        f"  - harness.py MUST expose `build_agent(model, env, step_limit, cost_limit)`\n"
        f"- Write `pending_eval.json` to: `{pending_eval_path}`"
    )


def run_proposer(prompt, out_dir, logs_dir, iteration, proposer_model):
    """Run Claude Code proposer with the SKILL prior as system prompt.

    Like TB2: strips ANTHROPIC_API_KEY so claude uses subscription auth,
    sets cwd to repo root so proposer can navigate the full codebase."""
    out_dir.mkdir(parents=True, exist_ok=True)
    skill_text = SKILL.read_text()
    os.environ.pop("CLAUDECODE", None)
    saved_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        res = cw.run(
            prompt,
            model=proposer_model,
            allowed_tools=PROPOSER_TOOLS,
            system_prompt=skill_text,
            cwd=str(REPO),
            log_dir=str(logs_dir / "sessions"),
            name=f"iter{iteration}",
            progress=False,
            timeout_seconds=2400,
            effort="max",
        )
    finally:
        if saved_key:
            os.environ["ANTHROPIC_API_KEY"] = saved_key
    return res


# ── Frontier tracking ────────────────────────────────────────

def load_frontier(path):
    if path.exists():
        return json.loads(path.read_text())
    return {"best_avg": -1.0, "best_agent": None, "per_issue": {}}


def update_frontier(frontier_path, name, per_issue, avg, harness_dir):
    frontier = load_frontier(frontier_path)

    for idx, rate in per_issue.items():
        idx_str = str(idx)
        current = frontier.get("per_issue", {}).get(idx_str, {}).get("pass_rate", -1)
        if rate > current:
            frontier.setdefault("per_issue", {})[idx_str] = {
                "best_agent": name, "pass_rate": rate
            }

    if avg > frontier.get("best_avg", -1):
        frontier["best_avg"] = avg
        frontier["best_agent"] = name
        frontier["best_dir"] = str(harness_dir)

    frontier_path.write_text(json.dumps(frontier, indent=2))
    return frontier


def append_evolution_summary(summary_path, entry):
    with open(summary_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def count_iterations(summary_path):
    """Highest iteration in evolution_summary.jsonl (for resume)."""
    if not summary_path.exists():
        return 0
    max_iter = 0
    for line in summary_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            max_iter = max(max_iter, json.loads(line).get("iteration", 0))
        except json.JSONDecodeError:
            continue
    return max_iter


# ── Main loop ────────────────────────────────────────────────

def run_evolve(args):
    # Register local vLLM models with litellm (avoids cost calculation crash)
    if "openai/" in args.student or "hosted_vllm/" in args.student:
        import litellm
        litellm.drop_params = True
        litellm.register_model({
            args.student: {
                "max_tokens": 40960, "max_input_tokens": 40960,
                "max_output_tokens": 40960,
                "input_cost_per_token": 0, "output_cost_per_token": 0,
            }
        })

    repo = args.repo
    search_indices = parse_subset(args.search_subset)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")

    logs_dir = REPO / "experiments" / "logs" / "meta_harness" / run_name
    logs_dir.mkdir(parents=True, exist_ok=True)
    frontier_path = logs_dir / "frontier.json"
    summary_path = logs_dir / "evolution_summary.jsonl"

    # Isolate artifacts by student model + run-name
    # e.g., artifacts/dvc/haiku/dvc-opus-50-v3/
    _STUDENT_SLUGS = {
        "claude-haiku-4-5": "haiku",
        "claude-sonnet-4-6": "sonnet",
        "claude-opus-4-6": "opus",
    }
    raw_name = args.student.split("/")[-1]
    student_slug = _STUDENT_SLUGS.get(raw_name, raw_name.lower().replace(".", "-"))
    run_artifacts = ARTIFACTS / repo / student_slug / run_name
    run_artifacts.mkdir(parents=True, exist_ok=True)

    issues = load_issues(repo, "train")

    if args.fresh:
        for f in [frontier_path, summary_path]:
            if f.exists():
                f.unlink()
        for d in run_artifacts.glob("iter*"):
            if d.is_dir():
                shutil.rmtree(d)
        print(f"  Fresh start: cleared artifacts and logs for {run_name}")

    trajectory_dir = logs_dir / "trajectories"

    # Check for resume
    start_iteration = count_iterations(summary_path)
    resuming = start_iteration > 0

    print(f"{_ts()} {_bold('Meta-Harness Evolution')}  run={_cyan(run_name)}  "
          f"repo={_cyan(repo)}  iters={args.iterations}  "
          f"subset={len(search_indices)} issues  trials={args.trials}"
          + (f"  (resuming from iter {start_iteration})" if resuming else ""))

    # ── Phase 0: Baseline ─────────────────────────────────────
    if not resuming:
        print(f"\n{_ts()} {_bold('Phase 0: Baseline')}  "
              f"subset={search_indices[:5]}{'...' if len(search_indices) > 5 else ''}  "
              f"trials={args.trials}")
        t0 = time.time()
        base_results, base_statuses = eval_harness(
            "default", search_indices, issues, args.student, args.trials,
            args.concurrency, repo, args.max_steps, args.cost_limit,
            trajectory_dir=trajectory_dir / "baseline")
        base_per_issue, base_avg = compute_pass_rates(base_results)
        print(f"  {_ts()} baseline: {_rate_str(base_avg)} ({_elapsed(time.time() - t0)})")

        update_frontier(frontier_path, "baseline", base_per_issue, base_avg, "default")
        append_evolution_summary(summary_path, {
            "iteration": 0, "agent": "baseline", "avg_rate": round(base_avg, 4),
            "per_issue": {str(k): round(v, 3) for k, v in base_per_issue.items()},
        })

        for idx in sorted(search_indices):
            rate = base_per_issue.get(idx, 0)
            print(f"    idx {idx:>3d}: {_rate_str(rate)}")
    else:
        print(f"\n{_ts()} {_dim('Phase 0: skipped (resuming)')}")
        base_avg = 0.0
        base_per_issue = {}
        with open(summary_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("iteration") == 0:
                    base_avg = d["avg_rate"]
                    base_per_issue = {int(k): v for k, v in d["per_issue"].items()}
                    break
        if not base_per_issue:
            print(f"  {_yellow('warning')}: no baseline entry in summary, using defaults")
        print(f"  baseline was: {_rate_str(base_avg)}")

    # ── Phase 1..N: Evolution ─────────────────────────────────
    pending_eval_path = logs_dir / "pending_eval.json"
    history = []
    if resuming and summary_path.exists():
        for line in summary_path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("iteration", 0) > 0:
                history.append({
                    "iteration": d["iteration"],
                    "agent": d.get("agent", f"iter{d['iteration']}"),
                    "avg_rate": d.get("avg_rate", 0),
                    "valid": d.get("valid", True),
                })

    for i in range(args.iterations):
        if _interrupted:
            print("Interrupted.")
            break

        iteration = start_iteration + i + 1
        iter_start = time.time()
        fr = load_frontier(frontier_path)
        best_avg = fr.get("best_avg", base_avg)
        best_agent = fr.get("best_agent", "baseline")
        print(f"\n{_ts()} {_bold(f'Iteration {iteration}')} ({i+1}/{args.iterations})  "
              f"frontier={best_agent} @ {_rate_str(best_avg)}")
        print(f"{'─' * 60}")

        # Clear pending_eval from previous iteration
        if pending_eval_path.exists():
            pending_eval_path.unlink()

        # Propose
        print(f"  {_ts()} {_cyan('proposing')} new harness candidates...", flush=True)
        propose_start = time.time()
        prompt = render_task_prompt(repo, args.student, iteration, logs_dir, pending_eval_path, run_artifacts)
        res = run_proposer(prompt, run_artifacts, logs_dir, iteration, args.proposer_model)
        propose_time = time.time() - propose_start
        print(f"  {_ts()} proposer: exit={res.exit_code} cost=${res.cost_usd:.3f} "
              f"{len(res.tool_calls)} tools ({_elapsed(propose_time)})")
        res.show()

        if res.exit_code != 0:
            print(f"  {_red('proposer failed')}")
            append_evolution_summary(summary_path, {
                "iteration": iteration, "agent": f"iter{iteration}",
                "avg_rate": 0, "valid": False, "reason": "proposer failed",
                "propose_time_s": round(propose_time, 1),
            })
            history.append({"iteration": iteration, "avg_rate": 0, "valid": False})
            continue

        # Read candidates from pending_eval.json (like TB2)
        if pending_eval_path.exists():
            pending = json.loads(pending_eval_path.read_text())
            candidates = pending.get("candidates", [])
        else:
            # Fallback: if proposer didn't write pending_eval.json, check for default dir
            fallback_dir = run_artifacts / f"iter{iteration}"
            if fallback_dir.exists() and (fallback_dir / "harness.py").exists():
                candidates = [{"name": f"iter{iteration}", "harness_dir": str(fallback_dir),
                               "hypothesis": "", "changes": ""}]
            else:
                print(f"  {_red('no candidates')} — pending_eval.json missing and no fallback")
                append_evolution_summary(summary_path, {
                    "iteration": iteration, "agent": f"iter{iteration}",
                    "avg_rate": 0, "valid": False, "reason": "no candidates produced",
                    "propose_time_s": round(propose_time, 1),
                })
                history.append({"iteration": iteration, "avg_rate": 0, "valid": False})
                continue

        print(f"  {_ts()} proposed {len(candidates)} candidate(s)")
        for ci, c in enumerate(candidates):
            print(f"    {ci+1}. {_bold(c['name'])}: {c.get('hypothesis', '')[:80]}")

        # Validate + smoke test + evaluate each candidate
        valid_candidates = []
        print(f"  {_ts()} {_cyan('validating')} {len(candidates)} candidate(s)...")
        for ci, c in enumerate(candidates):
            if _interrupted:
                break
            name = c["name"]
            harness_dir = Path(c["harness_dir"])
            prefix = f"    [{ci+1}/{len(candidates)}] {name}:"

            # Check artifacts
            for fname in ("harness.py", "RULES_MEMORY.md", "SUMMARY.md"):
                p = harness_dir / fname
                s = _green("OK") if p.exists() else _red("MISSING")
                print(f"      {s} {fname}" + (f" ({len(p.read_text())} chars)" if p.exists() else ""))

            # Validate import
            ok, msg = validate_harness(harness_dir)
            if not ok:
                print(f"{prefix} {_red('import FAIL')}: {msg}")
                continue
            print(f"{prefix} {_green('import OK')}")

            # Smoke test
            smoke_ok, smoke_msg = smoke_test(harness_dir, issues, args.student, repo,
                                              args.max_steps, args.cost_limit)
            if not smoke_ok:
                print(f"{prefix} {_red('smoke FAIL')}: {smoke_msg}")
                continue
            print(f"{prefix} {_green('smoke OK')}: {smoke_msg}")

            valid_candidates.append(c)

        if not valid_candidates:
            print(f"  {_red('0 valid')} out of {len(candidates)} candidates")
            append_evolution_summary(summary_path, {
                "iteration": iteration, "agent": "none",
                "avg_rate": 0, "valid": False, "reason": "all candidates failed validation",
                "propose_time_s": round(propose_time, 1),
            })
            history.append({"iteration": iteration, "avg_rate": 0, "valid": False})
            continue
        print(f"  {_green(f'{len(valid_candidates)} valid')} out of {len(candidates)} candidates")

        # Benchmark each valid candidate
        bench_start = time.time()
        n_evals = len(valid_candidates) * len(search_indices) * args.trials
        print(f"  {_ts()} {_cyan('benchmarking')} {len(valid_candidates)} candidate(s) × "
              f"{len(search_indices)} issues × {args.trials} trials = {n_evals} evals")

        for ci, c in enumerate(valid_candidates):
            if _interrupted:
                break
            name = c["name"]
            harness_dir = str(Path(c["harness_dir"]))

            print(f"    [{ci+1}/{len(valid_candidates)}] {_bold(name)}...", flush=True)
            eval_start = time.time()
            task_results, statuses = eval_harness(
                harness_dir, search_indices, issues, args.student, args.trials,
                args.concurrency, repo, args.max_steps, args.cost_limit,
                trajectory_dir=trajectory_dir / name)
            per_issue, avg = compute_pass_rates(task_results)
            eval_time = time.time() - eval_start

            delta = avg - best_avg
            delta_str = f"{delta:+.1%}"
            delta_colored = (_green(delta_str) if delta > 0
                             else (_red(delta_str) if delta < 0 else _dim(delta_str)))
            print(f"         avg={_rate_str(avg)}  delta={delta_colored}  ({_elapsed(eval_time)})")

            # Per-issue results
            for idx in sorted(search_indices):
                rate = per_issue.get(idx, 0)
                base_rate_i = base_per_issue.get(idx, 0)
                d = rate - base_rate_i
                d_s = f"{d:+.0%}" if d != 0 else "  ="
                d_c = _green(d_s) if d > 0 else (_red(d_s) if d < 0 else _dim(d_s))
                print(f"         idx {idx:>3d}: {_rate_str(rate)}  {d_c}")

            # Update tracking
            entry = {
                "iteration": iteration, "agent": name,
                "harness_dir": harness_dir,
                "avg_rate": round(avg, 4),
                "per_issue": {str(k): round(v, 3) for k, v in per_issue.items()},
                "delta_vs_frontier": round(delta, 4),
                "hypothesis": c.get("hypothesis", ""),
                "changes": c.get("changes", ""),
                "propose_time_s": round(propose_time, 1),
                "eval_time_s": round(eval_time, 1),
                "valid": True, "smoke": True,
            }
            append_evolution_summary(summary_path, entry)
            history.append({"iteration": iteration, "agent": name, "avg_rate": avg})

            update_frontier(frontier_path, name, per_issue, avg, harness_dir)
            if avg > best_avg:
                print(f"  {_green('** NEW FRONTIER **')}: {name} @ {_rate_str(avg)}")
            else:
                print(f"  {_dim('no improvement')}")

        bench_time = time.time() - bench_start

        wall_time = time.time() - iter_start
        print(f"  {_dim(f'timing: propose={_elapsed(propose_time)} bench={_elapsed(bench_time)} total={_elapsed(wall_time)}')}")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{_ts()} {_bold('=== EVOLUTION SUMMARY ===')}")
    print(f"  baseline: {_rate_str(base_avg)}")
    for h in history:
        valid = h.get("valid", True)
        smoke = h.get("smoke", True)
        name = h.get("agent", f"iter{h['iteration']}")
        tag = "" if valid and smoke else " (invalid)" if not valid else " (smoke fail)"
        print(f"  {name}: {_rate_str(h['avg_rate'])}{tag}")

    best = load_frontier(frontier_path)
    best_dir = best.get("best_dir")
    best_avg = best.get("best_avg", 0)
    print(f"  BEST: {_rate_str(best_avg)} @ {best.get('best_agent', 'none')}")

    if best_dir and best_dir != "default":
        gen = run_artifacts / "general"
        if gen.exists():
            shutil.rmtree(gen)
        shutil.copytree(best_dir, gen)
        print(f"  copied best → {gen}")
    elif best_dir == "default":
        print(f"  {_yellow('baseline wins')} — no harness improvement found")

    print(f"\n  logs: {logs_dir}")
    print(f"  frontier: {frontier_path}")
    print(f"  summary: {summary_path}")

    # ── Generate CONCEPT.md from evolution data ──────────────
    if best_dir and best_dir != "default":
        print(f"\n{_ts()} Generating CONCEPT.md from evolution data...")
        try:
            from turbo_harness.generate_concept import generate_concept
            concept_path = generate_concept(
                run_name=run_name,
                repo=args.repo,
                student_label=args.student,
            )
            if concept_path and concept_path.exists():
                gen = run_artifacts / "general"
                gen_concept = gen / "CONCEPT.md"
                if gen.exists() and not gen_concept.exists():
                    shutil.copy2(concept_path, gen_concept)
                print(f"  CONCEPT.md → {concept_path}")
        except Exception as e:
            print(f"  WARNING: CONCEPT.md generation failed: {e}")

    # ── Generate default_config.json for the advisor ──────────
    if best_dir and best_dir != "default":
        gen = run_artifacts / "general"
        if gen.exists() and (gen / "harness.py").exists():
            print(f"\n{_ts()} Generating default_config.json...")
            try:
                from turbo_harness.harness_config import generate_default_config
                generate_default_config(gen)
                print(f"  default_config.json → {gen / 'default_config.json'}")
            except Exception as e:
                print(f"  WARNING: default_config.json generation failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Meta-harness evolution for SWE-smith")
    parser.add_argument("--repo", default="dvc")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--search-subset", default="0-29",
                        help="Training indices to optimize on (e.g. 0-29 or 0,5,10)")
    parser.add_argument("--trials", type=int, default=2,
                        help="Trials per issue during evolution")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--proposer-model", default="sonnet",
                        help="Claude model for proposer")
    parser.add_argument("--student", default="vertex_ai/claude-haiku-4-5")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--cost-limit", type=float, default=3.0)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--fresh", action="store_true",
                        help="Clear previous run artifacts and logs")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    run_evolve(args)


if __name__ == "__main__":
    main()
