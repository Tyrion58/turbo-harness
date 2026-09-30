"""Meta-harness evolution loop for ALFWorld (Stage 1 of Turbo Meta-harness) — FULL SCAFFOLD.

Each candidate harness is a `harness.py` dir exposing run_episode(env, llm, max_steps) — the WHOLE
agent loop (context, memory, control flow, retries, stop logic). The proposer (Claude Code) evolves
the loop; we drop AgentBench entirely and run our own runner against the alfworld env + a frozen
Qwen3.5-9B.
  Phase 0: evaluate the default harness (faithful ReAct baseline) on the search subset
  Phase 1..N: propose (1 harness) -> validate (import) -> smoke -> eval -> update frontier
  Output: best harness -> artifacts/alfworld/<student>/<run_name>/general/

Metric = ALFWorld pass rate. Mirrors turbo_harness/featurebench_meta_harness.py (harness.py dirs),
with the executor swapped for the batched ALFWorld runner.

  python -m turbo_harness.alfworld_meta_harness \
      --iterations 8 --search-subset 0-99 --concurrency 32 --proposer-model claude-opus-4-6
"""

import argparse
import json
import os
import shutil
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

for _k in (
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL",
):
    if _k in os.environ:
        os.environ[_k] = os.environ[_k].replace("[1m]", "")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "meta-harness/reference_examples/terminal_bench_2"))
import claude_wrapper as cw  # noqa: E402

from turbo_harness.eval import alfworld_executor as ALFX  # noqa: E402

ARTIFACTS = REPO / "artifacts"
ALF_DATA = Path(os.environ.get("ALFWORLD_DATA", "data/alfworld_data"))
DEFAULT_HARNESS = REPO / "turbo_harness/alfworld/default_harness"
API_REF = REPO / "turbo_harness/alfworld/api.py"
ENV_REF = REPO / "turbo_harness/alfworld/env.py"
SKILL = REPO / "turbo_harness/ALFWORLD_PROPOSER_SKILL.md"

_USE_COLOR = sys.stdout.isatty()
_interrupted = False


def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _bold(t):
    return _c("1", t)


def _dim(t):
    return _c("2", t)


def _green(t):
    return _c("32", t)


def _red(t):
    return _c("31", t)


def _yellow(t):
    return _c("33", t)


def _cyan(t):
    return _c("36", t)


def _ts():
    return _dim(datetime.now().strftime("[%H:%M:%S]"))


def _elapsed(s):
    m, s = divmod(int(s), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def _rate_str(r):
    s = f"{r:.1%}"
    return _green(s) if r >= 0.5 else (_yellow(s) if r >= 0.25 else _red(s))


def _handle_signal(signum, frame):
    global _interrupted
    _interrupted = True
    print("\nInterrupted, finishing current step...", flush=True)


# ── Data ─────────────────────────────────────────────────────


def load_issues(split="harness_r1_train"):
    data = json.load(open(ALF_DATA / f"{split}.json"))
    games = []
    for v in data.values():
        games.extend(v)
    return games


def parse_subset(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


# ── Eval ─────────────────────────────────────────────────────


def eval_harness(
    harness,
    indices,
    split,
    student,
    trials,
    concurrency,
    max_steps,
    trajectory_dir=None,
    harness_name=None,
):
    return ALFX.eval_harness(
        harness,
        indices,
        split=split,
        student=student,
        trials=trials,
        concurrency=concurrency,
        max_steps=max_steps,
        trajectory_dir=trajectory_dir,
        harness_name=harness_name,
    )


def compute_pass_rates(task_results):
    per_issue, total, n = {}, 0.0, 0
    for idx, vals in task_results.items():
        per_issue[idx] = sum(vals) / len(vals) if vals else 0.0
        total += sum(vals)
        n += len(vals)
    return per_issue, (total / n if n else 0.0)


def smoke_test(harness_dir, indices, split, student, max_steps):
    idx0 = [sorted(indices)[0]]
    try:
        results, statuses, _ = ALFX.eval_harness(
            str(harness_dir),
            idx0,
            split=split,
            student=student,
            trials=1,
            concurrency=1,
            max_steps=max_steps,
        )
        st = statuses[0] if statuses else "?"
        if "MISSING" in st or "ERR" in st.upper():
            return False, f"bad status: {st}"
        return True, f"OK (status={st}, reward={results[idx0[0]][0]})"
    except Exception as e:  # noqa: BLE001
        return False, f"crash: {type(e).__name__}: {e}"


# ── Proposer ─────────────────────────────────────────────────

PROPOSER_TOOLS = ["Read", "Glob", "Grep", "Agent", "Write", "Edit", "Bash"]


def render_task_prompt(student, iteration, logs_dir, pending_eval_path, run_artifacts):
    return (
        f"Run iteration {iteration} of the harness evolution loop for **ALFWorld** "
        f"(text embodied household tasks: pick/place/clean/heat/cool/examine).\n\n"
        f"## Objective\n"
        f"Improve the **harness** = the full agent loop `run_episode(env, llm, max_steps)`, so the "
        f"frozen student solves MORE games (pass rate = fraction won). You write the WHOLE loop; you "
        f"do NOT change the model.\n\n"
        f"## The harness you are evolving (READ)\n"
        f"- Default/baseline loop: `{DEFAULT_HARNESS / 'harness.py'}` — a faithful zero-shot ReAct loop.\n"
        f"- Helpers you may use: `{API_REF}` (SYSTEM_PROMPT, TAKE_ACTION_TOOL, canonicalize, "
        f"extract_turn, LLMClient, available_actions_str, task_category, load_fewshot).\n"
        f"- Env interface: `{ENV_REF}` — `env.reset()->(obs,admissible)`; "
        f"`env.step(action)->(obs,admissible,done,won)`; `env.game_relpath`.\n"
        f"- `llm.complete(messages, tools=?, tool_choice=?, max_tokens=?)` -> raw OpenAI response.\n\n"
        f"## Context\n"
        f"- Student (frozen): **{student}** (served, tool-calling). One tool `take_action(action)`; "
        f"each turn pick one string from AVAILABLE ACTIONS. Actions are canonicalized to the nearest "
        f"admissible command.\n"
        f"- You may change ANYTHING in the loop: what context/history is kept or summarized, memory / "
        f"subgoal / stage state across steps, retries/repair on invalid actions, when to stop, "
        f"re-planning, injected reminders, few-shot, etc. Keep the signature "
        f"`run_episode(env, llm, max_steps) -> dict(won, messages, steps, status)`.\n\n"
        f"## Baseline weakness (test-set baseline)\n"
        f"- Multi-step tasks near the floor: clean 16%, heat 15%, **cool 7%**; simple pick&place 79%. "
        f"The agent fails to sequence subgoals (find -> clean/heat/cool at the right receptacle -> "
        f"place). Target these.\n\n"
        f"## Evolution state (READ)\n"
        f"- `{logs_dir / 'evolution_summary.jsonl'}` — past iterations + pass rates\n"
        f"- `{logs_dir / 'frontier.json'}` — best harness so far + best per game\n"
        f"- Previous candidates: `{run_artifacts}/iter*/` (read their harness.py + SUMMARY.md)\n\n"
        f"## Agent trajectories (CRITICAL — deep-read)\n"
        f"- `{logs_dir / 'trajectories'}/` — per-game JSONL ({{messages, resolved, status, steps}}): "
        f"`baseline/`, `iter*/`. Deep-read FAILED games (resolved=false) to find WHY the agent "
        f"loops/gives up/mis-sequences; read solved games to see what works.\n\n"
        f"## Output — propose exactly ONE candidate harness\n"
        f"- Start from the current frontier's harness.py (read `frontier.json` -> best_dir) and make a "
        f"targeted change; do not rewrite from scratch unless the frontier is the default.\n"
        f"- Write it to `{run_artifacts}/iter{iteration}_<name>/` containing: harness.py (required, "
        f"exposes run_episode), SUMMARY.md (hypothesis + what changed). Keep it GENERAL (no hard-coded "
        f"object/receptacle instance ids or per-game answers).\n"
        f"- Write `pending_eval.json` to `{pending_eval_path}` with a `candidates` list "
        f'(one entry: {{"name", "harness_dir", "hypothesis", "changes"}}).'
    )


def run_proposer(prompt, out_dir, logs_dir, iteration, proposer_model):
    out_dir.mkdir(parents=True, exist_ok=True)
    skill_text = SKILL.read_text() if SKILL.exists() else ""
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


# ── Frontier ─────────────────────────────────────────────────


def load_frontier(path):
    if path.exists():
        return json.loads(path.read_text())
    return {"best_avg": -1.0, "best_agent": None, "per_issue": {}}


def update_frontier(frontier_path, name, per_issue, avg, harness_dir):
    frontier = load_frontier(frontier_path)
    for idx, rate in per_issue.items():
        idx_str = str(idx)
        cur = frontier.get("per_issue", {}).get(idx_str, {}).get("pass_rate", -1)
        if rate > cur:
            frontier.setdefault("per_issue", {})[idx_str] = {
                "best_agent": name,
                "pass_rate": rate,
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
    if not summary_path.exists():
        return 0
    mx = 0
    for line in summary_path.read_text().strip().split("\n"):
        if line.strip():
            try:
                mx = max(mx, json.loads(line).get("iteration", 0))
            except json.JSONDecodeError:
                continue
    return mx


# ── Main ─────────────────────────────────────────────────────


def run_evolve(args):
    search_indices = parse_subset(args.search_subset)
    split = args.split
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")

    logs_dir = REPO / "experiments" / "logs" / "alfworld_meta_harness" / run_name
    logs_dir.mkdir(parents=True, exist_ok=True)
    frontier_path = logs_dir / "frontier.json"
    summary_path = logs_dir / "evolution_summary.jsonl"
    trajectory_dir = logs_dir / "trajectories"

    student_slug = args.student.split("/")[-1].lower().replace(".", "-")
    run_artifacts = ARTIFACTS / "alfworld" / student_slug / run_name
    run_artifacts.mkdir(parents=True, exist_ok=True)

    issues = load_issues(split)
    if max(search_indices) >= len(issues):
        print(_red(f"subset max {max(search_indices)} >= {len(issues)} games"))
        return

    if args.fresh:
        for f in [frontier_path, summary_path]:
            if f.exists():
                f.unlink()
        for d in run_artifacts.glob("iter*"):
            if d.is_dir():
                shutil.rmtree(d)
        print("  Fresh start: cleared artifacts and logs")

    start_iteration = count_iterations(summary_path)

    def _baseline_done():
        if not summary_path.exists():
            return False
        for ln in summary_path.read_text().splitlines():
            if ln.strip():
                try:
                    if json.loads(ln).get("iteration") == 0:
                        return True
                except json.JSONDecodeError:
                    pass
        return False

    resuming = start_iteration > 0 or _baseline_done()
    print(
        f"{_ts()} {_bold('ALFWorld Meta-Harness (full scaffold)')}  run={_cyan(run_name)}  "
        f"student={_cyan(args.student.split('/')[-1])}  iters={args.iterations}  "
        f"subset={len(search_indices)} games  split={split}"
        + (f"  (resuming from iter {start_iteration})" if resuming else "")
    )

    # Phase 0
    if not resuming:
        print(
            f"\n{_ts()} {_bold('Phase 0: Baseline (default harness)')}  subset={args.search_subset}"
        )
        t0 = time.time()
        base_results, _, base_resolved = eval_harness(
            "default",
            search_indices,
            split,
            args.student,
            args.trials,
            args.concurrency,
            args.max_steps,
            trajectory_dir=trajectory_dir / "baseline",
            harness_name="baseline",
        )
        base_per_issue, base_avg = compute_pass_rates(base_results)
        n_res = sum(1 for r in base_resolved if r)
        print(
            f"  {_ts()} baseline: pass_rate={_rate_str(base_avg)}  solved={n_res}/{len(base_resolved)}  "
            f"({_elapsed(time.time() - t0)})"
        )
        update_frontier(
            frontier_path, "baseline", base_per_issue, base_avg, str(DEFAULT_HARNESS)
        )
        append_evolution_summary(
            summary_path,
            {
                "iteration": 0,
                "agent": "baseline",
                "avg_rate": round(base_avg, 4),
                "resolved": n_res,
                "per_issue": {str(k): round(v, 3) for k, v in base_per_issue.items()},
            },
        )
    else:
        print(f"\n{_ts()} {_dim('Phase 0: skipped (resuming)')}")
        base_avg, base_per_issue = 0.0, {}
        for line in summary_path.read_text().strip().split("\n"):
            if line.strip():
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("iteration") == 0:
                    base_avg = d["avg_rate"]
                    base_per_issue = {int(k): v for k, v in d["per_issue"].items()}
                    break
        print(f"  baseline was: {_rate_str(base_avg)}")

    pending_eval_path = logs_dir / "pending_eval.json"
    history = []

    for i in range(args.iterations):
        if _interrupted:
            break
        iteration = start_iteration + i + 1
        iter_start = time.time()
        fr = load_frontier(frontier_path)
        best_avg = fr.get("best_avg", base_avg)
        print(
            f"\n{_ts()} {_bold(f'Iteration {iteration}')} ({i + 1}/{args.iterations})  "
            f"frontier={fr.get('best_agent', 'baseline')} @ {_rate_str(best_avg)}\n{'─' * 60}"
        )

        if pending_eval_path.exists():
            pending_eval_path.unlink()

        print(f"  {_ts()} {_cyan('proposing')} a harness...", flush=True)
        propose_start = time.time()
        prompt = render_task_prompt(
            args.student, iteration, logs_dir, pending_eval_path, run_artifacts
        )
        res = run_proposer(
            prompt, run_artifacts, logs_dir, iteration, args.proposer_model
        )
        propose_time = time.time() - propose_start
        print(
            f"  {_ts()} proposer: exit={res.exit_code} cost=${res.cost_usd:.3f} "
            f"{len(res.tool_calls)} tools ({_elapsed(propose_time)})"
        )
        res.show()

        if res.exit_code != 0:
            append_evolution_summary(
                summary_path,
                {
                    "iteration": iteration,
                    "agent": f"iter{iteration}",
                    "avg_rate": 0,
                    "valid": False,
                    "reason": "proposer failed",
                },
            )
            history.append({"iteration": iteration, "avg_rate": 0, "valid": False})
            continue

        if pending_eval_path.exists():
            candidates = json.loads(pending_eval_path.read_text()).get("candidates", [])
        else:
            fb = run_artifacts / f"iter{iteration}"
            candidates = (
                [
                    {
                        "name": f"iter{iteration}",
                        "harness_dir": str(fb),
                        "hypothesis": "",
                        "changes": "",
                    }
                ]
                if (fb / "harness.py").exists()
                else []
            )
        if not candidates:
            print(f"  {_red('no candidates')}")
            append_evolution_summary(
                summary_path,
                {
                    "iteration": iteration,
                    "agent": f"iter{iteration}",
                    "avg_rate": 0,
                    "valid": False,
                    "reason": "no candidates",
                },
            )
            history.append({"iteration": iteration, "avg_rate": 0, "valid": False})
            continue

        print(f"  {_ts()} proposed {len(candidates)} candidate(s)")
        valid = []
        for ci, c in enumerate(candidates):
            if _interrupted:
                break
            name, hd = c["name"], Path(c["harness_dir"])
            prefix = f"    [{ci + 1}/{len(candidates)}] {name}:"
            if not (hd / "harness.py").exists():
                print(f"{prefix} {_red('MISSING harness.py')}")
                continue
            ok, msg = ALFX.validate_harness(str(hd))
            if not ok:
                print(f"{prefix} {_red('import FAIL')}: {msg}")
                continue
            sok, smsg = smoke_test(
                hd, search_indices, split, args.student, args.max_steps
            )
            if not sok:
                print(f"{prefix} {_red('smoke FAIL')}: {smsg}")
                continue
            print(f"{prefix} {_green('valid + smoke OK')}: {smsg}")
            valid.append(c)

        if not valid:
            append_evolution_summary(
                summary_path,
                {
                    "iteration": iteration,
                    "agent": "none",
                    "avg_rate": 0,
                    "valid": False,
                    "reason": "all candidates failed validation",
                },
            )
            history.append({"iteration": iteration, "avg_rate": 0, "valid": False})
            continue

        for ci, c in enumerate(valid):
            if _interrupted:
                break
            name, hd = c["name"], str(Path(c["harness_dir"]))
            print(f"    [{ci + 1}/{len(valid)}] {_bold(name)}...", flush=True)
            eval_start = time.time()
            task_results, statuses, resolved_flags = eval_harness(
                hd,
                search_indices,
                split,
                args.student,
                args.trials,
                args.concurrency,
                args.max_steps,
                trajectory_dir=trajectory_dir / name,
                harness_name=name,
            )
            per_issue, avg = compute_pass_rates(task_results)
            n_res = sum(1 for r in resolved_flags if r)
            delta = avg - best_avg
            dc = (
                _green(f"{delta:+.1%}")
                if delta > 0
                else (_red(f"{delta:+.1%}") if delta < 0 else _dim(f"{delta:+.1%}"))
            )
            print(
                f"         pass_rate={_rate_str(avg)}  delta={dc}  solved={n_res}/{len(resolved_flags)}  "
                f"({_elapsed(time.time() - eval_start)})"
            )
            append_evolution_summary(
                summary_path,
                {
                    "iteration": iteration,
                    "agent": name,
                    "harness_dir": hd,
                    "avg_rate": round(avg, 4),
                    "resolved": n_res,
                    "per_issue": {str(k): round(v, 3) for k, v in per_issue.items()},
                    "delta_vs_frontier": round(delta, 4),
                    "hypothesis": c.get("hypothesis", ""),
                    "changes": c.get("changes", ""),
                    "propose_time_s": round(propose_time, 1),
                    "eval_time_s": round(time.time() - eval_start, 1),
                    "valid": True,
                    "smoke": True,
                },
            )
            history.append({"iteration": iteration, "agent": name, "avg_rate": avg})
            update_frontier(frontier_path, name, per_issue, avg, hd)
            print(
                _green("  ** NEW FRONTIER **")
                if avg > best_avg
                else _dim("  no improvement")
            )

        print(f"  {_dim(f'timing: total={_elapsed(time.time() - iter_start)}')}")

    # Summary
    print(f"\n{_ts()} {_bold('=== EVOLUTION SUMMARY ===')}")
    print(f"  baseline: {_rate_str(base_avg)}")
    for h in history:
        tag = "" if h.get("valid", True) else " (invalid)"
        print(
            f"  {h.get('agent', 'iter' + str(h['iteration']))}: {_rate_str(h['avg_rate'])}{tag}"
        )
    best = load_frontier(frontier_path)
    best_dir = best.get("best_dir")
    print(
        f"  BEST: {_rate_str(best.get('best_avg', 0))} @ {best.get('best_agent', 'none')}"
    )
    if best_dir and best.get("best_agent") != "baseline":
        gen = run_artifacts / "general"
        if gen.exists():
            shutil.rmtree(gen)
        shutil.copytree(best_dir, gen)
        print(f"  copied best → {gen}")
    elif best.get("best_agent") == "baseline":
        print(f"  {_yellow('baseline wins')} — no harness improvement found")
    print(
        f"\n  logs: {logs_dir}\n  frontier: {frontier_path}\n  summary: {summary_path}"
    )


def main():
    p = argparse.ArgumentParser(
        description="Meta-harness evolution for ALFWorld (full scaffold)"
    )
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument(
        "--search-subset", default="0-99", help="contiguous train indices, e.g. 0-99"
    )
    p.add_argument("--split", default="harness_r1_train")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--proposer-model", default="claude-opus-4-6")
    p.add_argument("--student", default="Qwen3.5-9B")
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--run-name", default=None)
    p.add_argument("--fresh", action="store_true")
    args = p.parse_args()
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    run_evolve(args)


if __name__ == "__main__":
    main()
