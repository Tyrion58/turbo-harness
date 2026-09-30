"""Trajectory-grounded playbook for Terminal-Bench-2 (Stage 2 of Turbo Meta-harness).

Mirrors `turbo_harness/scienceworld_build_playbook.py` + reuses `turbo_harness/playbook/schemas`.
Stage 1 (meta-harness) is DATA CONSTRUCTION; its exhaust here = the evolution_summary.jsonl —
10 harness variants (tactical/robust/focused/pruning/poll_wait/...) each scored per-task on the
train split, with the proposer's `hypothesis` (what it tried + why) and `outcome` (accepted vs
REGRESSED). We mine this into a playbook a per-instance advisor can use — and crucially its
ANTI-PATTERNS (the variants that REGRESSED = "changes that backfire"), the guard the RL advisor
lacked when it over-patched already-working harnesses.

Contrastive unit = one TASK where a SOLVER harness (pass=1) and a FAILER harness (pass=0) diverge.
Signal = the two variants' scaffold source + their hypotheses. Reflect (opus) -> per-task strategy;
curate (opus) -> clustered strategies + anti-patterns. Regressed-variant hypotheses are fed to the
curator as known anti-patterns.

Usage:
  VERTEXAI_PROJECT=your-gcp-project VERTEXAI_LOCATION=global PYTHONPATH=$PWD \
  .venv/bin/python -m turbo_harness.tb2_build_playbook \
    --run-dir turbo_harness/terminal_bench/logs/sonnet_tb2_evo_20260912_023330 \
    --model vertex_ai/claude-opus-4-6 --max-tasks 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import litellm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from turbo_harness.playbook.schemas import (  # noqa: E402
    ExperienceRecord,
    Playbook,
    PlaybookEntry,
    Reflection,
    StrategyType,
)

AGENTS_DIR = REPO / "turbo_harness/terminal_bench/agents"
MAX_CODE = 6000  # per-variant source cap (KIRA variants ~400-700 lines)

# ── Reflection prompts (TB2-worded) ──────────────────────────────────────────
TB2_REFLECT_SYSTEM = """\
You are analyzing TERMINAL-BENCH-2 HARNESSES: full agent LOOPS (a Terminus-2 / KIRA `AgentHarness`)
wrapped around a FROZEN LLM that solves terminal tasks by calling tools in a tmux terminal. The
harness shapes the SCAFFOLD: poll-wait timing for long-running commands (builds/servers/downloads),
stall/loop detection, the COMPLETION GATE (how it decides the task is done), output-truncation
limits, shell hardening, ret/recovery logic, and task-type guidance in the prompt. It does NOT
change the model.

For ONE task you receive two harness variants — a SOLVER (passed the task) and a FAILER (failed it)
— each with its source and the proposer's hypothesis for what it does.

Your job: explain WHICH scaffold/behavior difference plausibly drove solve-vs-fail on THIS task, and
when that scaffold design helps. Be concrete about the lever (poll-wait duration, stall threshold,
completion-gate strictness, truncation, shell hardening, task hint).

## Rules
- Focus on SCAFFOLD / BEHAVIOR (loop control, timing, completion gate, recovery), NOT model
  capability and NOT prompt-wording tweaks.
- Tie the strategy to the kind of task (build/compile, long server, DB, archive, ML-train, file-edit).
- Concrete + actionable. Generic observations are useless.

## Output format
Respond with ONLY a JSON object:
{{
    "issue_characteristics": ["task condition 1", "task condition 2"],
    "behavioral_difference": "what the solver's scaffold did that the failer's did not",
    "strategy": "a concise, actionable scaffold-design strategy (state the task kind it applies to)",
    "rationale": "WHY it works, grounded in the scaffold difference",
    "confidence": 0.0,
    "outcome": "pass"
}}"""

TB2_REFLECT_USER = """\
## Task id: {task}
{problem_statement}

## SOLVER harness "{pass_harness}" (passed)
hypothesis: {pass_hypothesis}
scaffold source:
{pass_source}

## FAILER harness "{fail_harness}" (failed)
hypothesis: {fail_hypothesis}
scaffold source:
{fail_source}

Which scaffold/behavior difference made "{pass_harness}" pass while "{fail_harness}" failed on THIS \
task, and for what kind of terminal task does that scaffold design help?"""

# ── Curator prompts (TB2-worded) ─────────────────────────────────────────────
TB2_CURATE_SYSTEM = """\
You are a playbook curator for TERMINAL-BENCH-2 HARNESSES (full agent loops that drive a frozen LLM
through tools in a tmux terminal). You receive per-task contrastive strategy reflections (same task,
one harness passed, another failed) AND a list of KNOWN ANTI-PATTERNS: harness changes that
REGRESSED overall pass-rate during evolution. Produce a clean, deduplicated playbook a small
per-instance advisor uses to decide WHEN and HOW to adapt the agent loop for a new task.

## Rules
- Group strategies sharing one underlying idea; give ONE canonical, actionable strategy per cluster,
  the merged CONDITIONS (task kind: build/compile, server, DB, archive, ML-train, file-edit), and
  aggregated evidence.
- CRITICAL — surface ANTI-PATTERNS prominently: scaffold changes whose evidence is mostly harmful,
  PLUS the provided regressed-variant anti-patterns. The advisor's biggest risk is OVER-PATCHING
  harnesses that already work, so include an explicit anti-pattern about not rewriting the
  completion gate / stall logic / core loop unless the task specifically needs it.
- Only scaffold/behavior strategies (timing, completion gate, recovery, truncation, shell setup).
  Drop prompt-rewording and anything that changes the model.

## Output format
Respond with ONLY a JSON object:
{{
    "clusters": [
        {{"strategy": "...", "conditions": ["task=build/compile", "..."], "helpful": <int>,
          "harmful": <int>, "confidence": <float 0-1>, "constituent_ids": ["..."]}}
    ],
    "anti_patterns": ["..."]
}}"""

TB2_CURATE_USER = """\
## Contrastive reflections
{reflections_text}

## Known anti-patterns (harness variants that REGRESSED overall pass-rate)
{anti_text}

Group the {count} reflections into semantic clusters (keyed on task kind). Merge similar strategies,
aggregate evidence, and produce a strong anti-pattern list (include over-patching of already-working
scaffolds)."""

GENERAL_NOTES = [
    "Strategies are grounded in contrastive analysis of the meta-harness evolution: for a given "
    "terminal task, one harness variant passed and another failed.",
    "Adapt the agent loop's poll-wait timing, stall/loop handling, completion-gate strictness, "
    "truncation, and task-kind guidance — NOT prompt wording, NOT the model.",
    "The base (winner) harness already solves most tasks. Do NOT rewrite the completion gate, stall "
    "detector, or core loop unless THIS task's failure mode specifically calls for it; when in doubt "
    "on a task the base already handles, prefer NO_PATCH_NEEDED.",
]


def _strip_json(text: str) -> str:
    text = text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    return text.strip()


def _source(agent_name: str) -> str:
    p = AGENTS_DIR / f"{agent_name}.py"
    return (
        p.read_text()[:MAX_CODE]
        if p.exists()
        else f"(source for '{agent_name}' not found)"
    )


def _passed(v) -> bool:
    return isinstance(v, (int, float)) and v >= 1.0


def build_experiences(
    recs: list[dict], max_tasks: int
) -> tuple[list[ExperienceRecord], list[str]]:
    """Contrastive per-task experiences (solver vs failer) + regressed-variant anti-patterns."""
    # per_task matrix: task -> {agent: pass_rate}
    tasks: set[str] = set()
    for r in recs:
        tasks.update((r.get("per_task") or {}).keys())
    rec_by_agent = {r["agent"]: r for r in recs}

    # rank tasks by how contested they are (both solvers and failers present) — most informative first
    cand = []
    for task in tasks:
        rated = {r["agent"]: (r.get("per_task") or {}).get(task) for r in recs}
        rated = {h: v for h, v in rated.items() if v is not None}
        solvers = [h for h, v in rated.items() if _passed(v)]
        failers = [h for h, v in rated.items() if not _passed(v)]
        if solvers and failers:
            # prefer the winner (poll_wait) as reference on either side for grounding
            solver = (
                "kira_poll_wait_sonnet"
                if "kira_poll_wait_sonnet" in solvers
                else solvers[0]
            )
            failer = failers[0]
            contest = min(len(solvers), len(failers))  # balanced = more informative
            cand.append((contest, task, solver, failer))
    cand.sort(reverse=True)

    records = []
    for (
        _,
        task,
        solver,
        failer,
    ) in cand[:max_tasks]:
        sr, fr = rec_by_agent[solver], rec_by_agent[failer]
        harness_diff = (
            f"SOLVER '{solver}' hypothesis: {sr.get('hypothesis', '')}\n\n"
            f"FAILER '{failer}' hypothesis: {fr.get('hypothesis', '')}"
        )
        records.append(
            ExperienceRecord(
                instance_id=task,
                repo=task,
                problem_statement=f"Terminal-Bench-2 task '{task}'.",
                contrast_type="contrastive",
                pass_harness=solver,
                fail_harness=failer,
                pass_trajectory=sr.get("hypothesis", ""),
                fail_trajectory=fr.get("hypothesis", ""),
                harness_diff=harness_diff,
                delta="improved",
            )
        )

    # regressed variants -> explicit anti-patterns (outcome delta < 0)
    anti = []
    for r in recs:
        d = r.get("delta")
        if isinstance(d, (int, float)) and d < 0:
            anti.append(
                f"'{r['agent']}' REGRESSED {d * 100:.1f}%: {r.get('hypothesis', '')[:300]}"
            )
    return records, anti


def _reflect_one(
    record: ExperienceRecord, rec_by_agent: dict, model_name: str
) -> Reflection:
    sr = rec_by_agent.get(record.pass_harness, {})
    fr = rec_by_agent.get(record.fail_harness, {})
    user = TB2_REFLECT_USER.format(
        task=record.instance_id,
        problem_statement=record.problem_statement,
        pass_harness=record.pass_harness,
        pass_hypothesis=sr.get("hypothesis", ""),
        pass_source=_source(record.pass_harness),
        fail_harness=record.fail_harness,
        fail_hypothesis=fr.get("hypothesis", ""),
        fail_source=_source(record.fail_harness),
    )
    litellm.drop_params = True
    resp = litellm.completion(
        model=model_name,
        messages=[
            {"role": "system", "content": TB2_REFLECT_SYSTEM},
            {"role": "user", "content": user},
        ],
        max_tokens=2048,
        temperature=0.2,
    )
    raw = resp.choices[0].message.content or ""
    try:
        d = json.loads(_strip_json(raw))
    except json.JSONDecodeError:
        return Reflection(
            instance_id=record.instance_id, confidence=0.0, outcome="pass"
        )
    return Reflection(
        instance_id=record.instance_id,
        issue_characteristics=d.get("issue_characteristics", []),
        strategy=d.get("strategy", ""),
        rationale=d.get("rationale", ""),
        confidence=float(d.get("confidence", 0.0)),
        outcome=d.get("outcome", "pass"),
    )


def reflect_batch(records, rec_by_agent, model_name, max_workers) -> list[Reflection]:
    reflections: list[Reflection | None] = [None] * len(records)
    t0 = time.time()
    print(
        f"Reflecting on {len(records)} experiences (model={model_name}, workers={max_workers})"
    )
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {
            pool.submit(_reflect_one, records[i], rec_by_agent, model_name): i
            for i in range(len(records))
        }
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                reflections[i] = fut.result()
            except Exception as e:  # noqa: BLE001
                reflections[i] = Reflection(
                    instance_id=records[i].instance_id,
                    rationale=f"Error: {e}",
                    confidence=0.0,
                    outcome="pass",
                )
            done += 1
            if done % 5 == 0 or done == len(records):
                print(f"  {done}/{len(records)} reflected ({time.time() - t0:.0f}s)...")
    return [r for r in reflections if r is not None]


def _format_reflections(reflections: list[Reflection]) -> tuple[str, int]:
    parts, valid = [], 0
    for i, r in enumerate(reflections):
        if not r.strategy or r.confidence < 0.1:
            continue
        valid += 1
        chars = (
            ", ".join(r.issue_characteristics[:5])
            if r.issue_characteristics
            else "general"
        )
        parts.append(
            f"[{i}] task={r.instance_id}\n    conditions: {chars}\n"
            f"    strategy: {r.strategy}\n    rationale: {r.rationale[:200]}\n"
            f"    confidence: {r.confidence:.2f}"
        )
    return "\n\n".join(parts), valid


def curate(
    reflections: list[Reflection], anti_in: list[str], model_name: str
) -> Playbook:
    reflections_text, valid = _format_reflections(reflections)
    anti_text = "\n".join(f"- {a}" for a in anti_in) or "(none)"
    user = TB2_CURATE_USER.format(
        reflections_text=reflections_text, anti_text=anti_text, count=valid
    )
    litellm.drop_params = True
    resp = litellm.completion(
        model=model_name,
        messages=[
            {"role": "system", "content": TB2_CURATE_SYSTEM},
            {"role": "user", "content": user},
        ],
        max_tokens=8192,
        temperature=0.1,
    )
    try:
        d = json.loads(_strip_json(resp.choices[0].message.content or ""))
    except json.JSONDecodeError:
        d = {"clusters": [], "anti_patterns": []}
    anti_patterns = list(d.get("anti_patterns", []))
    entries = []
    for idx, c in enumerate(d.get("clusters", [])):
        strategy = c.get("strategy", "")
        if not strategy:
            continue
        helpful, harmful = int(c.get("helpful", 0)), int(c.get("harmful", 0))
        if harmful > helpful:
            anti_patterns.append(
                f"AVOID: {strategy} (harmful={harmful}, helpful={helpful})"
            )
            continue
        conditions = c.get("conditions", [])
        entries.append(
            PlaybookEntry(
                strategy_id=f"tb2_ace_{idx:03d}",
                condition=", ".join(conditions[:5]) if conditions else "general",
                strategy=strategy,
                evidence={"helpful": helpful, "harmful": harmful},
                confidence=round(float(c.get("confidence", 0.5)), 3),
                strategy_type=StrategyType.SCAFFOLD_MODIFICATION,
            )
        )
    return Playbook(
        entries=entries, general_notes=GENERAL_NOTES, anti_patterns=anti_patterns
    )


def main():
    ap = argparse.ArgumentParser(
        description="Trajectory-grounded playbook for Terminal-Bench-2"
    )
    ap.add_argument(
        "--run-dir",
        default="turbo_harness/terminal_bench/logs/sonnet_tb2_evo_20260912_023330",
    )
    ap.add_argument("--model", default="vertex_ai/claude-opus-4-6")
    ap.add_argument(
        "--max-tasks", type=int, default=30, help="max contrastive tasks to reflect on"
    )
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--extract-only", action="store_true")
    args = ap.parse_args()

    for var in ("VERTEXAI_PROJECT", "VERTEXAI_LOCATION"):
        if os.environ.get(var):
            setattr(
                litellm,
                "vertex_project" if var.endswith("PROJECT") else "vertex_location",
                os.environ[var],
            )

    run_dir = Path(args.run_dir)
    recs = [json.loads(line) for line in open(run_dir / "evolution_summary.jsonl")]
    rec_by_agent = {r["agent"]: r for r in recs}
    print(f"loaded {len(recs)} evolution iterations: {[r['agent'] for r in recs]}")

    records, anti = build_experiences(recs, args.max_tasks)
    print(
        f"Extracted {len(records)} contrastive task experiences + {len(anti)} regressed anti-patterns"
    )
    for r in records[:10]:
        print(f"  {r.instance_id[:38]:38s} {r.pass_harness} > {r.fail_harness}")

    out = run_dir / "ace" / "playbook.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_name("experiences.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")
    if args.extract_only or not records:
        return

    reflections = reflect_batch(records, rec_by_agent, args.model, args.max_workers)
    with open(out.with_name("reflections.jsonl"), "w") as f:
        for r in reflections:
            f.write(json.dumps(asdict(r)) + "\n")
    n_valid = sum(1 for r in reflections if r.strategy and r.confidence >= 0.1)
    print(f"Reflections -> {n_valid}/{len(reflections)} valid")

    playbook = curate(reflections, anti, args.model)
    playbook.metadata = {
        "domain": "terminal_bench_2",
        "method": "ace-trajectory-grounded",
        "source_run": str(run_dir),
        "n_experiences": len(records),
        "n_reflections_valid": n_valid,
        "curator_model": args.model,
    }
    playbook.save(out)
    print(
        f"\nPlaybook -> {out}: {len(playbook.entries)} strategies, "
        f"{len(playbook.anti_patterns)} anti-patterns"
    )
    for e in playbook.entries:
        print(
            f"\n  [{e.strategy_id}] (conf {e.confidence}, cond: {e.condition})\n    {e.strategy}"
        )
    for a in playbook.anti_patterns:
        print(f"  ANTI: {a}")


if __name__ == "__main__":
    main()
