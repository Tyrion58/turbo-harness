"""Trajectory-grounded playbook for ALFWorld (Stage 2 of Turbo Meta-harness).

Mirrors `turbo_harness/scienceworld_build_playbook.py`. Stage 1 (meta-harness) is DATA
CONSTRUCTION; its exhaust = per-(game, harness) trajectories. Here we mine them into a per-task-type
playbook a per-instance advisor can use.

Contrastive unit = one GAME where a harness that WON and a harness that LOST diverge (ALFWorld reward
is binary won ∈ {0,1}, so the default --min-gap 0.5 selects exactly won-vs-lost pairs). Signal = the
two agents' trajectories (behavior) + the two harness.py sources (scaffold code). Reflect (frontier
model) -> per-task-type strategy; curate -> playbook. The won/lost outcome is embedded per trajectory
(`resolved`); the split file maps each instance to its game path -> its task type (6 ALFWorld types).

Usage:
  VERTEXAI_PROJECT=your-gcp-project VERTEXAI_LOCATION=global PYTHONPATH=$PWD \
  .venv/bin/python -m turbo_harness.alfworld_build_playbook \
    --run-dir experiments/logs/alfworld_meta_harness/alf_stage1_sonnet \
    --model vertex_ai/claude-sonnet-4-5@20250929 --min-gap 0.5 --max-per-task 3
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

MAX_TRAJ = 8000  # ALFWorld episodes ~ up to 50 steps
MAX_CODE = 8000
DEFAULT_HARNESS = REPO / "turbo_harness/alfworld/default_harness/harness.py"
ALF_DATA = Path(os.environ.get("ALFWORLD_DATA", "data/alfworld_data"))

# ALFWorld task-type prefix (first segment of the game dir) -> short label.
_ALF_TYPE = {
    "pick_and_place_simple": "pick_and_place",
    "look_at_obj_in_light": "look_at",
    "pick_clean_then_place_in_recep": "clean",
    "pick_heat_then_place_in_recep": "heat",
    "pick_cool_then_place_in_recep": "cool",
    "pick_two_obj_and_place": "pick_two",
}

# ── Reflection prompts (ALFWorld-worded) ─────────────────────────────────────
ALF_REFLECT_SYSTEM = """\
You are analyzing ALFWORLD HARNESSES: full agent LOOPS (run_episode) wrapped around a FROZEN LLM in
the ALFWorld embodied-household TEXT environment. The agent issues text commands (go to, take, open,
put, clean, heat, cool, examine, use) to complete a household task (6 task types: pick_and_place,
look_at (examine under a desklamp), clean-then-place, heat-then-place, cool-then-place, pick_two).
The harness shapes context/memory management, control flow, retries/repair of inadmissible or
malformed actions, stop logic, subgoal decomposition, and receptacle-search order/memory. It does
NOT change the model. Reward is BINARY: the task is either WON or LOST.

For ONE game you receive: the task; two runs under two harnesses — one that WON and one that LOST —
with trajectories; and the two harnesses' source.

Your job: explain WHICH scaffold/behavior difference drove the win-vs-loss on THIS task, CONDITIONED
ON THE TASK TYPE, and when that scaffold change helps.

## Rules
- Focus on SCAFFOLD / BEHAVIOR: per-task-type subgoal decomposition, receptacle-search order/memory,
  action canonicalization/repair, recovery from "nothing happens" / inadmissible actions, stop logic,
  what's surfaced each step. NOT model capability, NOT prompt-wording tweaks.
- Ground the strategy in the OBSERVED trajectory difference between the two runs.
- Tie the strategy to the TASK TYPE and its procedure (e.g. "heat-then-place: find object -> pick ->
  go to microwave -> heat -> go to target receptacle -> put"; "look_at: find object -> pick -> find
  a desklamp -> use desklamp").
- Concrete + actionable; generic observations are useless.

## Output format
Respond with ONLY a JSON object:
{{
    "issue_characteristics": ["task condition 1", "task condition 2"],
    "behavioral_difference": "what the winning run did that the losing one did not",
    "strategy": "a concise, actionable scaffold-design strategy (state the task type it applies to)",
    "rationale": "WHY it works, grounded in the observed difference",
    "confidence": 0.0,
    "outcome": "pass"
}}"""

ALF_REFLECT_USER = """\
## Task type: {task_type}
## Task
{problem_statement}

## WON — harness "{pass_harness}"
{pass_trajectory}

## LOST — harness "{fail_harness}"
{fail_trajectory}

## Harness sources (scaffold code)
{harness_diff}

What scaffold/behavior difference made "{pass_harness}" WIN where "{fail_harness}" LOST on THIS \
{task_type} task, and for what kind of task does that design help?"""

# ── Curator prompts (ALFWorld-worded) ────────────────────────────────────────
ALF_CURATE_SYSTEM = """\
You are a playbook curator for ALFWORLD HARNESSES (full agent loops for embodied household tasks).
You receive per-game contrastive strategy reflections (same task, one harness WON, another LOST).
Cluster semantically similar strategies into a clean, deduplicated playbook a small per-instance
advisor can use to adapt the agent loop for a new task — keyed by TASK TYPE.

## Rules
- Group strategies sharing one underlying idea, even if worded differently.
- For each cluster give ONE canonical, actionable strategy; the merged CONDITIONS under which it
  applies (KEY THESE ON TASK TYPE where possible); aggregated evidence (helpful/harmful); average
  confidence.
- Identify ANTI-PATTERNS: ideas whose evidence is mostly harmful (caused losses).
- Only scaffold/behavior strategies (per-task subgoal decomposition, receptacle-search order/memory,
  action repair/recovery, stop logic, what's surfaced). Drop prompt-rewording and anything that
  changes the model.

## Output format
Respond with ONLY a JSON object:
{{
    "clusters": [
        {{"strategy": "...", "conditions": ["task-type=... ", "..."], "helpful": <int>, "harmful": <int>,
          "confidence": <float 0-1>, "constituent_ids": ["..."]}}
    ],
    "anti_patterns": ["..."]
}}"""

ALF_CURATE_USER = """\
## Reflections to curate

{reflections_text}

Group these {count} reflections into semantic clusters. Each cluster = ONE distinct scaffold/behavior \
strategy, keyed on task type where possible. Merge similar strategies, aggregate evidence, filter noise."""

GENERAL_NOTES = [
    "Strategies are grounded in contrastive per-game analysis: same ALFWorld task, one harness WON, "
    "another LOST.",
    "Adapt the agent loop's per-task-type SUBGOAL DECOMPOSITION, receptacle-search order/memory, "
    "action repair/recovery, and stop logic — not prompt wording, not the model.",
]


def _strip_json(text: str) -> str:
    text = text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    return text.strip()


def _serialize_trajectory(messages: list[dict], cap: int = MAX_TRAJ) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        parts.append(f"[{role}]\n{content}")
    s = "\n\n".join(parts)
    if len(s) > cap:
        s = s[: cap // 2] + "\n\n...[trajectory truncated]...\n\n" + s[-cap // 2 :]
    return s


_GAMES_CACHE: dict[str, list[str]] = {}


def _games(split_name: str) -> list[str]:
    if split_name not in _GAMES_CACHE:
        data = json.load(open(ALF_DATA / f"{split_name}.json"))
        out: list[str] = []
        for v in data.values():
            out.extend(v)
        _GAMES_CACHE[split_name] = out
    return _GAMES_CACHE[split_name]


def _type_from_path(game_relpath: str) -> str:
    parts = game_relpath.split("/")
    seg = parts[2] if len(parts) > 2 else (parts[-1] if parts else game_relpath)
    prefix = seg.split("-", 1)[0]
    return _ALF_TYPE.get(prefix, prefix)


def _task_type(instance_id: str) -> str:
    """instance_id = '<split_name>_<index>' -> the game's ALFWorld task type (via the split file)."""
    split_name, _, idx = instance_id.rpartition("_")
    try:
        games = _games(split_name)
        return _type_from_path(games[int(idx)])
    except (OSError, KeyError, ValueError, IndexError, json.JSONDecodeError):
        return split_name or "unknown"


def _task_desc(messages: list[dict]) -> str:
    """Task description = the first user message (the reset obs: room state + 'Your task is to: ...')."""
    for m in messages:
        if m.get("role") == "user":
            c = str(m.get("content", ""))
            if c.startswith("Here is your task. "):
                c = c[len("Here is your task. "):]
            # cut off the appended AVAILABLE ACTIONS block if present
            for marker in ("\nAVAILABLE ACTIONS", "AVAILABLE ACTIONS"):
                if marker in c:
                    c = c.split(marker, 1)[0]
                    break
            return c.strip()[:1500]
    return ""


def _harness_source(artifacts_dir: Path, harness_name: str) -> str:
    if harness_name in ("baseline", "default"):
        return DEFAULT_HARNESS.read_text()[:MAX_CODE]
    hp = Path(artifacts_dir) / harness_name / "harness.py"
    return (
        hp.read_text()[:MAX_CODE]
        if hp.exists()
        else f"(harness.py for '{harness_name}' not found)"
    )


def load_trajectories(run_dir: Path) -> dict[str, dict[str, dict]]:
    """{instance_id: {harness_name: full_record}} — each .jsonl is ONE json object."""
    tdir = Path(run_dir) / "trajectories"
    out: dict[str, dict[str, dict]] = {}
    for hdir in sorted(p for p in tdir.iterdir() if p.is_dir()):
        harness = hdir.name
        for f in sorted(hdir.glob("*.jsonl")):
            inst_id = f.stem
            try:
                rec = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(rec, dict) and rec.get("messages"):
                out.setdefault(inst_id, {})[harness] = rec
    return out


def build_experiences(
    trajs, artifacts_dir: Path, min_gap: float, max_per_task: int
) -> list[ExperienceRecord]:
    # candidates per game, grouped by task type for coverage across the 6 ALFWorld task types.
    # ALFWorld reward is binary (won), so rate = 1.0 if resolved else 0.0; min_gap ~0.5 => won-vs-lost.
    by_type: dict[str, list] = {}
    for inst_id, hmap in trajs.items():
        rated = {h: (1.0 if rec.get("resolved") else 0.0) for h, rec in hmap.items()}
        if len(rated) < 2:
            continue
        best, worst = max(rated, key=rated.get), min(rated, key=rated.get)
        gap = rated[best] - rated[worst]
        if gap >= min_gap:
            by_type.setdefault(_task_type(inst_id), []).append(
                (gap, inst_id, best, worst, rated[best], rated[worst])
            )
    cand = []
    for tt, lst in by_type.items():
        lst.sort(reverse=True)
        cand.extend(lst[:max_per_task])
    cand.sort(reverse=True)

    records = []
    for gap, inst_id, best, worst, bs, ws in cand:
        best_rec, worst_rec = trajs[inst_id][best], trajs[inst_id][worst]
        harness_diff = (
            f"### '{best}' (WON) scaffold:\n{_harness_source(artifacts_dir, best)}\n\n"
            f"### '{worst}' (LOST) scaffold:\n{_harness_source(artifacts_dir, worst)}"
        )
        records.append(
            ExperienceRecord(
                instance_id=inst_id,
                repo=_task_type(inst_id),
                problem_statement=_task_desc(best_rec["messages"]),
                contrast_type="contrastive",
                pass_harness=best,
                fail_harness=worst,
                pass_trajectory=f"harness '{best}' outcome = WON\n\n"
                + _serialize_trajectory(best_rec["messages"]),
                fail_trajectory=f"harness '{worst}' outcome = LOST\n\n"
                + _serialize_trajectory(worst_rec["messages"]),
                pass_steps=int(best_rec.get("steps", 0) or 0),
                fail_steps=int(worst_rec.get("steps", 0) or 0),
                fail_status=str(worst_rec.get("status", "")),
                harness_diff=harness_diff,
                delta="improved",
            )
        )
    return records


def _reflect_one(record: ExperienceRecord, model_name: str) -> Reflection:
    user = ALF_REFLECT_USER.format(
        task_type=record.repo,
        problem_statement=record.problem_statement,
        pass_harness=record.pass_harness,
        pass_trajectory=record.pass_trajectory,
        fail_harness=record.fail_harness,
        fail_trajectory=record.fail_trajectory,
        harness_diff=record.harness_diff[:8000],
    )
    litellm.drop_params = True
    resp = litellm.completion(
        model=model_name,
        messages=[
            {"role": "system", "content": ALF_REFLECT_SYSTEM},
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


def reflect_batch(
    records: list[ExperienceRecord], model_name: str, max_workers: int
) -> list[Reflection]:
    reflections: list[Reflection | None] = [None] * len(records)
    t0 = time.time()
    print(
        f"Reflecting on {len(records)} experiences (model={model_name}, workers={max_workers})"
    )
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {
            pool.submit(_reflect_one, records[i], model_name): i
            for i in range(len(records))
        }
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                reflections[i] = fut.result()
            except Exception as e:
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
            f"[{i}] instance={r.instance_id}\n    conditions: {chars}\n"
            f"    strategy: {r.strategy}\n    rationale: {r.rationale[:200]}\n"
            f"    confidence: {r.confidence:.2f}"
        )
    return "\n\n".join(parts), valid


def curate(reflections: list[Reflection], model_name: str) -> Playbook:
    reflections_text, valid = _format_reflections(reflections)
    user = ALF_CURATE_USER.format(reflections_text=reflections_text, count=valid)
    litellm.drop_params = True
    resp = litellm.completion(
        model=model_name,
        messages=[
            {"role": "system", "content": ALF_CURATE_SYSTEM},
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
                strategy_id=f"alf_ace_{idx:03d}",
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
        description="Trajectory-grounded playbook for ALFWorld"
    )
    ap.add_argument(
        "--run-dir", default="experiments/logs/alfworld_meta_harness/alf_stage1_sonnet"
    )
    ap.add_argument(
        "--artifacts-dir",
        default=None,
        help="dir with <harness_name>/harness.py (default: artifacts/alfworld/qwen3-5-9b/<run>)",
    )
    ap.add_argument("--model", default="vertex_ai/claude-sonnet-4-5@20250929")
    ap.add_argument(
        "--min-gap",
        type=float,
        default=0.5,
        help="min outcome gap for a contrastive pair (won=1/lost=0, so 0.5 => won-vs-lost)",
    )
    ap.add_argument("--max-per-task", type=int, default=3)
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
    artifacts_dir = (
        Path(args.artifacts_dir)
        if args.artifacts_dir
        else REPO / "artifacts" / "alfworld" / "qwen3-5-9b" / run_dir.name
    )

    trajs = load_trajectories(run_dir)
    print(
        f"instances with trajectories: {len(trajs)} | harness versions: "
        f"{sorted({h for hm in trajs.values() for h in hm})}"
    )

    records = build_experiences(trajs, artifacts_dir, args.min_gap, args.max_per_task)
    print(
        f"Extracted {len(records)} contrastive experiences (min_gap={args.min_gap}, "
        f"max_per_task={args.max_per_task})"
    )
    for r in records:
        print(f"  {r.instance_id[:36]:36s} [{r.repo}] {r.pass_harness}(WON) > {r.fail_harness}(LOST)")

    out = run_dir / "ace" / "playbook.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_name("experiences.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")
    print(f"Experiences -> {out.with_name('experiences.jsonl')}")
    if args.extract_only or not records:
        return

    reflections = reflect_batch(records, args.model, args.max_workers)
    with open(out.with_name("reflections.jsonl"), "w") as f:
        for r in reflections:
            f.write(json.dumps(asdict(r)) + "\n")
    n_valid = sum(1 for r in reflections if r.strategy and r.confidence >= 0.1)
    print(
        f"Reflections -> {out.with_name('reflections.jsonl')} ({n_valid}/{len(reflections)} valid)"
    )

    playbook = curate(reflections, args.model)
    playbook.metadata = {
        "domain": "alfworld",
        "method": "ace-trajectory-grounded",
        "source_run": str(run_dir),
        "n_experiences": len(records),
        "n_reflections_valid": n_valid,
        "curator_model": args.model,
        "min_gap": args.min_gap,
        "max_per_task": args.max_per_task,
    }
    playbook.save(out)
    print(
        f"\nPlaybook -> {out}: {len(playbook.entries)} strategies, {len(playbook.anti_patterns)} anti-patterns"
    )
    for e in playbook.entries:
        print(
            f"\n  [{e.strategy_id}] (conf {e.confidence}, cond: {e.condition})\n    {e.strategy}"
        )
    for a in playbook.anti_patterns:
        print(f"  ANTI: {a}")


if __name__ == "__main__":
    main()
