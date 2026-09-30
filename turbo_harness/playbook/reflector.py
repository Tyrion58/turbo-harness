"""Reflector: contrastive trajectory analysis.

Analyzes FULL agent trajectories under different harnesses for the same issue.
For contrastive issues (pass under one harness, fail under another), compares
both trajectories side-by-side to identify behavioral differences and extract
issue-conditional strategies.

Usage:
    python -m turbo_harness.playbook.reflector \
        --experiences experiences.jsonl \
        --output reflections.jsonl \
        --model vertex_ai/claude-opus-4-6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import litellm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from turbo_harness.playbook.schemas import ExperienceRecord, Reflection  # noqa: E402


CONTRASTIVE_SYSTEM_PROMPT = """\
You are a harness evolution analyst. You analyze how different harness \
configurations affect a bug-fixing agent's behavior on specific issues.

You will receive the FULL agent trajectories for the same issue under two \
different harnesses — one where the agent succeeded and one where it failed. \
Your job is to compare the agent's behavior and explain WHY the harness \
difference led to a different outcome for THIS specific issue.

## Critical rules

- Focus on BEHAVIORAL differences: what the agent did differently, not just \
what the harness changed. Cite specific steps and actions from the trajectories.
- Identify issue-SPECIFIC characteristics that made the difference matter. \
Generic observations are not useful.
- Produce ACTIONABLE strategies: what should a harness do for similar issues?
- Do NOT recommend modifying has_finished(), adding strict submission gates, \
or hardcoding test commands. These are proven anti-patterns.
- Soft nudges > hard gates.

## Output format

Respond with a JSON object:
{{
    "issue_characteristics": ["characteristic1", "characteristic2", ...],
    "behavioral_difference": "What the agent did differently under each harness",
    "strategy": "A concise, actionable strategy for adapting harnesses for similar issues",
    "rationale": "WHY this strategy works — grounded in the observed behavioral difference",
    "confidence": 0.0-1.0,
    "outcome": "pass"
}}"""


UNIFORM_SYSTEM_PROMPT = """\
You are a harness evolution analyst. You analyze agent behavior on issues \
where all harness variants produced the same outcome.

You will receive a single agent trajectory. Your job is to explain what \
about this issue made it consistently {outcome_word} regardless of harness, \
and what strategy (if any) would help for similar issues.

## Critical rules

- Identify issue-SPECIFIC characteristics that explain the uniform outcome.
- For all-pass issues: what makes this issue robust to harness variation?
- For all-fail issues: what makes this issue resistant to harness optimization?
- Do NOT recommend modifying has_finished() or adding hard gates.

## Output format

Respond with a JSON object:
{{
    "issue_characteristics": ["characteristic1", "characteristic2", ...],
    "behavioral_difference": "Why all harnesses produced the same outcome",
    "strategy": "Strategy for similar issues (or 'no adaptation needed')",
    "rationale": "WHY this issue is harness-agnostic or harness-resistant",
    "confidence": 0.0-1.0,
    "outcome": "{outcome_tag}"
}}"""


CONTRASTIVE_USER_TEMPLATE = """\
## Bug-Fix Issue

{problem_statement}

## PASS: Agent trajectory under "{pass_harness}" harness

{pass_trajectory}

Result: PASS in {pass_steps} steps

## FAIL: Agent trajectory under "{fail_harness}" harness

{fail_trajectory}

Result: FAIL ({fail_status}) in {fail_steps} steps

## Harness difference

{harness_diff}

Compare the agent's behavior under both harnesses. What specific behavioral \
difference led to success vs failure on THIS issue?"""


UNIFORM_USER_TEMPLATE = """\
## Bug-Fix Issue

{problem_statement}

## Agent trajectory under "{harness_name}" harness

{trajectory}

Result: {outcome} in {steps} steps

This issue consistently {outcome_word} across all harness variants. \
What about this issue explains the uniform outcome?"""


def _format_contrastive_prompt(record: ExperienceRecord) -> list[dict]:
    user = CONTRASTIVE_USER_TEMPLATE.format(
        problem_statement=record.problem_statement[:3000],
        pass_harness=record.pass_harness,
        pass_trajectory=record.pass_trajectory,
        pass_steps=record.pass_steps,
        fail_harness=record.fail_harness,
        fail_trajectory=record.fail_trajectory,
        fail_status=record.fail_status or "unknown",
        fail_steps=record.fail_steps,
        harness_diff=record.harness_diff[:4000],
    )
    return [
        {"role": "system", "content": CONTRASTIVE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _format_uniform_prompt(record: ExperienceRecord) -> list[dict]:
    is_pass = record.contrast_type == "all_pass"
    outcome_word = "passed" if is_pass else "failed"
    outcome_tag = "pass" if is_pass else "fail"

    system = UNIFORM_SYSTEM_PROMPT.replace("{outcome_word}", outcome_word)
    system = system.replace("{outcome_tag}", outcome_tag)

    trajectory = record.pass_trajectory or record.fail_trajectory
    harness_name = record.pass_harness or record.fail_harness
    steps = record.pass_steps or record.fail_steps
    outcome = "PASS" if is_pass else f"FAIL ({record.fail_status or 'unknown'})"

    user = UNIFORM_USER_TEMPLATE.format(
        problem_statement=record.problem_statement[:3000],
        harness_name=harness_name,
        trajectory=trajectory,
        outcome=outcome,
        steps=steps,
        outcome_word=outcome_word,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _parse_reflection_response(raw: str, record: ExperienceRecord) -> Reflection:
    text = raw.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        return Reflection(
            instance_id=record.instance_id,
            issue_characteristics=[],
            strategy="",
            rationale=f"Parse error: {raw[:200]}",
            confidence=0.0,
            outcome=record.delta,
        )

    return Reflection(
        instance_id=record.instance_id,
        issue_characteristics=d.get("issue_characteristics", []),
        strategy=d.get("strategy", ""),
        rationale=d.get("rationale", ""),
        confidence=float(d.get("confidence", 0.0)),
        outcome=d.get("outcome", record.delta),
    )


def validate_reflection(reflection: Reflection) -> tuple[bool, str]:
    if not reflection.strategy:
        return False, "empty strategy"
    if not reflection.issue_characteristics:
        return False, "no issue characteristics identified"

    bad_patterns = ["has_finished", "submission gate", "hardcoded test"]
    strategy_lower = reflection.strategy.lower()
    for pattern in bad_patterns:
        if pattern in strategy_lower:
            return False, f"over-specification: mentions '{pattern}'"

    if reflection.confidence > 1.0 or reflection.confidence < 0.0:
        return False, f"invalid confidence: {reflection.confidence}"

    return True, "ok"


def reflect_on_experience(
    record: ExperienceRecord,
    model_name: str = "vertex_ai/claude-opus-4-6",
) -> Reflection:
    if record.contrast_type == "contrastive":
        messages = _format_contrastive_prompt(record)
    else:
        messages = _format_uniform_prompt(record)

    litellm.drop_params = True
    response = litellm.completion(
        model=model_name,
        messages=messages,
        max_tokens=2048,
        temperature=0.2,
    )

    raw = response.choices[0].message.content.strip()
    reflection = _parse_reflection_response(raw, record)

    valid, _ = validate_reflection(reflection)
    if not valid:
        reflection.confidence = 0.0

    return reflection


def batch_reflect(
    experiences_file: str | Path,
    output_file: str | Path,
    model_name: str = "vertex_ai/claude-opus-4-6",
    max_workers: int = 5,
) -> list[Reflection]:
    records = []
    with open(experiences_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            records.append(ExperienceRecord(**d))

    from collections import Counter

    types = Counter(r.contrast_type for r in records)
    print(
        f"Reflecting on {len(records)} experiences "
        f"({types.get('contrastive', 0)} contrastive, "
        f"{types.get('all_pass', 0)} all-pass, "
        f"{types.get('all_fail', 0)} all-fail) "
        f"model={model_name}, workers={max_workers}"
    )

    reflections: list[Reflection | None] = [None] * len(records)
    t0 = time.time()

    def reflect_one(idx: int) -> tuple[int, Reflection]:
        return idx, reflect_on_experience(records[idx], model_name)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(reflect_one, i): i for i in range(len(records))}
        done = 0
        for fut in as_completed(futs):
            idx = futs[fut]
            try:
                _, ref = fut.result()
                reflections[idx] = ref
            except Exception as e:
                reflections[idx] = Reflection(
                    instance_id=records[idx].instance_id,
                    rationale=f"Error: {e}",
                    confidence=0.0,
                    outcome=records[idx].delta,
                )
            done += 1
            if done % 10 == 0 or done == len(records):
                elapsed = time.time() - t0
                print(f"  {done}/{len(records)} reflected ({elapsed:.0f}s)...")

    elapsed = time.time() - t0
    print(
        f"Reflection done: {elapsed:.1f}s ({elapsed / max(len(records), 1):.1f}s/record)"
    )

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as out:
        for entry in reflections:
            if entry is not None:
                out.write(json.dumps(asdict(entry)) + "\n")

    valid = sum(1 for r in reflections if r and r.strategy)
    print(f"Saved {len(reflections)} reflections ({valid} valid) to {output_path}")
    return [r for r in reflections if r is not None]


def main():
    parser = argparse.ArgumentParser(
        description="Reflector: contrastive trajectory analysis"
    )
    parser.add_argument("--experiences", required=True)
    parser.add_argument("--output", default="reflections.jsonl")
    parser.add_argument("--model", default="vertex_ai/claude-opus-4-6")
    parser.add_argument("--max-workers", type=int, default=5)
    args = parser.parse_args()

    batch_reflect(args.experiences, args.output, args.model, args.max_workers)


if __name__ == "__main__":
    main()
