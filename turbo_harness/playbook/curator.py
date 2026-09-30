"""Curator: LLM-based semantic grouping of reflections into a playbook.

Replaces keyword-overlap dedup with a single LLM call that clusters
semantically similar strategies, merges evidence, and produces a clean
playbook. Keeps quality filters (over-specification blocklist, net-positive).

Usage:
    python -m turbo_harness.playbook.curator \
        --reflections reflections.jsonl \
        --output playbook.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import litellm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from turbo_harness.playbook.schemas import (  # noqa: E402
    Playbook,
    PlaybookEntry,
    Reflection,
    StrategyType,
)

CURATOR_SYSTEM_PROMPT = """\
You are a playbook curator. You receive a list of strategy reflections \
extracted from analyzing agent trajectories under different harness \
configurations. Your job is to group semantically similar strategies into \
clusters and produce a clean, deduplicated playbook.

## Rules

- Group strategies that describe the same underlying idea, even if worded \
differently (e.g., "allocate more budget to verification" and "give more \
steps to test validation" are the same strategy).
- For each cluster, produce ONE canonical strategy with:
  - A clear, actionable description
  - The merged issue characteristics (conditions under which it applies)
  - Aggregated evidence (total helpful + harmful counts from constituent reflections)
  - Average confidence
- Identify anti-patterns: strategies where most evidence is harmful.
- Filter out vague/generic strategies that don't specify conditions.
- Do NOT include strategies that modify has_finished(), add hard gates, \
or hardcode test commands.

## Output format

Respond with a JSON object:
{{
    "clusters": [
        {{
            "strategy": "Canonical strategy description",
            "conditions": ["condition1", "condition2"],
            "helpful": <int>,
            "harmful": <int>,
            "confidence": <float 0-1>,
            "constituent_ids": ["instance_id_1", "instance_id_2"]
        }},
        ...
    ],
    "anti_patterns": [
        "Description of a strategy that is harmful"
    ]
}}"""

CURATOR_USER_TEMPLATE = """\
## Reflections to curate

{reflections_text}

Group these {count} reflections into semantic clusters. Each cluster should \
represent ONE distinct strategy. Merge similar strategies, aggregate evidence, \
and filter noise."""


def _classify_strategy_type(strategy: str) -> StrategyType:
    scaffold_indicators = [
        "step", "phase", "budget", "scaffold", "loop", "control flow",
        "reflection prompt", "progress", "nudge", "checkpoint", "retry",
        "verify", "workflow",
    ]
    strategy_lower = strategy.lower()
    for indicator in scaffold_indicators:
        if indicator in strategy_lower:
            return StrategyType.SCAFFOLD_MODIFICATION
    return StrategyType.TEXT_INJECTION


def _is_overspecification(strategy: str) -> bool:
    bad_patterns = [
        "has_finished", "submission gate", "hardcoded test", "force test",
        "mandatory test", "must run test", "strict validation",
        "test-driven validation",
    ]
    strategy_lower = strategy.lower()
    return any(p in strategy_lower for p in bad_patterns)


def _format_reflections_for_curator(reflections: list[Reflection]) -> str:
    parts = []
    for i, ref in enumerate(reflections):
        if not ref.strategy or ref.confidence < 0.1:
            continue
        chars = ", ".join(ref.issue_characteristics[:5]) if ref.issue_characteristics else "general"
        parts.append(
            f"[{i}] instance={ref.instance_id}\n"
            f"    characteristics: {chars}\n"
            f"    strategy: {ref.strategy}\n"
            f"    rationale: {ref.rationale[:200]}\n"
            f"    confidence: {ref.confidence:.2f}, outcome: {ref.outcome}"
        )
    return "\n\n".join(parts)


def llm_curate(
    reflections: list[Reflection],
    model_name: str = "vertex_ai/claude-opus-4-6",
) -> tuple[list[dict], list[str]]:
    reflections_text = _format_reflections_for_curator(reflections)
    valid_count = sum(1 for r in reflections if r.strategy and r.confidence >= 0.1)

    user = CURATOR_USER_TEMPLATE.format(
        reflections_text=reflections_text,
        count=valid_count,
    )

    litellm.drop_params = True
    response = litellm.completion(
        model=model_name,
        messages=[
            {"role": "system", "content": CURATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        max_tokens=16384,
        temperature=0.1,
    )

    raw = response.choices[0].message.content.strip()
    text = raw
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        print(f"  WARNING: Failed to parse curator response, falling back")
        return [], []

    clusters = d.get("clusters", [])
    anti_patterns = d.get("anti_patterns", [])
    return clusters, anti_patterns


def build_playbook(
    reflections_file: str | Path,
    min_evidence: int = 1,
    curator_model: str = "vertex_ai/claude-opus-4-6",
) -> Playbook:
    reflections = []
    with open(reflections_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            reflections.append(Reflection(**d))

    print(f"Loaded {len(reflections)} reflections")
    valid = [r for r in reflections if r.strategy and r.confidence >= 0.1]
    print(f"Valid reflections: {len(valid)}")

    print(f"Running LLM-based semantic curation (model={curator_model})...")
    clusters, anti_patterns = llm_curate(reflections, curator_model)
    print(f"Curator returned {len(clusters)} clusters, {len(anti_patterns)} anti-patterns")

    entries = []
    for idx, cluster in enumerate(clusters):
        strategy = cluster.get("strategy", "")
        if not strategy or _is_overspecification(strategy):
            continue

        helpful = cluster.get("helpful", 0)
        harmful = cluster.get("harmful", 0)
        if helpful + harmful < min_evidence:
            continue
        if harmful > helpful:
            anti_patterns.append(
                f"AVOID: {strategy} (harmful={harmful}, helpful={helpful})"
            )
            continue

        conditions = cluster.get("conditions", [])
        condition = ", ".join(conditions[:5]) if conditions else "general"
        strategy_type = _classify_strategy_type(strategy)

        entries.append(
            PlaybookEntry(
                strategy_id=f"ace_{idx:03d}",
                condition=condition,
                strategy=strategy,
                evidence={"helpful": helpful, "harmful": harmful},
                confidence=round(cluster.get("confidence", 0.5), 3),
                strategy_type=strategy_type,
            )
        )

    general_notes = [
        "Soft nudges outperform hard gates (reflection prompts +2pp, "
        "submission gates -8pp).",
        "Do not modify has_finished() — over-specification drops pass rate.",
        "Strategies are grounded in contrastive trajectory analysis: "
        "same issue, different harness, different agent behavior.",
    ]

    playbook = Playbook(
        entries=entries,
        general_notes=general_notes,
        anti_patterns=anti_patterns,
        metadata={
            "source": str(reflections_file),
            "total_reflections": len(reflections),
            "valid_reflections": len(valid),
            "clusters": len(clusters),
            "min_evidence": min_evidence,
            "curator_model": curator_model,
        },
    )

    print(f"Playbook: {len(entries)} entries, {len(anti_patterns)} anti-patterns")
    return playbook


def main():
    parser = argparse.ArgumentParser(
        description="Curator: LLM-based playbook curation"
    )
    parser.add_argument("--reflections", required=True)
    parser.add_argument("--output", default="playbook.json")
    parser.add_argument("--min-evidence", type=int, default=1)
    parser.add_argument("--curator-model", default="vertex_ai/claude-opus-4-6")
    args = parser.parse_args()

    playbook = build_playbook(
        args.reflections,
        min_evidence=args.min_evidence,
        curator_model=args.curator_model,
    )
    playbook.save(args.output)
    print(f"Playbook saved to {args.output}")


if __name__ == "__main__":
    main()
