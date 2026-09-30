"""Playbook end-to-end pipeline orchestrator.

Runs the full offline pipeline on training trajectory data:
  Step 1: extract contrastive experience pairs from trajectories
  Step 2: reflect on experiences (contrastive trajectory analysis)
  Step 3: curate playbook (LLM-based semantic grouping)

Output is saved to playbook_output/{run_name}/ where run_name defaults to a
timestamp. Re-running the pipeline creates a new directory, preserving
previous results.

Usage:
    python -m turbo_harness.playbook.run_pipeline \
        --evolution-log experiments/logs/meta_harness/multi_repo_5iter/evolution_summary.jsonl \
        --trajectory-dir experiments/logs/meta_harness/multi_repo_5iter/trajectories \
        --artifacts-dir artifacts/multi_repo/haiku/multi_repo_5iter \
        --data-file data/swe_smith/train_multi_repo.json \
        --output-dir playbook_output/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from turbo_harness.playbook.curator import build_playbook  # noqa: E402
from turbo_harness.playbook.extract_experiences import (  # noqa: E402
    build_experience_records,
    save_records,
)
from turbo_harness.playbook.reflector import batch_reflect  # noqa: E402
from turbo_harness.playbook.schemas import StrategyType  # noqa: E402


def run_pipeline(
    evolution_log: str,
    trajectory_dir: str,
    artifacts_dir: str,
    output_dir: str,
    data_file: str | None = None,
    reflector_model: str = "vertex_ai/claude-opus-4-6",
    curator_model: str = "vertex_ai/claude-opus-4-6",
    reflector_workers: int = 5,
    min_evidence: int = 1,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    experiences_path = out_path / "experiences.jsonl"
    reflections_path = out_path / "reflections.jsonl"
    playbook_path = out_path / "playbook.json"

    print(f"{'=' * 60}")
    print("Step 1: Extract contrastive experience pairs")
    print(f"{'=' * 60}")
    records = build_experience_records(
        evolution_log, trajectory_dir, artifacts_dir, data_file
    )
    save_records(records, experiences_path)

    from collections import Counter

    types = Counter(r.contrast_type for r in records)
    print(f"  {types.get('contrastive', 0)} contrastive, "
          f"{types.get('all_pass', 0)} all-pass, "
          f"{types.get('all_fail', 0)} all-fail")

    print(f"\n{'=' * 60}")
    print("Step 2: Contrastive trajectory reflection")
    print(f"{'=' * 60}")
    batch_reflect(
        experiences_path,
        reflections_path,
        model_name=reflector_model,
        max_workers=reflector_workers,
    )

    print(f"\n{'=' * 60}")
    print("Step 3: LLM-based semantic curation")
    print(f"{'=' * 60}")
    playbook = build_playbook(
        reflections_path,
        min_evidence=min_evidence,
        curator_model=curator_model,
    )
    playbook.save(playbook_path)

    print(f"\n{'=' * 60}")
    print("Playbook Summary")
    print(f"{'=' * 60}")
    print(f"\nEntries: {len(playbook.entries)}")

    n_scaffold = sum(
        1 for e in playbook.entries
        if e.strategy_type == StrategyType.SCAFFOLD_MODIFICATION
    )
    n_text = sum(
        1 for e in playbook.entries
        if e.strategy_type == StrategyType.TEXT_INJECTION
    )
    print(f"Strategy types: {n_scaffold} scaffold, {n_text} text-injection")

    print("\nStrategies:")
    for entry in playbook.entries:
        helpful = entry.evidence.get("helpful", 0)
        harmful = entry.evidence.get("harmful", 0)
        print(f"  [{entry.strategy_id}] {entry.strategy[:80]}")
        print(f"    condition: {entry.condition[:60]}")
        print(f"    evidence: {helpful}h/{harmful}harm, conf: {entry.confidence:.2f}")

    if playbook.anti_patterns:
        print(f"\nAnti-patterns ({len(playbook.anti_patterns)}):")
        for ap in playbook.anti_patterns[:5]:
            print(f"  - {ap[:100]}")

    config = {
        "evolution_log": str(evolution_log),
        "trajectory_dir": str(trajectory_dir),
        "artifacts_dir": str(artifacts_dir),
        "data_file": str(data_file) if data_file else None,
        "reflector_model": reflector_model,
        "curator_model": curator_model,
        "reflector_workers": reflector_workers,
        "min_evidence": min_evidence,
        "playbook_entries": len(playbook.entries),
        "anti_patterns": len(playbook.anti_patterns),
        "total_reflections": len(records),
    }
    config_path = out_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2))

    print(f"\nPipeline complete. Output: {out_path}")
    print(f"  config:      {config_path}")
    print(f"  experiences: {experiences_path}")
    print(f"  reflections: {reflections_path}")
    print(f"  playbook:    {playbook_path}")

    return playbook


def main():
    parser = argparse.ArgumentParser(description="Playbook end-to-end pipeline")
    parser.add_argument("--evolution-log", required=True)
    parser.add_argument("--trajectory-dir", required=True)
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--output-dir", default="playbook_output")
    parser.add_argument("--run-name", default=None,
                        help="Run name (default: auto-generated timestamp)")
    parser.add_argument("--reflector-model", default="vertex_ai/claude-opus-4-6")
    parser.add_argument("--curator-model", default="vertex_ai/claude-opus-4-6")
    parser.add_argument("--reflector-workers", type=int, default=5)
    parser.add_argument("--min-evidence", type=int, default=1)
    args = parser.parse_args()

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = str(Path(args.output_dir) / run_name)

    run_pipeline(
        args.evolution_log,
        args.trajectory_dir,
        args.artifacts_dir,
        output_dir,
        data_file=args.data_file,
        reflector_model=args.reflector_model,
        curator_model=args.curator_model,
        reflector_workers=args.reflector_workers,
        min_evidence=args.min_evidence,
    )


if __name__ == "__main__":
    main()
