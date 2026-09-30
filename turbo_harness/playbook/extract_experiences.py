"""Extract contrastive experience records from meta-harness evolution trajectories.

For each training issue, loads agent trajectories across all harness variants
and finds the best contrastive pair (one PASS, one FAIL). Includes the FULL
agent trajectories (no summarization).

Usage:
    python -m turbo_harness.playbook.extract_experiences \
        --evolution-log experiments/logs/meta_harness/multi_repo_5iter/evolution_summary.jsonl \
        --trajectory-dir experiments/logs/meta_harness/multi_repo_5iter/trajectories \
        --artifacts-dir artifacts/multi_repo/haiku/multi_repo_5iter \
        --data-file data/swe_smith/train_multi_repo.json \
        --output experiences.jsonl
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from turbo_harness.playbook.schemas import ExperienceRecord  # noqa: E402

HARNESS_NAME_MAP = {
    "baseline": 0,
    "phased_workflow_agent": 1,
    "format_reinforced_hints": 2,
    "test_result_ledger": 3,
    "verify_loop_enforcer": 4,
    "explicit_test_memory": 5,
}


def _serialize_trajectory(messages: list[dict]) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


def _compute_harness_diff(artifacts_dir: Path, harness_a: str, harness_b: str) -> str:
    def _find_harness_py(name: str) -> Path | None:
        if name == "baseline":
            return None
        for d in artifacts_dir.iterdir():
            if d.is_dir() and name in d.name and (d / "harness.py").exists():
                return d / "harness.py"
        return None

    path_a = _find_harness_py(harness_a)
    path_b = _find_harness_py(harness_b)

    if not path_a and not path_b:
        return "(both use default harness)"
    code_a = path_a.read_text() if path_a else "(default harness — no custom harness.py)"
    code_b = path_b.read_text() if path_b else "(default harness — no custom harness.py)"

    if path_a and path_b:
        diff = difflib.unified_diff(
            code_a.splitlines(keepends=True),
            code_b.splitlines(keepends=True),
            fromfile=f"harness.py ({harness_a})",
            tofile=f"harness.py ({harness_b})",
        )
        return "".join(diff) or "(identical)"

    present = path_a or path_b
    label = harness_a if path_a else harness_b
    return f"Only {label} has a custom harness.py:\n{present.read_text()[:3000]}"


def load_trajectories(trajectory_dir: str | Path) -> dict[str, dict[str, dict]]:
    """Load all trajectories: {issue_filename: {harness_name: trajectory_data}}."""
    trajectory_dir = Path(trajectory_dir)
    all_trajs: dict[str, dict[str, dict]] = {}

    for harness_name in sorted(os.listdir(trajectory_dir)):
        harness_path = trajectory_dir / harness_name
        if not harness_path.is_dir():
            continue
        for fname in sorted(os.listdir(harness_path)):
            if not fname.endswith(".jsonl"):
                continue
            # Tolerate a truncated/corrupt trajectory (e.g. a write cut off by a disk-full event):
            # skip it rather than crashing the whole playbook pipeline on one bad file.
            try:
                data = json.loads((harness_path / fname).read_text().split("\n")[0])
            except (json.JSONDecodeError, OSError) as e:
                print(f"  [load_trajectories] skipping corrupt {harness_name}/{fname}: {str(e)[:80]}")
                continue
            all_trajs.setdefault(fname, {})[harness_name] = data

    return all_trajs


def load_evolution_log(log_path: str | Path) -> dict[str, dict[str, float]]:
    """Load per-issue pass rates per iteration: {harness_name: {issue_idx: rate}}."""
    entries = [json.loads(l) for l in open(log_path) if l.strip()]

    harness_rates: dict[str, dict[str, float]] = {}
    for entry in entries:
        agent = entry.get("agent", "baseline")
        per_issue = entry.get("per_issue", {})
        existing = harness_rates.get(agent, {})
        for idx, rate in per_issue.items():
            existing[idx] = max(existing.get(idx, 0), float(rate))
        harness_rates[agent] = existing

    return harness_rates


def _pick_contrastive_pair(
    issue_trajs: dict[str, dict],
) -> tuple[str, str] | None:
    """Pick the best (pass_harness, fail_harness) pair for a contrastive issue."""
    passed = [h for h, d in issue_trajs.items() if d["resolved"]]
    failed = [h for h, d in issue_trajs.items() if not d["resolved"]]

    if not passed or not failed:
        return None

    def _harness_distance(h: str) -> int:
        return HARNESS_NAME_MAP.get(h, 99)

    passed.sort(key=lambda h: -_harness_distance(h))
    failed.sort(key=lambda h: _harness_distance(h))

    return passed[0], failed[0]


def build_experience_records(
    evolution_log: str | Path,
    trajectory_dir: str | Path,
    artifacts_dir: str | Path,
    data_file: str | None = None,
) -> list[ExperienceRecord]:
    artifacts_dir = Path(artifacts_dir)
    all_trajs = load_trajectories(trajectory_dir)
    load_evolution_log(evolution_log)

    instances = {}
    if data_file:
        data_path = Path(data_file)
        if data_path.exists():
            rows = [json.loads(line) for line in open(data_path) if line.strip()]
            for row in rows:
                gt = json.loads(row["reward_spec"]["ground_truth_json"])
                instances[gt["instance_id"]] = gt

    instance_by_filename: dict[str, dict] = {}
    for gt in instances.values():
        fname = gt["instance_id"].replace("/", "_") + ".jsonl"
        if fname in all_trajs:
            instance_by_filename[fname] = gt

    records = []
    for fname, harness_trajs in sorted(all_trajs.items()):
        gt = instance_by_filename.get(fname, {})
        iid = gt.get("instance_id", fname.replace(".jsonl", ""))
        problem = gt.get("problem_statement", "")
        repo = gt.get("repo", "unknown")

        passed = [h for h, d in harness_trajs.items() if d["resolved"]]
        failed = [h for h, d in harness_trajs.items() if not d["resolved"]]

        if passed and failed:
            contrast_type = "contrastive"
            pair = _pick_contrastive_pair(harness_trajs)
            pass_h, fail_h = pair
            pass_data = harness_trajs[pass_h]
            fail_data = harness_trajs[fail_h]

            records.append(
                ExperienceRecord(
                    instance_id=iid,
                    repo=repo,
                    problem_statement=problem,
                    contrast_type=contrast_type,
                    pass_harness=pass_h,
                    fail_harness=fail_h,
                    pass_trajectory=_serialize_trajectory(pass_data["messages"]),
                    fail_trajectory=_serialize_trajectory(fail_data["messages"]),
                    pass_steps=pass_data.get("steps", 0),
                    fail_steps=fail_data.get("steps", 0),
                    fail_status=fail_data.get("status", ""),
                    harness_diff=_compute_harness_diff(artifacts_dir, pass_h, fail_h),
                    delta="improved",
                )
            )
        elif passed:
            contrast_type = "all_pass"
            best_h = passed[0]
            best_data = harness_trajs[best_h]
            records.append(
                ExperienceRecord(
                    instance_id=iid,
                    repo=repo,
                    problem_statement=problem,
                    contrast_type=contrast_type,
                    pass_harness=best_h,
                    pass_trajectory=_serialize_trajectory(best_data["messages"]),
                    pass_steps=best_data.get("steps", 0),
                    delta="maintained",
                )
            )
        else:
            contrast_type = "all_fail"
            worst_h = failed[0]
            worst_data = harness_trajs[worst_h]
            records.append(
                ExperienceRecord(
                    instance_id=iid,
                    repo=repo,
                    problem_statement=problem,
                    contrast_type=contrast_type,
                    fail_harness=worst_h,
                    fail_trajectory=_serialize_trajectory(worst_data["messages"]),
                    fail_steps=worst_data.get("steps", 0),
                    fail_status=worst_data.get("status", ""),
                    delta="maintained",
                )
            )

    return records


def save_records(records: list[ExperienceRecord], output_path: str | Path) -> None:
    from dataclasses import asdict

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec)) + "\n")
    print(f"Saved {len(records)} experience records to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract contrastive experience records from meta-harness trajectories"
    )
    parser.add_argument("--evolution-log", required=True)
    parser.add_argument("--trajectory-dir", required=True)
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--output", default="experiences.jsonl")
    args = parser.parse_args()

    records = build_experience_records(
        args.evolution_log, args.trajectory_dir, args.artifacts_dir, args.data_file
    )
    save_records(records, args.output)

    from collections import Counter

    types = Counter(r.contrast_type for r in records)
    print(f"\nSummary: {len(records)} records")
    print(f"  Contrastive: {types.get('contrastive', 0)}")
    print(f"  All-pass: {types.get('all_pass', 0)}")
    print(f"  All-fail: {types.get('all_fail', 0)}")


if __name__ == "__main__":
    main()
