"""Generate CONCEPT.md from meta-harness evolution data.

Passes the CONCEPT_TEMPLATE.md and data paths to Claude Code,
which reads all files and writes CONCEPT.md.

Usage:
  python -m turbo_harness.generate_concept \
      --run-name dvc-opus-50-fixed-eval \
      --repo dvc \
      --student haiku
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from turbo_harness import claude_wrapper as cw  # noqa: E402

ARTIFACTS = REPO / "artifacts"
TEMPLATE = Path(__file__).resolve().parent / "CONCEPT_TEMPLATE.md"


def generate_concept(run_name, repo, student_label):
    student_dir = "haiku" if "haiku" in student_label else student_label
    run_artifacts = ARTIFACTS / repo / student_dir / run_name
    logs_dir = REPO / "experiments" / "logs" / "meta_harness" / run_name
    summary_path = logs_dir / "evolution_summary.jsonl"
    traj_dir = logs_dir / "trajectories"

    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found")
        return None

    frontier_path = logs_dir / "frontier.json"
    best_dir = None
    if frontier_path.exists():
        import json
        frontier = json.loads(frontier_path.read_text())
        best_dir = frontier.get("best_dir")

    if best_dir:
        out_path = Path(best_dir) / "CONCEPT.md"
    else:
        out_path = run_artifacts / "general" / "CONCEPT.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    template_text = TEMPLATE.read_text()

    prompt = f"""{template_text}

## Data locations

- Evolution summary: `{summary_path}`
- Trajectories: `{traj_dir}/`
- Harness artifacts: `{run_artifacts}/` (each iter has harness.py, SUMMARY.md)

## Output

Write the CONCEPT.md to: `{out_path}`
"""

    print("Generating CONCEPT.md via Claude Code...")
    print(f"  Data: {logs_dir}")
    print(f"  Output: {out_path}")

    res = cw.run(
        prompt,
        model="opus",
        allowed_tools=["Read", "Write", "Bash", "Glob", "Grep"],
        cwd=str(REPO),
        log_dir=str(REPO / "experiments" / "logs" / "concept_generation"),
        name="generate-concept",
        progress=False,
        timeout_seconds=600,
        effort="high",
    )

    if out_path.exists():
        print(f"\nCONCEPT.md generated: {out_path} ({len(out_path.read_text())} chars)")
    else:
        print("\nWARNING: CONCEPT.md not generated. Check logs.")

    print(f"Claude cost: ${res.cost_usd:.4f}")

    general_path = run_artifacts / "general" / "CONCEPT.md"
    if out_path.exists() and general_path != out_path and general_path.parent.exists():
        import shutil
        shutil.copy2(out_path, general_path)

    return out_path


def main():
    ap = argparse.ArgumentParser(description="Generate CONCEPT.md from evolution data")
    ap.add_argument("--run-name", required=True, help="Meta-harness run name")
    ap.add_argument("--repo", default="dvc")
    ap.add_argument("--student", default="haiku")
    args = ap.parse_args()

    generate_concept(args.run_name, args.repo, args.student)


if __name__ == "__main__":
    main()
