"""Playbook-advisor comparison report generator.

Loads results from playbook-advisor eval runs and baseline results, produces a
markdown comparison report.

Usage:
    python -m turbo_harness.playbook.generate_report \
        --playbook-results ace_eval_results.json \
        --baseline-rate 28.33 \
        --general-rate 42.22 \
        --patch-advisor-rate 45.33 \
        --output ace_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def generate_report(
    ace_results_path: str | Path,
    baseline_rate: float = 28.33,
    general_rate: float = 42.22,
    patch_advisor_rate: float | None = None,
    output_path: str | Path | None = None,
) -> str:
    ace_data = json.loads(Path(ace_results_path).read_text())

    stats = ace_data.get("stats", {})
    playbook_mean = stats.get("mean", 0) * 100
    playbook_sem = stats.get("sem", 0) * 100
    per_run_rates = ace_data.get("per_run_rates", [])
    lines = [
        "# Playbook Advisor — Evaluation Report\n",
        "## Results Comparison\n",
        "| Method | Pass Rate (mean +/- SEM) | Delta vs Baseline | Delta vs General |",
        "|---|---|---|---|",
        f"| **Baseline** | {baseline_rate:.2f}% | — | "
        f"{baseline_rate - general_rate:+.2f} pts |",
        f"| **General harness** | {general_rate:.2f}% | "
        f"{general_rate - baseline_rate:+.2f} pts | — |",
    ]

    if patch_advisor_rate is not None:
        lines.append(
            f"| **Patch-advisor** | {patch_advisor_rate:.2f}% | "
            f"{patch_advisor_rate - baseline_rate:+.2f} pts | "
            f"{patch_advisor_rate - general_rate:+.2f} pts |"
        )

    lines.append(
        f"| **playbook advisor** | {playbook_mean:.2f} +/- {playbook_sem:.2f}% | "
        f"{playbook_mean - baseline_rate:+.2f} pts | "
        f"{playbook_mean - general_rate:+.2f} pts |"
    )
    lines.append("")

    lines.append("## Per-Run Results\n")
    for i, rate in enumerate(per_run_rates):
        lines.append(f"- Run {i + 1}: {rate * 100:.2f}%")
    lines.append("")

    pb_stats = ace_data.get("playbook_stats", {})
    if pb_stats:
        lines.append("## Playbook Utilization\n")
        lines.append(
            f"- Avg strategies matched per issue: "
            f"{pb_stats.get('avg_strategies_per_issue', 0):.1f}"
        )
        lines.append(f"- Issues with matches: {pb_stats.get('issues_with_matches', 0)}")
        lines.append(f"- Total matches: {pb_stats.get('total_matches', 0)}")

        match_counts = pb_stats.get("strategy_match_counts", {})
        if match_counts:
            lines.append("\n### Top Matched Strategies\n")
            sorted_counts = sorted(
                match_counts.items(), key=lambda x: x[1], reverse=True
            )[:10]
            for sid, count in sorted_counts:
                lines.append(f"- `{sid}`: matched {count} times")
        lines.append("")

    lines.append("## Success Criteria\n")
    target = 47.00
    gap_closed = (playbook_mean - general_rate) / max(56.00 - general_rate, 0.01) * 100
    lines.append(
        f"- Target: >= {target:.2f}% pass rate (closes >= 33% of frontier gap)"
    )
    lines.append(f"- Achieved: {playbook_mean:.2f}%")
    lines.append(
        f"- Gap closed: {gap_closed:.1f}% of {56.00 - general_rate:.2f} pt frontier gap"
    )
    met = playbook_mean >= target
    lines.append(f"- Status: **{'MET' if met else 'NOT MET'}**")
    lines.append("")

    report = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(report)
        print(f"Report saved to {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Generate playbook-advisor comparison report")
    parser.add_argument("--playbook-results", required=True)
    parser.add_argument("--baseline-rate", type=float, default=28.33)
    parser.add_argument("--general-rate", type=float, default=42.22)
    parser.add_argument("--patch-advisor-rate", type=float, default=None)
    parser.add_argument("--output", default="ace_report.md")
    args = parser.parse_args()

    report = generate_report(
        args.playbook_results,
        baseline_rate=args.baseline_rate,
        general_rate=args.general_rate,
        patch_advisor_rate=args.patch_advisor_rate,
        output_path=args.output,
    )
    print(report)


if __name__ == "__main__":
    main()
