"""Playbook Patch-Advisor: extends patch-advisor with full playbook injection.

The entire playbook is injected into the prompt
context, and the LLM decides which strategies are relevant. No retrieval
or keyword matching — the playbook is small enough to fit in context.

Usage:
    python -m turbo_harness.playbook.memory_advisor \
        --harness-dir artifacts/multi_repo/haiku/multi_repo_5iter/general \
        --playbook playbook.json \
        --issue-idx 0 --repo multi_repo
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "turbo_harness/infra/mini-swe-agent/src"))

from turbo_harness.playbook.schemas import Playbook  # noqa: E402
from turbo_harness.patch_advisor import (  # noqa: E402
    apply_patch,
    generate_patch,
)
from turbo_harness.proposer import validate_artifact  # noqa: E402


def format_playbook_context(playbook: Playbook) -> str:
    if not playbook.entries:
        return ""

    lines = ["## Empirical Playbook (from contrastive trajectory analysis)\n"]
    lines.append(
        "The following strategies were extracted by analyzing agent behavior "
        "on the SAME issues under different harness configurations. Each "
        "strategy is grounded in observed behavioral differences between "
        "passing and failing runs. Apply whichever strategies are relevant "
        "to the current issue.\n"
    )

    if playbook.general_notes:
        lines.append("**General principles:**")
        for note in playbook.general_notes:
            lines.append(f"- {note}")
        lines.append("")

    for i, entry in enumerate(playbook.entries, 1):
        helpful = entry.evidence.get("helpful", 0)
        harmful = entry.evidence.get("harmful", 0)
        total = helpful + harmful
        pct = helpful / max(total, 1) * 100

        lines.append(f"**[{entry.strategy_id}] Strategy {i}**: {entry.strategy}")
        lines.append(f"- Applies when: {entry.condition}")
        lines.append(
            f"- Evidence: {helpful} helpful, {harmful} harmful "
            f"({pct:.0f}% win rate, {total} observations)"
        )
        lines.append(f"- Confidence: {entry.confidence * 100:.0f}%")
        lines.append("")

    if playbook.anti_patterns:
        lines.append("**Anti-patterns (avoid these):**")
        for ap in playbook.anti_patterns:
            lines.append(f"- {ap}")
        lines.append("")

    return "\n".join(lines)


def generate_patch_with_memory(
    general_harness_dir: str | Path,
    problem_statement: str,
    playbook_path: str | Path,
    model_name: str = "vertex_ai/claude-sonnet-4-5@20250929",
    api_base: str | None = None,
    max_tokens: int = 16384,
    temperature: float = 0.3,
) -> list:
    playbook = Playbook.load(playbook_path)
    playbook_context = format_playbook_context(playbook)

    edits = generate_patch(
        general_harness_dir,
        problem_statement,
        model_name=model_name,
        api_base=api_base,
        max_tokens=max_tokens,
        temperature=temperature,
        playbook_context=playbook_context,
    )

    return edits


def _copy_base(general_harness_dir: Path, out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(
        general_harness_dir,
        out_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def generate_and_apply_with_memory(
    general_harness_dir: str | Path,
    problem_statement: str,
    out_dir: str | Path,
    playbook_path: str | Path,
    model_name: str = "vertex_ai/claude-sonnet-4-5@20250929",
    api_base: str | None = None,
    max_tokens: int = 16384,
) -> tuple[list, bool, str]:
    general_harness_dir = Path(general_harness_dir)
    out_dir = Path(out_dir)

    try:
        edits = generate_patch_with_memory(
            general_harness_dir,
            problem_statement,
            playbook_path,
            model_name=model_name,
            api_base=api_base,
            max_tokens=max_tokens,
        )
    except Exception as e:
        _copy_base(general_harness_dir, out_dir)
        return [], False, f"generation failed: {e}"

    success, msg = apply_patch(edits, general_harness_dir, out_dir)
    if not success:
        _copy_base(general_harness_dir, out_dir)
        return edits, False, f"apply failed: {msg}"

    ok, val_msg = validate_artifact(out_dir)
    if not ok:
        _copy_base(general_harness_dir, out_dir)
        return edits, False, f"validation failed: {val_msg}"

    return edits, True, msg


def main():
    parser = argparse.ArgumentParser(description="Playbook Patch-Advisor")
    parser.add_argument("--harness-dir", required=True)
    parser.add_argument("--playbook", required=True)
    parser.add_argument("--issue-idx", type=int, required=True)
    parser.add_argument("--repo", default="multi_repo")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default="vertex_ai/claude-sonnet-4-5@20250929")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    from turbo_harness.patch_advisor import _load_issue

    gt = _load_issue(args.repo, args.split, args.issue_idx)
    print(f"Issue: {gt['instance_id']}")

    out_dir = args.out_dir or tempfile.mkdtemp(prefix="ace_harness_")
    print(f"Output: {out_dir}")

    edits, success, msg = generate_and_apply_with_memory(
        args.harness_dir,
        gt["problem_statement"],
        out_dir,
        args.playbook,
        model_name=args.model,
        api_base=args.api_base,
    )

    print(f"\nEdits: {len(edits)} blocks")
    print(f"Result: {'SUCCESS' if success else 'FALLBACK'} — {msg}")


if __name__ == "__main__":
    main()
