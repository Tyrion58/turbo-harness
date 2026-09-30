"""Patch-based harness advisor: single LLM call generates a unified diff to adapt a harness.

Merges advisor + materializer into one step. The LLM reads the base harness code
and problem statement, then outputs a targeted patch (not a full rewrite).

Usage:
    VERTEXAI_PROJECT=your-gcp-project VERTEXAI_LOCATION=us-east5 \
    python -m turbo_harness.patch_advisor \
        --harness-dir artifacts/multi_repo/haiku/multi_repo_5iter/general \
        --issue-idx 0 --repo multi_repo
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import litellm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "turbo_harness/infra/mini-swe-agent/src"))

from turbo_harness.proposer import validate_artifact, repo_file  # noqa: E402


PATCH_ADVISOR_SYSTEM_PROMPT = """\
You are a harness adaptation engineer for an autonomous bug-fixing agent.

You receive:
1. The FULL source code of a base harness (Python scaffold that orchestrates the agent)
2. A specific bug-fix issue the agent must solve

Your job is to adapt the harness so the agent is more likely to succeed on this issue.

Read the issue to understand what TYPE of problem it is, then adapt the harness's \
SCAFFOLD BEHAVIOR accordingly. Your primary focus should be:

1. **Scaffold logic** (most important): Phase budget reallocation, gate conditions, \
   control flow, submission timing. These are the highest-impact changes.
2. **Brief issue-type guidance** (secondary): Add 1-2 short hints in phase prompts \
   about the TYPE of problem (e.g., "this involves a regression — check git history"). \
   Keep hints general to the issue category, not specific to the solution.

Do NOT write long issue-specific instructions in the phase prompts. Do NOT name \
specific source files, describe the root cause, embed reproduction scripts, or \
tell the agent what commands to run. The agent discovers the solution itself — \
your job is to tune the scaffold that guides its workflow.

## Output format

Output one or more SEARCH/REPLACE blocks:

<<<SEARCH
exact text from the original file
===
replacement text
>>>REPLACE

Each SEARCH section must match EXACTLY one location in the original file — include \
enough surrounding context to be unique. Preserve indentation and whitespace exactly.

If no adaptation is needed, output EXACTLY: NO_PATCH_NEEDED

## Constraints

- The patched harness.py MUST still expose `build_agent(model, env, step_limit, cost_limit)`
- Keep all Jinja2 template variables (e.g. {{task}}) intact
- Output ONLY the SEARCH/REPLACE blocks — no explanation"""


PATCH_ADVISOR_USER_TEMPLATE = """\
## Base harness.py (FULL SOURCE CODE — read carefully)

```python
{harness_code}
```

## Harness Design Rationale
{summary}

## Design Concept (Evolution Lessons)
{concept}

## Repository Knowledge
{rules_memory}

{playbook}
## Bug-Fix Issue

{problem_statement}

---

Produce SEARCH/REPLACE edits that adapt harness.py for this specific issue. \
Focus on changes that help the agent succeed on THIS particular bug. \
Output ONLY the SEARCH/REPLACE blocks."""


def _load_harness_context(harness_dir):
    harness_dir = Path(harness_dir)
    harness_path = harness_dir / "harness.py"
    if not harness_path.exists():
        raise FileNotFoundError(f"No harness.py in {harness_dir}")
    harness_code = harness_path.read_text().strip()

    summary = ""
    concept = ""
    rules_memory = ""
    for name, attr in [
        ("SUMMARY.md", "summary"),
        ("CONCEPT.md", "concept"),
        ("RULES_MEMORY.md", "rules_memory"),
    ]:
        path = harness_dir / name
        if path.exists():
            content = path.read_text().strip()
            if attr == "summary":
                summary = content
            elif attr == "concept":
                concept = content
            elif attr == "rules_memory":
                rules_memory = content

    return harness_code, summary, concept, rules_memory


def _strip_think(text):
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    elif text.startswith("<think>"):
        for marker in ("\n---", "\n```", "\ndiff"):
            if marker in text:
                text = text[text.index(marker) :].strip()
                break
    return text


def _extract_reasoning(response, raw):
    """Best-effort recovery of the editor's reasoning, for display only.

    Tries, in order: a dedicated ``reasoning_content`` field (vLLM reasoning parser); an
    inline ``<think>...</think>`` block; then the prose the editor writes *before* the first
    SEARCH/REPLACE block (the editor reasons about the issue before emitting edits). Returns
    "" if none is present (e.g. a frontier editor that outputs only edits).
    """
    try:
        rc = getattr(response.choices[0].message, "reasoning_content", None)
    except Exception:
        rc = None
    if rc:
        return rc.strip()
    if "<think>" in raw and "</think>" in raw:
        return raw.split("<think>", 1)[1].split("</think>", 1)[0].strip()
    # fall back to the rationale the editor writes before the first edit block
    if "<<<SEARCH" in raw:
        pre = raw.split("<<<SEARCH", 1)[0].replace("<think>", "").strip()
        if len(pre) > 40:  # a substantive rationale, not a stray line
            return pre
    return ""


def _extract_edits(raw):
    """Extract SEARCH/REPLACE blocks from LLM output.

    Returns list of (search, replace) tuples.
    """
    raw = _strip_think(raw)

    if "NO_PATCH_NEEDED" in raw:
        return []

    edits = []
    parts = raw.split("<<<SEARCH")
    for part in parts[1:]:
        if "===\n" not in part or ">>>REPLACE" not in part:
            continue
        search_section, rest = part.split("===\n", 1)
        replace_section = rest.split(">>>REPLACE", 1)[0]
        search = search_section.strip("\n")
        replace = replace_section.strip("\n")
        if search:
            edits.append((search, replace))

    return edits


def generate_patch(
    general_harness_dir,
    problem_statement,
    model_name="vertex_ai/claude-sonnet-4-5@20250929",
    api_base=None,
    max_tokens=16384,
    temperature=0.3,
    playbook_context="",
    return_reasoning=False,
):
    """Generate a unified diff patch to adapt a harness for a specific issue.

    If ``return_reasoning=True`` returns ``(edits, reasoning_text)``, where
    ``reasoning_text`` is the editor's chain-of-thought ("" if none was emitted).
    Otherwise returns just ``edits`` (backward-compatible with existing callers).
    """
    harness_code, summary, concept, rules_memory = _load_harness_context(
        general_harness_dir
    )

    user_content = PATCH_ADVISOR_USER_TEMPLATE.format(
        harness_code=harness_code,
        summary=summary or "(no summary available)",
        concept=concept or "(no concept document available)",
        rules_memory=rules_memory or "(no repository knowledge available)",
        playbook=playbook_context + "\n" if playbook_context else "",
        problem_statement=problem_statement,
    )

    litellm.drop_params = True

    completion_kwargs = {
        "messages": [
            {"role": "system", "content": PATCH_ADVISOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    if api_base:
        completion_kwargs["model"] = model_name
        completion_kwargs["api_base"] = api_base
    else:
        completion_kwargs["model"] = model_name

    response = litellm.completion(**completion_kwargs)
    raw = (response.choices[0].message.content or "").strip()
    edits = _extract_edits(raw)
    if return_reasoning:
        return edits, _extract_reasoning(response, raw)
    return edits


def apply_patch(edits, general_harness_dir, out_dir):
    """Apply search-and-replace edits to create an adapted harness.

    Copies the general harness to out_dir, then applies edits to harness.py.
    Returns (success, message).
    """
    general_harness_dir = Path(general_harness_dir)
    out_dir = Path(out_dir)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(
        general_harness_dir,
        out_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    if not edits:
        return True, "no edits — using base harness"

    harness_path = out_dir / "harness.py"
    content = harness_path.read_text()
    applied = 0
    failed = []

    for i, (search, replace) in enumerate(edits):
        if search in content:
            content = content.replace(search, replace, 1)
            applied += 1
        else:
            failed.append(i)

    harness_path.write_text(content)

    if failed:
        return applied > 0, (
            f"{applied}/{len(edits)} edits applied, "
            f"{len(failed)} failed (blocks: {failed})"
        )
    return True, f"all {applied} edits applied"


def generate_and_apply(
    general_harness_dir,
    problem_statement,
    out_dir,
    model_name="vertex_ai/claude-sonnet-4-5@20250929",
    api_base=None,
    max_tokens=16384,
):
    """Generate edits and apply them, with validation and fallback."""
    general_harness_dir = Path(general_harness_dir)
    out_dir = Path(out_dir)

    # Generate
    try:
        edits = generate_patch(
            general_harness_dir,
            problem_statement,
            model_name=model_name,
            api_base=api_base,
            max_tokens=max_tokens,
        )
    except Exception as e:
        _copy_base(general_harness_dir, out_dir)
        return [], False, f"patch generation failed: {e}"

    # Apply
    success, msg = apply_patch(edits, general_harness_dir, out_dir)
    if not success:
        _copy_base(general_harness_dir, out_dir)
        return edits, False, f"patch apply failed: {msg}"

    # Validate
    ok, val_msg = validate_artifact(out_dir)
    if not ok:
        _copy_base(general_harness_dir, out_dir)
        return edits, False, f"validation failed: {val_msg}"

    return edits, True, msg


def _copy_base(general_harness_dir, out_dir):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(
        general_harness_dir,
        out_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def _load_issue(repo, split, idx):
    data = REPO / f"data/swe_smith/{split}_{repo_file(repo)}.json"
    rows = [json.loads(line) for line in open(data) if line.strip()]
    gt = json.loads(rows[idx]["reward_spec"]["ground_truth_json"])
    return gt


def main():
    parser = argparse.ArgumentParser(description="Generate and apply a harness patch")
    parser.add_argument("--harness-dir", required=True)
    parser.add_argument("--issue-idx", type=int, required=True)
    parser.add_argument("--repo", default="dvc")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", default="vertex_ai/claude-sonnet-4-5@20250929")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    gt = _load_issue(args.repo, args.split, args.issue_idx)
    print(f"Issue: {gt['instance_id']}")
    print(f"Repo: {gt['repo']}")

    out_dir = args.out_dir or tempfile.mkdtemp(prefix="patch_harness_")
    print(f"Output: {out_dir}")

    edits, success, msg = generate_and_apply(
        args.harness_dir,
        gt["problem_statement"],
        out_dir,
        model_name=args.model,
        api_base=args.api_base,
    )

    print(f"\nEdits: {len(edits)} blocks")
    for i, (search, replace) in enumerate(edits):
        print(f"  Block {i}: {len(search)} → {len(replace)} chars")
    print(f"\nResult: {'SUCCESS' if success else 'FALLBACK'} — {msg}")


if __name__ == "__main__":
    main()
