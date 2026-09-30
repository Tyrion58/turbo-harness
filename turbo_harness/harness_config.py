"""Structured harness configuration.

Defines the harness as a JSON with fixed dimensions but free-form values.
The advisor fills in each dimension per-instance. The materializer implements it.

Dimensions:
  workflow         — step-by-step workflow for the agent
  context_strategy — how to manage conversation context
  submission_strategy — when/how the agent should submit
  hints            — repo/issue-type knowledge to inject
  budget_guidance  — how to manage the step budget
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class HarnessConfig:
    workflow: str
    context_strategy: str
    submission_strategy: str
    hints: str
    budget_guidance: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "HarnessConfig":
        d = json.loads(s)
        return cls(**{k: d.get(k, "") for k in cls.__dataclass_fields__})

    def to_prompt(self) -> str:
        """Format config for the materializer prompt."""
        lines = []
        for key, val in asdict(self).items():
            lines.append(f"### {key}\n{val}")
        return "\n\n".join(lines)


DEFAULT_CONFIG = HarnessConfig(
    workflow=(
        "Start by reading the issue description carefully to understand what needs to "
        "be fixed. Use grep and find commands to locate relevant files (limit initial "
        "exploration to 5-8 read commands). Once you identify the likely location, make "
        "a targeted edit using sed -i or echo with redirection. After editing, run the "
        "repository's test suite to verify your fix works. If tests pass, submit "
        "immediately. If tests fail, analyze the failure output and iterate with at most "
        "one more edit attempt before trying a completely different approach. Do not "
        "spend more than 10 steps exploring without making an edit."
    ),
    context_strategy=(
        "Keep conversation history focused on the current fix attempt. After every 10 "
        "steps, explicitly summarize what you've learned and what action you'll take "
        "next. If you find yourself reading the same files or running similar grep "
        "commands multiple times, stop exploring and commit to an edit based on what "
        "you already know. Avoid accumulating long chains of exploration commands - the "
        "structured workflow (analyze → edit → verify → test → submit) should guide "
        "your message flow."
    ),
    submission_strategy=(
        "Before submitting, verify that: (1) you have made at least one code edit to "
        "address the issue, (2) you have run tests and they pass, or you have strong "
        "evidence the fix is correct if tests are unavailable, (3) your edit directly "
        "addresses the problem described in the issue. Do not submit after only "
        "exploration steps. Do not submit if you haven't attempted a fix. If you're "
        "unsure, make one more targeted edit rather than submitting prematurely."
    ),
    hints=(
        "This repository uses standard Python testing (pytest or similar). Test files "
        "are typically in a ./tests/ directory. The codebase structure requires "
        "understanding multiple files before knowing which test applies - do not force "
        "yourself to run tests before understanding the code. If you notice you're "
        "running the same grep/find/cat commands on files you've already examined, "
        "that's a signal to stop exploring and start editing. Issues with explicit "
        "reproduction code in the description benefit from running that code first to "
        "confirm the bug. Simple localized fixes often need minimal exploration (3-5 "
        "files). Structural or refactoring issues may need broader understanding but "
        "should still result in edits within 15-20 steps."
    ),
    budget_guidance=(
        "Allocate roughly 30% of your budget to initial exploration (reading files, "
        "understanding structure), 20% to making edits, 30% to testing and "
        "verification, and reserve 20% for iteration if the first fix fails. In a "
        "40-step budget, this means: 12 steps exploring, 8 steps editing, 12 steps "
        "testing, 8 steps for recovery. If you hit step 20 without making an edit, "
        "force yourself to commit to a fix based on current knowledge. If you hit step "
        "30 without a passing test, submit your best attempt rather than continuing to "
        "explore. Avoid spending more than 15 consecutive steps on exploration without "
        "editing - this is the primary failure mode."
    ),
)


DIMENSIONS_DESCRIPTION = """\
A harness config has 5 dimensions. For each dimension, describe what the harness \
should do for THIS specific issue. Be specific and actionable.

### workflow
The step-by-step workflow the agent should follow. What should it do first? \
What tests to run? What files to look at? In what order?

### context_strategy
How to manage the agent's conversation context as it grows. Keep full history? \
Truncate old messages? Summarize? This affects whether the agent "remembers" \
earlier exploration or starts fresh each step.

### submission_strategy
When and how the agent should submit its fix. Should it verify tests pass first? \
Should empty submissions be rejected? Should there be a review step?

### hints
Repository-specific or issue-type-specific knowledge to give the agent. \
Relevant directories, test patterns, common pitfalls, useful commands.

### budget_guidance
How to manage the 40-step budget. How many steps for exploration vs editing? \
Is this a simple fix (act fast) or complex (allow more exploration)?"""


def generate_default_config(harness_dir, model="vertex_ai/claude-sonnet-4-5@20250929"):
    """Read harness.py + SUMMARY.md and generate a matching default_config.json.

    Called automatically after meta-harness evolution creates a new harness,
    or standalone to backfill existing harnesses.
    """
    import litellm

    harness_dir = Path(harness_dir)
    harness_code = (harness_dir / "harness.py").read_text()
    summary = ""
    if (harness_dir / "SUMMARY.md").exists():
        summary = (harness_dir / "SUMMARY.md").read_text()

    concept = ""
    if (harness_dir / "CONCEPT.md").exists():
        concept = (harness_dir / "CONCEPT.md").read_text()

    prompt = (
        "Read this harness code, summary, and concept document. Then produce a JSON "
        "object with exactly 5 keys: workflow, context_strategy, submission_strategy, "
        "hints, budget_guidance.\n\n"
        "IMPORTANT: Write each value as a TASK-LEVEL STRATEGY for a bug-fixing agent, "
        "NOT as a description of the Python code. Think of it as: 'what should the "
        "agent DO to fix a typical bug in this repo?' The output will be shown to an "
        "advisor model as a starting point to adapt per-issue.\n\n"
        "Guidelines:\n"
        "- workflow: concrete step-by-step bug-fixing workflow with example commands\n"
        "- context_strategy: how should the agent manage its conversation history\n"
        "- submission_strategy: what should the agent verify before submitting\n"
        "- hints: repo-specific knowledge (directory structure, test commands, patterns)\n"
        "- budget_guidance: how to allocate the step budget (exploration vs editing)\n\n"
        "Do NOT mention Python class names, method names, config fields, or code "
        "implementation details. Write as practical advice for an agent solving bugs.\n\n"
        "Output ONLY the JSON object.\n\n"
        f"## harness.py\n```python\n{harness_code}\n```\n\n"
        f"## SUMMARY.md\n{summary}\n\n"
        f"## CONCEPT.md (learnings from harness evolution)\n{concept}\n"
    )

    litellm.drop_params = True
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()
    config = parse_config(raw)
    if config is None:
        raise ValueError(f"Failed to parse generated config: {raw[:200]}")

    out_path = harness_dir / "default_config.json"
    out_path.write_text(config.to_json())
    return config


def parse_config(text: str) -> Optional[HarnessConfig]:
    """Parse advisor output as HarnessConfig. Handles JSON in markdown code blocks."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    try:
        return HarnessConfig.from_json(text)
    except (json.JSONDecodeError, TypeError):
        return None
