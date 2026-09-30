"""Build the Terminal-Bench-2 RL dataset for the harness PATCH advisor.

Mirrors the ScienceWorld dataset (`build_scienceworld_rl_dataset.py`): each row's prompt =
(system = harness-adaptation-engineer role; user = FULL winner-harness source + the TB2 task
instruction) -> the policy outputs SEARCH/REPLACE edits to the winner agent file (or NO_PATCH_NEEDED).
env_class=tb2_harness_patch. reward_spec.ground_truth_json = {index, task, instance_id}; the reward
env runs exactly that task via harbor with the (patched) harness + Sonnet student.

Usage:
  PYTHONPATH=$PWD .venv/bin/python -m turbo_harness.rl.build_tb2_rl_dataset \
    --split turbo_harness/terminal_bench/data/train_tasks.json \
    --output data/rl/tb2_train.parquet [--subset 0-9]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from turbo_harness.playbook.memory_advisor import format_playbook_context  # noqa: E402
from turbo_harness.playbook.schemas import Playbook  # noqa: E402

ENV_CLASS = "tb2_harness_patch"
WINNER_AGENT = REPO / "turbo_harness/terminal_bench/agents/kira_auto_test.py"
HARBOR_CACHE = Path(os.path.expanduser("~/.cache/harbor"))
MAX_PROMPT_TOKENS = 32768  # keep in sync with train_tb2_rl.sh trainer.max_prompt_length

SYSTEM_PROMPT = (
    "You are a harness adaptation engineer for an autonomous agent that solves terminal-based tasks "
    "(terminal-bench-2). You receive:\n"
    "1. The FULL source code of a base harness — a Python agent class `AgentHarness` (a Terminus-2 / "
    "KIRA variant) that drives the agent loop: it calls the model with tools, executes shell commands "
    "in a tmux terminal, polls for long-running commands, detects stalls/loops, and decides when to "
    "mark the task complete.\n"
    "2. A specific terminal task the agent must solve.\n\n"
    "Your job: adapt the harness's SCAFFOLD BEHAVIOR so the agent is more likely to succeed on THIS "
    "task. Focus on high-impact, targeted changes to the loop/control logic and the guidance the "
    "scaffold gives the model (e.g. poll-wait timing for builds/servers, stall/loop handling, "
    "completion-gate strictness, output-truncation limits, task-type hints in the prompt). Do NOT "
    "hard-code the task's solution or specific commands; tune the scaffold that guides the agent.\n\n"
    "Keep changes targeted to THIS task; do NOT rewrite the whole harness.\n\n"
    "You are given a LEARNED PLAYBOOK of scaffold strategies and ANTI-PATTERNS distilled from prior "
    "harness experiments on this benchmark. Apply its strategies when they fit THIS task's kind, and "
    "HEED its anti-patterns. IMPORTANT: the base harness already solves MOST tasks — if it already "
    "handles this task's kind and no specific failure mode applies, output NO_PATCH_NEEDED rather "
    "than making speculative edits. Over-patching an already-working harness (especially rewriting "
    "the completion gate, stall logic, or core loop) is the most common way to REGRESS.\n\n"
    "## Output format\n"
    "Output one or more SEARCH/REPLACE blocks:\n\n"
    "<<<SEARCH\n"
    "exact text from the original file\n"
    "===\n"
    "replacement text\n"
    ">>>REPLACE\n\n"
    "Each SEARCH section must match EXACTLY one location in the original file — include enough "
    "surrounding context to be unique. Preserve indentation and whitespace exactly.\n\n"
    "If no adaptation is needed, output EXACTLY: NO_PATCH_NEEDED\n\n"
    "## Constraints\n"
    "- The patched file MUST still define `class AgentHarness(Terminus2)` and stay importable.\n"
    "- Do NOT break the harbor API (tool calls, tmux session, Command objects).\n"
    "- Output ONLY the SEARCH/REPLACE blocks — no explanation."
)

USER_TEMPLATE = (
    "## Base harness (FULL SOURCE CODE — read carefully)\n\n"
    "```python\n{harness_code}\n```\n\n"
    "## Learned playbook (scaffold strategies + anti-patterns)\n{playbook}\n\n"
    "## Terminal task\n\n"
    "Task id: {task}\n\n"
    "{task_desc}\n\n"
    "---\n\n"
    "Produce SEARCH/REPLACE edits that adapt the harness scaffold for THIS specific task. Focus on "
    "loop/control logic and scaffold guidance that help the agent succeed on THIS task. Output ONLY "
    "the SEARCH/REPLACE blocks (or NO_PATCH_NEEDED)."
)


def _task_instructions() -> dict[str, str]:
    """Map task_name -> instruction text, read from cached harbor `<task>/instruction.md` files."""
    out: dict[str, str] = {}
    for f in glob.glob(
        str(HARBOR_CACHE / "tasks" / "**" / "instruction.md"), recursive=True
    ):
        name = os.path.basename(os.path.dirname(f))
        if name in out:
            continue
        try:
            txt = open(f, errors="ignore").read().strip()
        except OSError:
            continue
        if txt:
            out[name] = txt
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Build TB2 RL dataset (harness patch advisor)"
    )
    ap.add_argument(
        "--split",
        default=str(REPO / "turbo_harness/terminal_bench/data/train_tasks.json"),
    )
    ap.add_argument(
        "--winner",
        default=str(WINNER_AGENT),
        help="Base winner agent file the policy patches.",
    )
    ap.add_argument("--output", default=str(REPO / "data/rl/tb2_train.parquet"))
    ap.add_argument(
        "--playbook",
        default=str(REPO / "artifacts/tb2/playbook.json"),
        help="playbook.json (Stage 2); pass '' for a no-playbook ablation.",
    )
    ap.add_argument(
        "--subset", default=None, help='comma list / ranges, e.g. "0-9" or "0,3,5"'
    )
    args = ap.parse_args()

    tasks = json.loads(Path(args.split).read_text())
    harness_code = Path(args.winner).read_text().strip()
    descs = _task_instructions()
    if args.playbook and not Path(args.playbook).exists():
        raise FileNotFoundError(
            f"--playbook {args.playbook} not found (a silent empty playbook guts the method). "
            f"Pass --playbook '' to intentionally run a no-playbook ablation."
        )
    playbook_ctx = (
        format_playbook_context(Playbook.load(args.playbook))
        if args.playbook
        else "(none)"
    )
    missing = [t for t in tasks if t not in descs]
    if missing:
        print(
            f"WARNING: no cached instruction for {len(missing)} tasks (using task id only): "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    if args.subset:
        idxs = []
        for part in args.subset.split(","):
            if "-" in part:
                a, b = part.split("-", 1)
                idxs += list(range(int(a), int(b) + 1))
            else:
                idxs.append(int(part))
    else:
        idxs = list(range(len(tasks)))

    records = []
    for i in idxs:
        task = tasks[i]
        task_desc = descs.get(
            task, "(no cached task description — infer the task type from the id)"
        )
        user = USER_TEMPLATE.format(
            harness_code=harness_code,
            playbook=playbook_ctx,
            task=task,
            task_desc=task_desc,
        )
        gt = {"index": i, "task": task, "instance_id": task}
        records.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                "env_class": ENV_CLASS,
                "reward_spec": {"ground_truth_json": json.dumps(gt)},
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(out)
    plens = [sum(len(m["content"]) for m in r["prompt"]) for r in records]
    print(f"Built {len(records)} TB2 RL rows -> {out} (env_class={ENV_CLASS})")
    print(
        f"Prompt chars: min={min(plens)} max={max(plens)} mean={sum(plens) // len(plens)}"
    )
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", trust_remote_code=True)
        tl = [
            sum(len(tok(m["content"]).input_ids) for m in r["prompt"]) for r in records
        ]
        over = sum(1 for n in tl if n > MAX_PROMPT_TOKENS)
        print(
            f"Prompt tokens (Qwen3.5): min={min(tl)} max={max(tl)} mean={sum(tl) // len(tl)}"
        )
        if over:
            print(
                f"WARNING: {over}/{len(tl)} prompts exceed max_prompt_length={MAX_PROMPT_TOKENS} "
                f"tokens and WILL BE FILTERED by SkyRL."
            )
    except Exception as e:  # noqa: BLE001
        print(f"(token report skipped: {type(e).__name__})")


if __name__ == "__main__":
    main()
