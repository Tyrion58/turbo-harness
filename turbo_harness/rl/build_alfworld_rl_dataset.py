"""Build the ALFWorld RL dataset for the harness PATCH advisor.

Mirrors the ScienceWorld / SWE harness-advisor datasets: each row's prompt = (system = harness-
adaptation-engineer role; user = FULL general-harness source + playbook + the task) -> the policy
outputs SEARCH/REPLACE edits to `harness.py` (or NO_PATCH_NEEDED). env_class=alfworld_harness_patch.
reward_spec.ground_truth_json = {index, game_relpath, instance_id, task_desc, split}; `index` indexes
the split's game list (runner order) so the reward env runs exactly that game.

ALFWorld's split is a NAME (harness_r1_train / harness_r1_eval), not a file path — the reward env /
executor resolve ALF_DATA/<split>.json. task_desc per game = the env.reset() observation (the room
description + "Your task is to: ..."), dumped in the alfworld worker venv (env import needs the
ALFWORLD_DATA dir + TextWorld).

Usage:
  PYTHONPATH=$PWD .venv/bin/python -m turbo_harness.rl.build_alfworld_rl_dataset \
    --split harness_r1_train \
    --harness-dir artifacts/alfworld/qwen3-5-9b/alf_stage1_sonnet/general \
    --playbook experiments/logs/alfworld_meta_harness/alf_stage1_sonnet/playbook.json \
    --output data/rl/alfworld_train.parquet [--subset 5,6,7,11,12,14,19]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from turbo_harness.playbook.memory_advisor import format_playbook_context  # noqa: E402
from turbo_harness.playbook.schemas import Playbook  # noqa: E402
from turbo_harness.alfworld.runner import _games  # noqa: E402  (pure-json split loader, no alfworld import)

WORKER_PY = os.environ.get("ALFWORLD_WORKER_PY", sys.executable)
ENV_CLASS = "alfworld_harness_patch"
DEFAULT_HARNESS_DIR = str(REPO / "artifacts/alfworld/general")
MAX_PROMPT_TOKENS = 16384  # SkyRL silently DROPS rows whose prompt exceeds trainer.max_prompt_length
# The FULL general harness.py + playbook + task make up the user prompt; keep max_prompt_length (in
# scripts/train_alfworld_rl.sh) and the advisor serve --max-model-len above the max reported below.

SYSTEM_PROMPT = (
    "You are a harness adaptation engineer for an autonomous agent that plays the ALFWorld embodied "
    "household environment (TextWorld).\n\n"
    "You receive:\n"
    "1. The FULL source code of a base harness — a Python scaffold `run_episode(env, llm, max_steps)` "
    "that orchestrates the agent: it manages context/memory across steps, decides what to show each "
    "turn, canonicalizes the model's chosen action to an admissible command, repairs malformed "
    "tool calls, tracks subgoals, and controls the loop / stop logic.\n"
    "2. A specific ALFWorld task the agent must solve (one of: put an object somewhere; clean then "
    "place; heat then place; cool then place; examine an object under a desklamp; put two objects "
    "somewhere).\n\n"
    "Your job: adapt the harness's SCAFFOLD BEHAVIOR so the agent is more likely to succeed on THIS "
    "task. Focus on high-impact, targeted changes:\n"
    "1. Subgoal decomposition for THIS task type: the ordered sub-steps (e.g. for heat-then-place: "
    "find object -> pick it up -> go to microwave -> heat -> go to target receptacle -> put), and "
    "which receptacles to search and in what order.\n"
    "2. Control flow: step budget, recovery when an action does nothing / is inadmissible, memory of "
    "already-searched receptacles, and stop logic.\n\n"
    "Keep changes targeted to THIS task; do NOT rewrite the whole harness.\n\n"
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
    "- The patched harness.py MUST still expose `run_episode(env, llm, max_steps)` returning a dict "
    "with a 'won' key.\n"
    "- Do NOT break the ALFWorld API: env.reset() -> (obs, admissible); env.step(action) -> "
    "(obs, admissible, done, won); and the api.* helpers (SYSTEM_PROMPT, TAKE_ACTION_TOOL, "
    "extract_turn, canonicalize, available_actions_str).\n"
    "- Output ONLY the SEARCH/REPLACE blocks — no explanation."
)

USER_TEMPLATE = (
    "## Base harness.py (FULL SOURCE CODE — read carefully)\n\n"
    "```python\n{harness_code}\n```\n\n"
    "## Learned playbook (task-type strategies + anti-patterns)\n{playbook}\n\n"
    "## ALFWorld Task\n\n{task_desc}\n\n"
    "---\n\n"
    "Produce SEARCH/REPLACE edits that adapt harness.py for THIS specific task. Focus on the subgoal "
    "decomposition and receptacle-search / recovery logic that help the agent succeed on THIS task. "
    "Output ONLY the SEARCH/REPLACE blocks (or NO_PATCH_NEEDED)."
)


def _load_harness_code(harness_dir: str) -> str:
    hp = Path(harness_dir) / "harness.py"
    if not hp.exists():
        raise FileNotFoundError(f"No harness.py in {harness_dir}")
    return hp.read_text().strip()


def _dump_task_descs(split: str) -> dict[int, str]:
    """Reset each game in the split via the alfworld worker venv and return {index: reset_obs}.

    The reset observation is the task description (room state + 'Your task is to: ...') the agent
    starts from. Runs in the worker venv because AlfEnv import needs ALFWORLD_DATA + TextWorld.
    """
    code = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        "from turbo_harness.alfworld.runner import _games\n"
        "from turbo_harness.alfworld.env import AlfEnv\n"
        f"games=_games({split!r})\n"
        "out={}\n"
        "for i,g in enumerate(games):\n"
        "    e=AlfEnv(g)\n"
        "    try:\n"
        "        obs,_=e.reset(); out[i]=obs\n"
        "    finally:\n"
        "        e.close()\n"
        "print('DESCS_JSON:'+json.dumps(out))\n"
    )
    r = subprocess.run([WORKER_PY, "-c", code], capture_output=True, text=True, timeout=3600)
    for line in r.stdout.splitlines():
        if line.startswith("DESCS_JSON:"):
            return {int(k): v for k, v in json.loads(line[len("DESCS_JSON:"):]).items()}
    raise RuntimeError(f"task-desc dump failed: {r.stderr[-500:]}")


def main():
    ap = argparse.ArgumentParser(description="Build ALFWorld RL dataset (harness patch advisor)")
    ap.add_argument("--split", default="harness_r1_train", help="split NAME (ALF_DATA/<split>.json)")
    ap.add_argument("--harness-dir", default=DEFAULT_HARNESS_DIR,
                    help="Base general harness whose harness.py the policy patches.")
    ap.add_argument("--playbook",
                    default="experiments/logs/alfworld_meta_harness/alf_stage1_sonnet/playbook.json")
    ap.add_argument("--output", default="data/rl/alfworld_train.parquet")
    ap.add_argument("--subset", default=None, help="comma list of indices (MVP), else all")
    args = ap.parse_args()

    games = _games(args.split)
    harness_code = _load_harness_code(args.harness_dir)
    if args.playbook and not Path(args.playbook).exists():
        raise FileNotFoundError(
            f"--playbook {args.playbook} not found (a silent empty playbook would gut the "
            f"method — every training row would carry no strategies). Pass --playbook '' to "
            f"intentionally build a no-playbook ablation."
        )
    playbook_context = (
        format_playbook_context(Playbook.load(args.playbook)) if args.playbook else ""
    )
    descs = _dump_task_descs(args.split)

    # --subset accepts comma lists and inclusive ranges: "5,6,7" or "0-79" or "80-99,105".
    if args.subset:
        idxs = []
        for part in args.subset.split(","):
            if "-" in part:
                a, b = part.split("-", 1)
                idxs += list(range(int(a), int(b) + 1))
            else:
                idxs.append(int(part))
    else:
        idxs = list(range(len(games)))
    records = []
    for i in idxs:
        game_relpath = games[i]
        task_desc = descs.get(i, "")
        user = USER_TEMPLATE.format(harness_code=harness_code,
                                    playbook=playbook_context or "(none)",
                                    task_desc=task_desc)
        gt = {"index": i, "game_relpath": game_relpath,
              "instance_id": f"{args.split}_{i}", "task_desc": task_desc, "split": args.split}
        records.append({
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                       {"role": "user", "content": user}],
            "env_class": ENV_CLASS,
            "reward_spec": {"ground_truth_json": json.dumps(gt)},
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(out)
    plens = [sum(len(m["content"]) for m in r["prompt"]) for r in records]
    print(f"Built {len(records)} ALFWorld RL rows -> {out} (env_class={ENV_CLASS})")
    print(f"Prompt chars: min={min(plens)} max={max(plens)} mean={sum(plens) // len(plens)}")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", trust_remote_code=True)
        tl = [sum(len(tok(m["content"]).input_ids) for m in r["prompt"]) for r in records]
        over = sum(1 for n in tl if n > MAX_PROMPT_TOKENS)
        print(f"Prompt tokens (Qwen3.5): min={min(tl)} max={max(tl)} mean={sum(tl) // len(tl)}")
        if over:
            print(f"WARNING: {over}/{len(tl)} prompts exceed max_prompt_length={MAX_PROMPT_TOKENS} "
                  f"tokens and WILL BE FILTERED by SkyRL.")
    except Exception as e:  # noqa: BLE001
        print(f"(token report skipped: {type(e).__name__})")


if __name__ == "__main__":
    main()
