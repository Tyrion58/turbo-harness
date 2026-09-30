"""Build RL training dataset for the harness patch advisor.

Converts train_multi_repo.json to SkyRL parquet format with the
patch advisor prompt (system + user with harness code + playbook + issue).

Usage:
    python -m turbo_harness.rl.build_rl_dataset \
        --data-file data/swe_smith/train_multi_repo.json \
        --harness-dir artifacts/swe_smith/haiku/general \
        --playbook artifacts/swe_smith/haiku/playbook.json \
        --output data/rl/train.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from turbo_harness.playbook.memory_advisor import format_playbook_context  # noqa: E402
from turbo_harness.playbook.schemas import Playbook  # noqa: E402
from turbo_harness.patch_advisor import (  # noqa: E402
    PATCH_ADVISOR_SYSTEM_PROMPT,
    PATCH_ADVISOR_USER_TEMPLATE,
    _load_harness_context,
)


def build_dataset(
    data_file: str,
    harness_dir: str,
    playbook_path: str,
    output_path: str,
):
    playbook = Playbook.load(playbook_path)
    playbook_context = format_playbook_context(playbook)

    harness_code, summary, concept, rules_memory = _load_harness_context(harness_dir)

    rows = [json.loads(line) for line in open(data_file) if line.strip()]
    records = []

    for row in rows:
        gt = json.loads(row["reward_spec"]["ground_truth_json"])
        problem = gt.get("problem_statement", row.get("original_question", ""))

        user_content = PATCH_ADVISOR_USER_TEMPLATE.format(
            harness_code=harness_code,
            summary=summary or "(no summary available)",
            concept=concept or "(no concept document available)",
            rules_memory=rules_memory or "(no repository knowledge available)",
            playbook=playbook_context + "\n" if playbook_context else "",
            problem_statement=problem,
        )

        prompt = [
            {"role": "system", "content": PATCH_ADVISOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        records.append({
            "prompt": prompt,
            "env_class": "harness_advisor",
            "reward_spec": {"ground_truth_json": json.dumps(gt)},
        })

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    df.to_parquet(out_path)
    print(f"Built {len(records)} RL training examples -> {out_path}")

    prompt_lens = [
        sum(len(m["content"]) for m in r["prompt"]) for r in records
    ]
    print(f"Prompt length: min={min(prompt_lens)}, max={max(prompt_lens)}, "
          f"mean={sum(prompt_lens)//len(prompt_lens)} chars")

    # Token-accurate report: SkyRL silently DROPS prompts exceeding trainer.max_prompt_length
    # tokens (dataset.py filter). Report tokens (not just chars) and flag rows that will be
    # filtered so the effective dataset size is never a surprise.
    MAX_PROMPT_TOKENS = 16384
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", trust_remote_code=True)
        tok_lens, over = [], []
        for r in records:
            n = sum(len(tok(m["content"]).input_ids) for m in r["prompt"])
            tok_lens.append(n)
            if n > MAX_PROMPT_TOKENS:
                gt = json.loads(r["reward_spec"]["ground_truth_json"])
                over.append((gt.get("instance_id", "?"), n))
        print(f"Prompt tokens (Qwen3.5): min={min(tok_lens)}, max={max(tok_lens)}, "
              f"mean={sum(tok_lens)//len(tok_lens)}")
        if over:
            print(f"WARNING: {len(over)}/{len(records)} prompt(s) exceed "
                  f"max_prompt_length={MAX_PROMPT_TOKENS} tokens and WILL BE FILTERED by SkyRL:")
            for iid, n in over:
                print(f"  - {iid}: {n} tokens")
        else:
            print(f"All {len(records)} prompts fit within {MAX_PROMPT_TOKENS} tokens.")
    except Exception as e:  # tokenizer unavailable — fall back to the char report above
        print(f"(token-accurate report skipped: {type(e).__name__}: {e})")


def main():
    parser = argparse.ArgumentParser(description="Build RL dataset for harness advisor")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--harness-dir", required=True)
    parser.add_argument("--playbook", required=True)
    parser.add_argument("--output", default="data/rl/train.parquet")
    args = parser.parse_args()

    build_dataset(args.data_file, args.harness_dir, args.playbook, args.output)


if __name__ == "__main__":
    main()
