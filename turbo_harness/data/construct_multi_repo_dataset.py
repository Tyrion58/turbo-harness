"""Construct a multi-repo dataset from SWE-smith for cross-repo evaluation.

Samples 2-4 instances per repo from diverse, popular repos to test whether
per-instance harness adaptation adds value when issues span multiple repos.

Usage:
    HF_HOME=~/.cache/huggingface \
    python -m turbo_harness.data.construct_multi_repo_dataset \
        --output_dir data/swe_smith \
        --train_per_repo 2 --test_per_repo 2 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

from turbo_harness.infra.scoring import build_advisor_prompt

SWE_SMITH_DATASET = "SWE-bench/SWE-smith"

SELECTED_REPOS = [
    # Python — Web/API
    "swesmith/encode__starlette.db5063c2",
    "swesmith/tornadoweb__tornado.d5ac65c1",
    "swesmith/bottlepy__bottle.a8dfef30",
    "swesmith/benoitc__gunicorn.bacbf8aa",
    "swesmith/pallets__jinja.ada0a9a6",
    # Python — Data/ML
    "swesmith/pandas-dev__pandas.95280573",
    "swesmith/dask__dask.5f61e423",
    "swesmith/stanfordnlp__dspy.651a4c71",
    # Python — CLI/Config/Validation
    "swesmith/pallets__click.fde47b4b",
    "swesmith/python-jsonschema__jsonschema.93e0caa5",
    "swesmith/pydantic__pydantic.acb0f10f",
    # Python — Serialization
    "swesmith/marshmallow-code__marshmallow.9716fc62",
    # Python — Auth/Network
    "swesmith/oauthlib__oauthlib.1fd52536",
    "swesmith/paramiko__paramiko.23f92003",
    # Python — Text/Parsing
    "swesmith/pygments__pygments.27649ebb",
    "swesmith/sqlfluff__sqlfluff.50a1c4b6",
    "swesmith/chardet__chardet.9630f238",
    # Python — Testing/DevTools
    "swesmith/joke2k__faker.8b401a7d",
    "swesmith/scrapy__scrapy.35212ec5",
    "swesmith/cookiecutter__cookiecutter.b4451231",
    # Python — Misc
    "swesmith/arrow-py__arrow.1d70d009",
    # Python — Config/Infra
    "swesmith/facebookresearch__hydra.0f03eb60",
    "swesmith/getmoto__moto.694ce1f4",
    # Python — Formatting/Display
    "swesmith/prettytable__prettytable.ca90b055",
    "swesmith/adrienverge__yamllint.8513d9b9",
]


def build_dataset_row(instance: dict) -> dict:
    info_dict = {
        "problem_statement": instance.get("problem_statement", ""),
        "repo": instance.get("repo", ""),
    }
    advisor_prompt_text = build_advisor_prompt(info_dict)
    prompt = [{"role": "user", "content": advisor_prompt_text}]

    instance_dict = {
        "instance_id": str(instance.get("instance_id", "")),
        "patch": str(instance.get("patch", "")),
        "repo": str(instance.get("repo", "")),
        "problem_statement": str(instance.get("problem_statement", "")),
        "image_name": str(instance.get("image_name", "")),
        "FAIL_TO_PASS": list(instance.get("FAIL_TO_PASS", [])),
        "PASS_TO_PASS": list(instance.get("PASS_TO_PASS", [])),
    }
    instance_json = json.dumps(instance_dict)

    return {
        "prompt": prompt,
        "env_class": "swe_smith",
        "reward_spec": {"ground_truth_json": instance_json},
        "original_question": instance.get("problem_statement", ""),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Construct multi-repo SWE-smith dataset"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--train_per_repo", type=int, default=2, help="Training samples per repo"
    )
    parser.add_argument(
        "--test_per_repo", type=int, default=2, help="Test samples per repo"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    needed_per_repo = args.train_per_repo + args.test_per_repo

    print("Loading SWE-smith dataset...")
    ds = load_dataset(SWE_SMITH_DATASET, split="train")
    print(f"Loaded {len(ds)} total instances")

    repo_set = set(SELECTED_REPOS)
    per_repo: dict[str, list] = {r: [] for r in SELECTED_REPOS}

    for item in ds:
        repo = item["repo"]
        if repo in repo_set and item.get("problem_statement", ""):
            per_repo[repo].append(dict(item))

    train_instances = []
    test_instances = []
    skipped = []

    for repo in SELECTED_REPOS:
        instances = per_repo[repo]
        if len(instances) < needed_per_repo:
            skipped.append((repo, len(instances)))
            continue

        random.shuffle(instances)
        sampled = instances[:needed_per_repo]
        train_instances.extend(sampled[: args.train_per_repo])
        test_instances.extend(sampled[args.train_per_repo :])

        short_name = repo.split("/")[1].split(".")[0]
        print(
            f"  {short_name}: {len(instances)} available, "
            f"sampled {args.train_per_repo} train + {args.test_per_repo} test"
        )

    if skipped:
        print(f"\nSkipped {len(skipped)} repos (too few instances):")
        for repo, count in skipped:
            print(f"  {repo}: only {count} instances (need {needed_per_repo})")

    random.shuffle(train_instances)
    random.shuffle(test_instances)

    print(f"\nTotal: {len(train_instances)} train, {len(test_instances)} test")
    print(f"Repos: {len(SELECTED_REPOS) - len(skipped)}")

    for split_name, instances in [("train", train_instances), ("test", test_instances)]:
        if not instances:
            continue
        rows = [build_dataset_row(inst) for inst in instances]
        filename = f"{split_name}_multi_repo.json"
        with open(output_dir / filename, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"Saved {len(rows)} {split_name} examples to {filename}")

    # Save repo manifest for reference
    manifest = []
    for repo in SELECTED_REPOS:
        if repo in {r for r, _ in skipped}:
            continue
        short_name = repo.split("/")[1].split(".")[0]
        lang = "python"
        manifest.append({
            "repo": repo,
            "short_name": short_name,
            "language": lang,
            "available": len(per_repo[repo]),
            "train": args.train_per_repo,
            "test": args.test_per_repo,
        })
    with open(output_dir / "multi_repo_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("Saved repo manifest to multi_repo_manifest.json")


if __name__ == "__main__":
    main()
