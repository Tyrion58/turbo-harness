"""Dataset construction for SWE-Smith domain.

Generates training, validation, and test datasets for software engineering patch generation with advisor feedback.
Uses SWE-Smith harness for evaluation. Saves datasets as JSON files (JSONL format).

Example usage:
    # Generate dataset with train/val/test split
    uv run python -m advisor_models.swe_smith.construct_dataset \
        --output_dir data/swe_smith \
        --repo iterative__dvc \
        --train_size 200 \
        --val_size 118 \
        --test_size 120
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from datasets import load_dataset

from turbo_harness.infra.scoring import build_advisor_prompt

# SWE-Smith dataset on HuggingFace
SWE_SMITH_DATASET = "SWE-bench/SWE-smith"


def load_swe_smith_data(
    split: str = "train", repo_filter: str = None
) -> List[Dict[str, Any]]:
    """Load SWE-Smith data from HuggingFace.

    Args:
        split: Dataset split to load ('train' or 'test')
        repo_filter: Optional repo name to filter by (e.g., 'jd__tenacity')

    Returns:
        List of problem instances
    """
    print(f"Loading SWE-Smith {split} data...")

    try:
        dataset = load_dataset(SWE_SMITH_DATASET, split=split)
        instances = []

        for item in dataset:
            # Filter by repo if specified
            if repo_filter and repo_filter not in item.get("repo", ""):
                continue

            if item.get("problem_statement", "") == "":
                continue

            instances.append(dict(item))

        print(
            f"Loaded {len(instances)} instances"
            + (f" for repo {repo_filter}" if repo_filter else "")
        )
        return instances

    except Exception as e:
        print(f"Error loading SWE-Smith data: {e}")
        return []


def build_dataset_row(instance: Dict[str, Any]) -> Dict[str, Any]:
    """Build a dataset row for a single SWE-Smith instance.

    Args:
        instance: SWE-Smith problem instance

    Returns:
        Dataset row dict
    """
    # Build advisor prompt
    info_dict = {
        "problem_statement": instance.get("problem_statement", ""),
        "repo": instance.get("repo", ""),
    }
    advisor_prompt_text = build_advisor_prompt(info_dict)

    # Create the prompt structure for training
    prompt = [{"role": "user", "content": advisor_prompt_text}]

    # Serialize the entire instance as JSON to avoid schema issues with list fields
    # The instance will be deserialized in the environment
    # Convert to dict and explicitly handle any Arrow types from HuggingFace datasets
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


def process_instances(instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process multiple instances into dataset rows.

    Args:
        instances: List of SWE-Smith instances

    Returns:
        List of dataset rows
    """
    print(f"Processing {len(instances)} instances...")

    rows = []
    for instance in tqdm(instances, desc="Building dataset"):
        try:
            row = build_dataset_row(instance)
            rows.append(row)
        except Exception as e:
            print(
                f"Error processing instance {instance.get('instance_id', 'unknown')}: {e}"
            )

    return rows


def main():
    parser = argparse.ArgumentParser(description="Construct SWE-Smith domain dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for dataset files",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Filter by specific repo (e.g., 'jd__tenacity')",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=100,
        help="Training set size (default: 100)",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=49,
        help="Validation set size (default: 49)",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=0,
        help="Test set size (default: 0, no test split)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix to add to output filenames",
    )

    args = parser.parse_args()
    random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    train_instances = load_swe_smith_data(split="train", repo_filter=args.repo)

    if len(train_instances) == 0:
        print("No instances loaded. Exiting.")
        return

    # Shuffle and split
    random.shuffle(train_instances)

    total_needed = args.train_size + args.val_size + args.test_size

    if total_needed > len(train_instances):
        print(
            f"Warning: Requested {total_needed} instances but only {len(train_instances)} available."
        )
        ratio = len(train_instances) / total_needed
        args.train_size = int(args.train_size * ratio)
        args.val_size = int(args.val_size * ratio)
        args.test_size = len(train_instances) - args.train_size - args.val_size

    i = args.train_size
    j = i + args.val_size
    train_instances_final = train_instances[:i]
    val_instances = train_instances[i:j]
    test_instances = train_instances[j : j + args.test_size] if args.test_size > 0 else []

    # Process instances
    train_rows = process_instances(train_instances_final)
    val_rows = process_instances(val_instances)
    test_rows = process_instances(test_instances) if test_instances else []

    # Save datasets as JSON (JSONL format - one JSON object per line)
    repo_suffix = f"_{args.repo}" if args.repo else ""
    suffix_str = f"_{args.suffix}" if args.suffix else ""

    for split_name, rows in [("train", train_rows), ("validation", val_rows), ("test", test_rows)]:
        if not rows:
            continue
        filename = f"{split_name}{repo_suffix}{suffix_str}.json"
        with open(output_dir / filename, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"Saved {len(rows)} {split_name} examples to {filename}")


if __name__ == "__main__":
    main()
