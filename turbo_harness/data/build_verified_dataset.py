"""Build SWE-bench Verified train/test splits in the SWE-smith row format for the Turbo ladder.

Repo-stratified, seed 42. Writes data/swe_smith/{train,val,test}_verified.json where each row is
`{"reward_spec": {"ground_truth_json": "<gt json>"}}` (the exact format load_issues/executor expect).
The gt json carries every field swebench's make_test_spec + grading need, plus a `benchmark: "verified"`
flag that routes executor.run_instance + infra.scoring.compute_score down their swebench branches
(the SWE-smith path is untouched).

  python -m turbo_harness.data.build_verified_dataset --test-total 50
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "data" / "swe_smith"


def _gt_of(r: dict, image_name: str) -> dict:
    # FAIL_TO_PASS / PASS_TO_PASS kept in the NATIVE swebench format (JSON strings) so make_test_spec
    # + get_eval_report consume them directly; version + environment_setup_commit are required by
    # make_test_spec to pick the right test-spec.
    return {
        "instance_id": r["instance_id"],
        "repo": r["repo"],
        "base_commit": r.get("base_commit", ""),
        "patch": r["patch"],                       # gold (for gold-validation; not used to score preds)
        "test_patch": r.get("test_patch", ""),
        "problem_statement": r.get("problem_statement", ""),
        "FAIL_TO_PASS": r["FAIL_TO_PASS"],
        "PASS_TO_PASS": r["PASS_TO_PASS"],
        "version": r.get("version", ""),
        "environment_setup_commit": r.get("environment_setup_commit", r.get("base_commit", "")),
        "image_name": image_name,
        "benchmark": "verified",
    }


def build(test_total: int = 150, val_total: int = 100, seed: int = 42) -> None:
    """Repo-stratified 3-way split: train / val / test.

    val is used for RL/harness CHECKPOINT SELECTION (never test), so the reported test
    number has no selection leakage. The SAME train split feeds BOTH meta-harness evolution
    AND RL (no per-stage subsetting) so Meta vs Turbo share identical training data — this
    fixes the earlier confound where evolution used 50 and RL a different 64 (2x data).
    """
    from datasets import load_dataset
    from swebench.harness.test_spec.test_spec import make_test_spec

    ds = [dict(r) for r in load_dataset("princeton-nlp/SWE-bench_Verified", split="test")]
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for r in ds:
        by_repo[r["repo"]].append(r)
    n = len(ds)

    rng = random.Random(seed)
    test, val, train = [], [], []
    for repo, group in sorted(by_repo.items()):
        g = sorted(group, key=lambda x: x["instance_id"])
        rng.shuffle(g)
        L = len(group)
        k_test = round(test_total * L / n)
        k_val = round(val_total * L / n)
        if k_test + k_val > L:              # tiny repo: don't starve train below 0
            k_val = max(0, L - k_test)
        test.extend(g[:k_test])
        val.extend(g[k_test:k_test + k_val])
        train.extend(g[k_test + k_val:])

    # Global shuffle within each split (order only; membership unchanged) so any prefix slice spans repos.
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    # disjointness assertion (correctness guard)
    ids = lambda rows: {r["instance_id"] for r in rows}
    assert not (ids(train) & ids(val)) and not (ids(train) & ids(test)) and not (ids(val) & ids(test)), \
        "splits overlap!"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def _write(rows: list[dict], path: Path) -> None:
        with open(path, "w") as f:
            for r in rows:
                gt = _gt_of(r, make_test_spec(r, namespace="swebench").instance_image_key)
                f.write(json.dumps({"reward_spec": {"ground_truth_json": json.dumps(gt)}}) + "\n")

    _write(train, OUT_DIR / "train_verified.json")
    _write(val, OUT_DIR / "val_verified.json")
    _write(test, OUT_DIR / "test_verified.json")
    print(f"train {len(train)} / val {len(val)} / test {len(test)} (of {n}), seed={seed}")
    for nm, rows in [("train", train), ("val", val), ("test", test)]:
        print(f"  {nm} by repo:", dict(sorted(Counter(r['repo'] for r in rows).items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-total", type=int, default=150, help="approx repo-stratified test size")
    ap.add_argument("--val-total", type=int, default=100, help="approx repo-stratified val size")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build(test_total=args.test_total, val_total=args.val_total, seed=args.seed)


if __name__ == "__main__":
    main()
