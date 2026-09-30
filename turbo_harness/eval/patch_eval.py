"""Evaluate the patch-advisor pipeline on SWE-smith.

Single LLM call generates a unified diff per instance, applied statically.

Usage:
  EVAL_SERVER_URL=http://localhost:5152 \
  VERTEXAI_PROJECT=your-gcp-project VERTEXAI_LOCATION=us-east5 \
  python -m turbo_harness.eval.patch_eval \
      --harness-dir artifacts/multi_repo/haiku/multi_repo_5iter/general \
      --data_file data/swe_smith/test_multi_repo.json \
      --patch-model vertex_ai/claude-sonnet-4-5@20250929 \
      --student vertex_ai/claude-haiku-4-5 \
      --num_runs 3 --max_workers 10
"""

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "turbo_harness/infra/mini-swe-agent/src"))

from turbo_harness.patch_advisor import generate_patch, apply_patch  # noqa: E402
from turbo_harness.proposer import validate_artifact  # noqa: E402
from turbo_harness import executor as EX  # noqa: E402
from turbo_harness.utils.eval_utils import compute_multi_run_statistics, format_ci_string  # noqa: E402


@dataclass
class PatchEvalResult:
    instance_id: str
    resolved: bool
    status: str
    patch_text: str = ""
    adapted_dir: str = ""
    steps: int = 0
    agent_cost: float = 0.0
    patch_applied: bool = False
    error: Optional[str] = None


def load_instances(data_file):
    rows = [json.loads(line) for line in open(data_file) if line.strip()]
    return [json.loads(r["reward_spec"]["ground_truth_json"]) for r in rows]


# ── Phase 1: Batch patch generation ────────────────────────────

def batch_generate_patches(instances, general_harness_dir, patch_model,
                           patch_api_base=None, max_workers=20):
    print(f"\n  Phase 1: Generating patches for {len(instances)} issues "
          f"(model={patch_model}, concurrency={max_workers})...")
    patches = [None] * len(instances)
    t0 = time.time()

    def gen_one(idx):
        gt = instances[idx]
        edits = generate_patch(
            general_harness_dir, gt["problem_statement"],
            model_name=patch_model, api_base=patch_api_base,
        )
        return idx, edits

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(gen_one, i): i for i in range(len(instances))}
        done = 0
        for f in as_completed(futs):
            idx = futs[f]
            try:
                _, edits = f.result()
                patches[idx] = edits
            except Exception as e:
                patches[idx] = f"ERROR: {e}"
            done += 1
            if done % 20 == 0 or done == len(instances):
                print(f"    {done}/{len(instances)} patches generated...")

    elapsed = time.time() - t0
    n_empty = sum(1 for p in patches if not p or p == [])
    n_error = sum(1 for p in patches if isinstance(p, str) and p.startswith("ERROR:"))
    n_valid = len(patches) - n_empty - n_error
    print(f"  Phase 1 done: {elapsed:.1f}s ({elapsed/len(instances):.1f}s/issue), "
          f"{n_valid} patches, {n_empty} empty (no adaptation), {n_error} errors")
    return patches


# ── Phase 2: Static apply + validate ──────────────────────────

def batch_apply_and_validate(instances, patches, general_harness_dir, work_dir=None):
    print(f"\n  Phase 2: Applying and validating {len(instances)} patches...")
    if work_dir is None:
        import tempfile
        work_dir = tempfile.mkdtemp(prefix="patch_eval_")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    harness_dirs = [None] * len(instances)
    applied_flags = [False] * len(instances)
    stats = {"applied": 0, "empty": 0, "apply_fail": 0, "validate_fail": 0, "error": 0}

    for idx in range(len(instances)):
        gt = instances[idx]
        edits = patches[idx]
        iid = gt["instance_id"].replace("/", "_")
        out_dir = work_dir / f"issue_{idx}_{iid}"

        if isinstance(edits, str) and edits.startswith("ERROR:"):
            shutil.copytree(
                general_harness_dir, out_dir,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                dirs_exist_ok=True,
            )
            harness_dirs[idx] = str(out_dir)
            stats["error"] += 1
            continue

        if not edits:
            shutil.copytree(
                general_harness_dir, out_dir,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                dirs_exist_ok=True,
            )
            harness_dirs[idx] = str(out_dir)
            stats["empty"] += 1
            continue

        # Apply edits
        success, msg = apply_patch(edits, general_harness_dir, out_dir)
        if not success:
            # Apply failed — fallback to base harness
            shutil.copytree(
                general_harness_dir, out_dir,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                dirs_exist_ok=True,
            )
            harness_dirs[idx] = str(out_dir)
            stats["apply_fail"] += 1
            print(f"    [issue {idx}] apply failed: {msg[:100]}")
            continue

        # Validate
        ok, val_msg = validate_artifact(out_dir)
        if not ok:
            # Validation failed — fallback to base harness
            shutil.copytree(
                general_harness_dir, out_dir,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                dirs_exist_ok=True,
            )
            harness_dirs[idx] = str(out_dir)
            stats["validate_fail"] += 1
            print(f"    [issue {idx}] validation failed: {val_msg[:100]}")
            continue

        harness_dirs[idx] = str(out_dir)
        applied_flags[idx] = True
        stats["applied"] += 1

    elapsed = time.time() - t0
    print(f"  Phase 2 done: {elapsed:.1f}s, "
          f"{stats['applied']} applied, {stats['empty']} empty, "
          f"{stats['apply_fail']} apply-failed, {stats['validate_fail']} validate-failed, "
          f"{stats['error']} gen-errors")
    return harness_dirs, applied_flags


# ── Phase 3: Batch execution ─────────────────────────────────

def batch_execute(instances, harness_dirs, student_model, repo,
                  step_limit, cost_limit, max_workers=10, trajectory_dir=None):
    print(f"\n  Phase 3: Executing {len(instances)} issues (concurrency={max_workers})...")
    results = [None] * len(instances)
    t0 = time.time()

    def run_one(idx):
        gt = instances[idx]
        harness = harness_dirs[idx]
        if harness is None:
            return idx, {
                "instance_id": gt["instance_id"], "resolved": False,
                "status": "NO_HARNESS", "steps": 0, "agent_cost": 0,
            }
        traj_dir = None
        if trajectory_dir:
            traj_dir = f"{trajectory_dir}/{gt['instance_id'].replace('/', '_')}"
        r = EX.run_instance(gt, student_model, harness=harness, repo=repo,
                            step_limit=step_limit, cost_limit=cost_limit,
                            trajectory_dir=traj_dir)
        return idx, r

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(run_one, i): i for i in range(len(instances))}
        done_count = 0
        for f in as_completed(futs):
            idx = futs[f]
            try:
                _, r = f.result()
                results[idx] = r
            except Exception as e:
                results[idx] = {
                    "instance_id": instances[idx]["instance_id"],
                    "resolved": False,
                    "status": f"ERROR:{type(e).__name__}",
                    "steps": 0, "agent_cost": 0,
                }
            done_count += 1
            if done_count % 20 == 0 or done_count == len(instances):
                resolved = sum(1 for r in results if r and r.get("resolved"))
                print(f"    {done_count}/{len(instances)} done, {resolved} resolved...")

    elapsed = time.time() - t0
    resolved = sum(1 for r in results if r and r.get("resolved"))
    total = len(instances)
    errors = sum(1 for r in results if r and "ERROR" in str(r.get("status", "")))
    costs = [r.get("agent_cost", 0) for r in results if r]
    steps_all = [r.get("steps", 0) for r in results if r]
    resolved_results = [r for r in results if r and r.get("resolved")]
    steps_resolved = [r.get("steps", 0) for r in resolved_results]

    avg_steps_all = sum(steps_all) / len(steps_all) if steps_all else 0
    avg_steps_resolved = sum(steps_resolved) / len(steps_resolved) if steps_resolved else 0

    print(f"  Phase 3 done: {elapsed:.1f}s, {resolved}/{total} resolved")
    print(f"  Run: {resolved}/{total} = {resolved/total*100:.2f}%  "
          f"errors={errors}  agent_cost=${sum(costs):.2f}  "
          f"avg_steps={avg_steps_all:.1f} (resolved: {avg_steps_resolved:.1f})")

    return results


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate patch-advisor pipeline on SWE-smith"
    )
    parser.add_argument("--harness-dir", required=True,
                        help="Path to general harness artifact dir")
    parser.add_argument("--data_file", required=True,
                        help="JSONL data file")
    parser.add_argument("--patch-model",
                        default="vertex_ai/claude-sonnet-4-5@20250929")
    parser.add_argument("--patch-api-base", default=None)
    parser.add_argument("--student", default="vertex_ai/claude-haiku-4-5")
    parser.add_argument("--repo", default="multi_repo")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--cost-limit", type=float, default=3.0)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--baseline-rate", type=float, default=None)
    parser.add_argument("--general-rate", type=float, default=None,
                        help="General harness rate for comparison")
    parser.add_argument("--out", default=None)
    parser.add_argument("--cache-harnesses", default=None,
                        help="Directory to save patched harnesses for reuse")
    parser.add_argument("--exec-only", action="store_true",
                        help="Skip Phase 1+2, load harnesses from --cache-harnesses")
    args = parser.parse_args()

    import litellm
    litellm.drop_params = True

    instances = load_instances(args.data_file)
    if args.num_samples and len(instances) > args.num_samples:
        instances = instances[:args.num_samples]

    print(f"Loaded {len(instances)} instances from {args.data_file}")
    print(f"General harness: {args.harness_dir}")
    print(f"Patch model: {args.patch_model}")
    print(f"Student: {args.student}")
    print(f"Runs: {args.num_runs}, Workers: {args.max_workers}")

    if args.exec_only:
        assert args.cache_harnesses, "--exec-only requires --cache-harnesses"
        print(f"\n{'=' * 60}")
        print(f"Exec-only mode: loading harnesses from {args.cache_harnesses}")
        print(f"{'=' * 60}")
        cache_dir = Path(args.cache_harnesses)
        harness_dirs = [None] * len(instances)
        applied_flags = [False] * len(instances)
        for i, gt in enumerate(instances):
            iid = gt["instance_id"].replace("/", "_")
            issue_dir = cache_dir / f"issue_{i}_{iid}"
            if issue_dir.exists() and (issue_dir / "harness.py").exists():
                harness_dirs[i] = str(issue_dir)
                # Check if it differs from general (i.e., patch was applied)
                import filecmp
                if not filecmp.cmp(
                    str(issue_dir / "harness.py"),
                    str(Path(args.harness_dir) / "harness.py"),
                    shallow=False,
                ):
                    applied_flags[i] = True
            else:
                print(f"  WARNING: no cached harness for issue {i} ({iid})")
        valid = sum(1 for h in harness_dirs if h is not None)
        n_applied = sum(applied_flags)
        print(f"  Loaded {valid}/{len(instances)} cached harnesses "
              f"({n_applied} with patches applied)")
        patches = [None] * len(instances)
    else:
        print(f"\n{'=' * 60}")
        print("Phase 1+2: Generate patches + Apply (once)")
        print(f"{'=' * 60}")

        patches = batch_generate_patches(
            instances, args.harness_dir, args.patch_model,
            patch_api_base=args.patch_api_base,
            max_workers=args.max_workers * 2,
        )

        harness_dirs, applied_flags = batch_apply_and_validate(
            instances, patches, args.harness_dir,
            work_dir=args.cache_harnesses,
        )

    # Phase 3: execute multiple runs with the SAME patched harnesses
    all_run_scores = []
    all_results = []

    traj_base = None
    if args.out:
        traj_base = str(Path(args.out).with_suffix("")) + "_trajectories"

    for run_idx in range(args.num_runs):
        print(f"\n{'=' * 60}")
        print(f"Run {run_idx + 1}/{args.num_runs} (execution only)")
        print(f"{'=' * 60}")

        traj_dir = f"{traj_base}/run_{run_idx + 1}" if traj_base else None
        exec_results = batch_execute(
            instances, harness_dirs, args.student, args.repo,
            args.max_steps, args.cost_limit,
            max_workers=min(args.max_workers, 10),
            trajectory_dir=traj_dir,
        )

        eval_results = []
        for i, gt in enumerate(instances):
            r = exec_results[i] or {}
            eval_results.append(PatchEvalResult(
                instance_id=gt["instance_id"],
                resolved=r.get("resolved", False),
                status=r.get("status", "unknown"),
                patch_text=str(patches[i])[:500] if patches[i] else "",
                adapted_dir=harness_dirs[i] or "",
                steps=r.get("steps", 0),
                agent_cost=r.get("agent_cost", 0),
                patch_applied=applied_flags[i],
            ))

        scores = [1.0 if er.resolved else 0.0 for er in eval_results]
        all_run_scores.append(scores)
        all_results.append(eval_results)

        resolved = sum(scores)
        total = len(scores)
        errors = sum(1 for er in eval_results if "ERROR" in er.status)
        costs = [er.agent_cost for er in eval_results]
        steps_resolved = [er.steps for er in eval_results if er.resolved]
        avg_steps_resolved = (sum(steps_resolved) / len(steps_resolved)
                              if steps_resolved else 0)
        print(f"  Run {run_idx + 1}: {int(resolved)}/{total} = "
              f"{resolved/total*100:.2f}%  errors={errors}  "
              f"agent_cost=${sum(costs):.2f}  "
              f"avg_steps_resolved={avg_steps_resolved:.1f}")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"EVALUATION SUMMARY: patch-advisor (model={args.patch_model})")
    print(f"{'=' * 70}\n")

    stats = compute_multi_run_statistics(all_run_scores)
    print(f"Resolve Rate: {format_ci_string(stats)}")
    print(f"Number of runs: {args.num_runs}")
    print(f"Total instances per run: {len(instances)}")
    n_applied = sum(applied_flags)
    print(f"Patches applied: {n_applied}/{len(instances)} "
          f"({n_applied/len(instances)*100:.0f}%)")

    print("\nPer-run results:")
    for run_idx, scores in enumerate(all_run_scores):
        resolved = sum(scores)
        total = len(scores)
        print(f"  Run {run_idx + 1}: {int(resolved)}/{total} = "
              f"{resolved/total*100:.2f}%")

    # Save results
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        per_run_issues = []
        for eval_results in all_results:
            run_issues = []
            for er in eval_results:
                run_issues.append({
                    "instance_id": er.instance_id,
                    "resolved": er.resolved,
                    "status": er.status,
                    "steps": er.steps,
                    "agent_cost": er.agent_cost,
                    "patch_applied": er.patch_applied,
                })
            per_run_issues.append(run_issues)

        out_data = {
            "harness_dir": str(args.harness_dir),
            "patch_model": args.patch_model,
            "data_file": args.data_file,
            "num_runs": args.num_runs,
            "patches_applied": n_applied,
            "patches_total": len(instances),
            "per_run_rates": [sum(s) / len(s) for s in all_run_scores],
            "per_run_issues": per_run_issues,
            "stats": stats,
        }
        json.dump(out_data, open(out_path, "w"), indent=2)
        print(f"\nResults saved to {out_path}")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
