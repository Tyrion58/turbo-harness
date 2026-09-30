"""Evaluate the playbook patch-advisor pipeline on SWE-smith.

Reuses the 3-phase pattern from patch_eval.py:
  Phase 1: Load playbook, match issue characteristics, generate patches
  Phase 2: Apply patches (static)
  Phase 3: Execute adapted harnesses with student model

Usage:
    EVAL_SERVER_URL=http://localhost:5152 \
    VERTEXAI_PROJECT=your-gcp-project VERTEXAI_LOCATION=us-east5 \
    python -m turbo_harness.eval.playbook_eval \
        --harness-dir artifacts/multi_repo/haiku/multi_repo_5iter/general \
        --data_file data/swe_smith/test_multi_repo.json \
        --playbook playbook.json \
        --patch-model vertex_ai/claude-sonnet-4-5@20250929 \
        --num_runs 3 --max_workers 10
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "turbo_harness/infra/mini-swe-agent/src"))

from turbo_harness.playbook.memory_advisor import format_playbook_context  # noqa: E402
from turbo_harness.playbook.schemas import Playbook  # noqa: E402
from turbo_harness.eval.patch_eval import (  # noqa: E402
    PatchEvalResult,
    batch_apply_and_validate,
    batch_execute,
    load_instances,
)
from turbo_harness.patch_advisor import generate_patch  # noqa: E402
from turbo_harness.utils.eval_utils import compute_multi_run_statistics, format_ci_string  # noqa: E402


def batch_generate_with_playbook(
    instances,
    general_harness_dir,
    playbook_path,
    patch_model,
    patch_api_base=None,
    max_workers=20,
):
    playbook = Playbook.load(playbook_path)
    playbook_context = format_playbook_context(playbook)

    print(
        f"\n  Phase 1: Generating patches with full playbook "
        f"({len(playbook.entries)} entries, {len(playbook_context)} chars)"
    )
    print(f"  Model: {patch_model}, Concurrency: {max_workers}")

    patches = [None] * len(instances)
    t0 = time.time()

    def gen_one(idx):
        gt = instances[idx]
        edits = generate_patch(
            general_harness_dir,
            gt["problem_statement"],
            model_name=patch_model,
            api_base=patch_api_base,
            playbook_context=playbook_context,
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
    print(
        f"  Phase 1 done: {elapsed:.1f}s, {n_valid} patches, "
        f"{n_empty} empty, {n_error} errors"
    )

    return patches


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate playbook patch-advisor on SWE-smith"
    )
    parser.add_argument("--harness-dir", required=True)
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--playbook", required=True)
    parser.add_argument("--patch-model", default="vertex_ai/claude-sonnet-4-5@20250929")
    parser.add_argument("--patch-api-base", default=None)
    parser.add_argument("--student", default="vertex_ai/claude-haiku-4-5")
    parser.add_argument("--repo", default="multi_repo")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--cost-limit", type=float, default=3.0)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--out", default=None,
                        help="Output JSON path (auto-generated if not set)")
    parser.add_argument("--cache-harnesses", default=None,
                        help="Dir for patched harnesses (auto-generated if not set)")
    parser.add_argument("--exec-only", action="store_true")
    parser.add_argument("--run-name", default=None,
                        help="Run name for auto-generated paths")
    args = parser.parse_args()

    import litellm
    from datetime import datetime

    litellm.drop_params = True

    # Auto-generate output paths if not specified
    patch_model_slug = args.patch_model.split("/")[-1].split("@")[0].replace("-", "")
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.out:
        args.out = f"experiments/results/ace_{patch_model_slug}_{run_name}.json"
    if not args.cache_harnesses:
        playbook_dir = str(Path(args.playbook).parent)
        args.cache_harnesses = f"{playbook_dir}/cached_harnesses_{run_name}"

    instances = load_instances(args.data_file)
    if args.num_samples and len(instances) > args.num_samples:
        instances = instances[: args.num_samples]

    print(f"Loaded {len(instances)} instances from {args.data_file}")
    print(f"General harness: {args.harness_dir}")
    print(f"Playbook: {args.playbook}")
    print(f"Patch model: {args.patch_model}")
    print(f"Student: {args.student}")
    print(f"Runs: {args.num_runs}, Workers: {args.max_workers}")
    print(f"Output: {args.out}")
    print(f"Cache: {args.cache_harnesses}")

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
                import filecmp

                if not filecmp.cmp(
                    str(issue_dir / "harness.py"),
                    str(Path(args.harness_dir) / "harness.py"),
                    shallow=False,
                ):
                    applied_flags[i] = True
        patches = [None] * len(instances)
    else:
        print(f"\n{'=' * 60}")
        print("Phase 1+2: Generate patches with playbook + Apply")
        print(f"{'=' * 60}")

        patches = batch_generate_with_playbook(
            instances,
            args.harness_dir,
            args.playbook,
            args.patch_model,
            args.patch_api_base,
            max_workers=args.max_workers * 2,
        )

        harness_dirs, applied_flags = batch_apply_and_validate(
            instances,
            patches,
            args.harness_dir,
            work_dir=args.cache_harnesses,
        )

    all_run_scores = []
    all_results = []

    traj_base = None
    if args.out:
        traj_base = str(Path(args.out).with_suffix("")) + "_trajectories"

    for run_idx in range(args.num_runs):
        print(f"\n{'=' * 60}")
        print(f"Run {run_idx + 1}/{args.num_runs}")
        print(f"{'=' * 60}")

        traj_dir = f"{traj_base}/run_{run_idx + 1}" if traj_base else None
        exec_results = batch_execute(
            instances,
            harness_dirs,
            args.student,
            args.repo,
            args.max_steps,
            args.cost_limit,
            max_workers=min(args.max_workers, 10),
            trajectory_dir=traj_dir,
        )

        eval_results = []
        for i, gt in enumerate(instances):
            r = exec_results[i] or {}
            eval_results.append(
                PatchEvalResult(
                    instance_id=gt["instance_id"],
                    resolved=r.get("resolved", False),
                    status=r.get("status", "unknown"),
                    patch_text=str(patches[i])[:500] if patches[i] else "",
                    adapted_dir=harness_dirs[i] or "",
                    steps=r.get("steps", 0),
                    agent_cost=r.get("agent_cost", 0),
                    patch_applied=applied_flags[i],
                )
            )

        scores = [1.0 if er.resolved else 0.0 for er in eval_results]
        all_run_scores.append(scores)
        all_results.append(eval_results)

        resolved = sum(scores)
        total = len(scores)
        errors = sum(1 for er in eval_results if "ERROR" in er.status)
        costs = [er.agent_cost for er in eval_results]
        steps_resolved = [er.steps for er in eval_results if er.resolved]
        avg_steps_resolved = (
            sum(steps_resolved) / len(steps_resolved) if steps_resolved else 0
        )
        print(
            f"  Run {run_idx + 1}: {int(resolved)}/{total} = "
            f"{resolved / total * 100:.2f}%  errors={errors}  "
            f"agent_cost=${sum(costs):.2f}  "
            f"avg_steps_resolved={avg_steps_resolved:.1f}"
        )

    print(f"\n{'=' * 70}")
    print("EVALUATION SUMMARY: playbook patch-advisor")
    print(f"{'=' * 70}\n")

    stats = compute_multi_run_statistics(all_run_scores)
    print(f"Resolve Rate: {format_ci_string(stats)}")
    print(f"Number of runs: {args.num_runs}")
    print(f"Total instances per run: {len(instances)}")
    n_applied = sum(applied_flags)
    print(
        f"Patches applied: {n_applied}/{len(instances)} "
        f"({n_applied / len(instances) * 100:.0f}%)"
    )

    print("\nPer-run results:")
    for run_idx, scores in enumerate(all_run_scores):
        resolved = sum(scores)
        total = len(scores)
        print(
            f"  Run {run_idx + 1}: {int(resolved)}/{total} = "
            f"{resolved / total * 100:.2f}%"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        per_run_issues = []
        for eval_results in all_results:
            run_issues = []
            for er in eval_results:
                run_issues.append(
                    {
                        "instance_id": er.instance_id,
                        "resolved": er.resolved,
                        "status": er.status,
                        "steps": er.steps,
                        "agent_cost": er.agent_cost,
                        "patch_applied": er.patch_applied,
                    }
                )
            per_run_issues.append(run_issues)

        out_data = {
            "harness_dir": str(args.harness_dir),
            "playbook": args.playbook,
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
