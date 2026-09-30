"""Evaluate a custom harness on a SWE-smith data split, multi-trial.

Uses executor.run_instance() directly (inline Docker) for custom harness support.
Set EVAL_SERVER_URL for reliable scoring via eval_server.

Usage (requires eval_server running):
  EVAL_SERVER_URL=http://localhost:5152 \
  VERTEXAI_PROJECT=your-gcp-project VERTEXAI_LOCATION=us-east5 \
  python -m turbo_harness.eval.harness_eval \
      --harness artifacts/dvc/general \
      --data_file data/swe_smith/test_iterative__dvc.json \
      --student vertex_ai/claude-haiku-4-5 \
      --num_runs 3 --max_workers 5
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

import litellm  # noqa: E402
from turbo_harness import executor as EX  # noqa: E402
from turbo_harness.utils.eval_utils import compute_multi_run_statistics, format_ci_string  # noqa: E402


def _register_local_model(model_name):
    """Register a local vLLM model with litellm so cost calculation doesn't crash."""
    litellm.drop_params = True
    litellm.register_model({
        model_name: {
            "max_tokens": 40960,
            "max_input_tokens": 40960,
            "max_output_tokens": 40960,
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
        }
    })


def load_instances(data_file):
    rows = [json.loads(line) for line in open(data_file) if line.strip()]
    return [json.loads(r["reward_spec"]["ground_truth_json"]) for r in rows]


def eval_one_run(harness, instances, student, max_workers, repo,
                 step_limit, cost_limit, run_idx, trajectory_dir=None):
    """Run one complete evaluation pass over all instances."""
    results = {}

    def run_one(idx):
        gt = instances[idx]
        t0 = time.time()
        r = EX.run_instance(gt, student, harness=harness, repo=repo,
                            step_limit=step_limit, cost_limit=cost_limit,
                            trajectory_dir=trajectory_dir)
        r["idx"] = idx
        r["solve_s"] = time.time() - t0
        return idx, r

    n = len(instances)
    print(f"\n  Run {run_idx + 1}: evaluating {n} issues with harness='{harness}'...")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(run_one, i): i for i in range(n)}
        done = 0
        for f in as_completed(futs):
            idx = futs[f]
            try:
                _, r = f.result()
                results[idx] = r
            except Exception as e:
                results[idx] = {
                    "idx": idx, "resolved": False,
                    "status": f"ERROR:{type(e).__name__}",
                    "steps": 0, "agent_cost": 0,
                }
            done += 1
            if done % 20 == 0 or done == n:
                resolved = sum(1 for r in results.values() if r["resolved"])
                print(f"    {done}/{n} done, {resolved} resolved so far...")

    resolved = sum(1 for r in results.values() if r["resolved"])
    total = len(results)
    errors = sum(1 for r in results.values() if "ERROR" in str(r.get("status", "")))
    costs = [r.get("agent_cost", 0) for r in results.values()]
    resolved_results = [r for r in results.values() if r["resolved"]]
    avg_steps_resolved = (sum(r.get("steps", 0) for r in resolved_results) / len(resolved_results)) if resolved_results else 0
    avg_steps_all = sum(r.get("steps", 0) for r in results.values()) / total if total else 0
    print(f"  Run {run_idx + 1}: {resolved}/{total} = {resolved/total*100:.2f}%  "
          f"errors={errors}  mean_cost=${sum(costs)/len(costs):.3f}  "
          f"avg_steps={avg_steps_all:.1f} (resolved: {avg_steps_resolved:.1f})")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a custom harness on SWE-smith")
    parser.add_argument("--harness", required=True,
                        help="Harness dir (e.g. artifacts/dvc/general) or 'default'")
    parser.add_argument("--data_file", required=True,
                        help="JSONL data file (e.g. data/swe_smith/test_iterative__dvc.json)")
    parser.add_argument("--student", default="vertex_ai/claude-haiku-4-5")
    parser.add_argument("--repo", default="dvc")
    parser.add_argument("--num_runs", type=int, default=1,
                        help="Number of evaluation runs for CI")
    parser.add_argument("--max_workers", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--cost-limit", type=float, default=3.0)
    parser.add_argument("--baseline-rate", type=float, default=None,
                        help="Baseline rate for comparison (e.g. 0.4167)")
    parser.add_argument("--out", default=None,
                        help="Output JSON file for results")
    args = parser.parse_args()

    # Register local vLLM models with litellm (avoids cost calculation crash)
    if "openai/" in args.student or "hosted_vllm/" in args.student:
        _register_local_model(args.student)

    instances = load_instances(args.data_file)
    print(f"Loaded {len(instances)} instances from {args.data_file}")
    print(f"Harness: {args.harness}")
    print(f"Student: {args.student}")
    print(f"Runs: {args.num_runs}, Workers: {args.max_workers}")

    # Validate harness (unless default)
    if args.harness != "default":
        from turbo_harness.proposer import validate_artifact
        ok, msg = validate_artifact(Path(args.harness))
        if not ok:
            print(f"ERROR: invalid harness: {msg}")
            sys.exit(1)
        print(f"Harness valid: {msg}")

    all_run_scores = []
    all_results = []
    total_cost = 0

    traj_base = None
    if args.out:
        traj_base = str(Path(args.out).with_suffix("")) + "_trajectories"

    for run_idx in range(args.num_runs):
        traj_dir = f"{traj_base}/run_{run_idx+1}" if traj_base else None
        results = eval_one_run(
            args.harness, instances, args.student, args.max_workers,
            args.repo, args.max_steps, args.cost_limit, run_idx,
            trajectory_dir=traj_dir)
        all_results.append(results)

        scores = [1.0 if results[i]["resolved"] else 0.0 for i in range(len(instances))]
        all_run_scores.append(scores)
        total_cost += sum(r.get("agent_cost", 0) for r in results.values())

    # Summary
    print(f"\n{'=' * 70}")
    print(f"EVALUATION SUMMARY: harness={args.harness}")
    print(f"{'=' * 70}")

    if args.num_runs > 1:
        stats = compute_multi_run_statistics(all_run_scores)
        print(f"\n{format_ci_string(stats, 'Resolve Rate')}")
        print(f"Number of runs: {args.num_runs}")
    else:
        resolved = sum(all_run_scores[0])
        total = len(all_run_scores[0])
        print(f"\nResolved: {int(resolved)}/{total} = {resolved/total*100:.2f}%")

    print(f"Total instances per run: {len(instances)}")
    print(f"Total cost: ${total_cost:.2f}")
    print(f"Mean cost per issue: ${total_cost / (len(instances) * args.num_runs):.3f}")

    if args.baseline_rate is not None:
        mean_rate = sum(sum(s) / len(s) for s in all_run_scores) / len(all_run_scores)
        delta = mean_rate - args.baseline_rate
        print(f"\nBaseline: {args.baseline_rate*100:.2f}%")
        print(f"This harness: {mean_rate*100:.2f}%")
        print(f"Delta: {delta*100:+.2f} pts")

    # Per-run breakdown
    print("\nPer-run results:")
    for run_idx, scores in enumerate(all_run_scores):
        resolved = sum(scores)
        total = len(scores)
        print(f"  Run {run_idx + 1}: {int(resolved)}/{total} = {resolved/total*100:.2f}%")

    # Save results
    if args.out:
        # Per-issue results for each run
        per_run_issues = []
        for run_idx, results in enumerate(all_results):
            run_issues = []
            for i in range(len(instances)):
                r = results.get(i, {})
                run_issues.append({
                    "idx": i,
                    "instance_id": instances[i]["instance_id"],
                    "resolved": r.get("resolved", False),
                    "status": r.get("status", "unknown"),
                    "steps": r.get("steps", 0),
                    "agent_cost": r.get("agent_cost", 0),
                })
            per_run_issues.append(run_issues)

        out_data = {
            "harness": args.harness,
            "data_file": args.data_file,
            "student": args.student,
            "num_runs": args.num_runs,
            "per_run_rates": [sum(s) / len(s) for s in all_run_scores],
            "total_cost": total_cost,
            "per_run_issues": per_run_issues,
        }
        if args.num_runs > 1:
            out_data["stats"] = stats
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out_data, indent=2))
        print(f"\nResults saved to {args.out}")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
