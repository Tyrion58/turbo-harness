"""Baseline evaluation script for SWE-Smith domain (no advisor).

Evaluates API models on SWE-Smith software engineering tasks using remote agent
execution in Docker containers (same as training environment, but without advisor).

The agent runs autonomously with its default system prompt until termination.

Usage:
    # Single run
    python -m turbo_harness.eval.baseline \
        --model vertex_ai/claude-haiku-4-5 \
        --data_file data/swe_smith/test_iterative__dvc.json \
        --max_workers 20

    # Multiple runs for confidence intervals
    python -m turbo_harness.eval.baseline \
        --model vertex_ai/claude-haiku-4-5 \
        --data_file data/swe_smith/test_iterative__dvc.json \
        --max_workers 20 \
        --num_runs 3

Note: Requires AGENT_SERVER_URL and EVAL_SERVER_URL environment variables.
"""

import argparse
import json
import sys
from typing import Dict, List, Any
from dataclasses import dataclass
from tqdm import tqdm
import litellm
from concurrent.futures import ThreadPoolExecutor, as_completed

from turbo_harness.infra.scoring import compute_score
from turbo_harness.infra.agent_client import RemoteAgentClient
from turbo_harness.utils.eval_utils import compute_multi_run_statistics, format_ci_string


@dataclass
class EvalResult:
    """Single evaluation result."""

    instance_id: str
    problem_statement: str
    repo: str
    generated_patch: str
    ground_truth_patch: str
    resolved: bool
    status: str
    total_steps: int = 0
    agent_cost: float = 0.0
    session_id: str = None
    eval_id: str = None
    error: str = None
    diff: str = None


class SWESmithEvaluator:
    """Evaluator for SWE-Smith problems using remote agent (no advisor)."""

    def __init__(self, model_name: str):
        """Initialize the evaluator with model name.

        Args:
            model_name: Name of the model to use for the agent
        """
        print(f"Using agent model: {model_name}")
        sys.stdout.flush()
        self.model_name = model_name
        litellm.drop_params = True

        # Remote agent client
        self.remote_agent = RemoteAgentClient()
        print("Remote agent client initialized")
        sys.stdout.flush()

    def generate_patch(self, instance: Dict[str, Any]) -> tuple[str, dict]:
        """Generate a patch using remote agent (no advisor guidance).

        Args:
            instance: Full SWE-Smith instance dict

        Returns:
            Tuple of (patch_string, info_dict)
        """
        session_id = None
        try:
            repo_name = instance["repo"]
            instance_id = instance["instance_id"]

            print(f"Repo: {repo_name}")
            print(f"Instance: {instance_id}")
            sys.stdout.flush()

            # Create remote agent session with specified model
            session_id = self.remote_agent.create_session(
                instance, model_name=self.model_name
            )
            print(f"Created remote session: {session_id} with model {self.model_name}")
            sys.stdout.flush()

            # Run agent without advisor (empty feedback = agent runs autonomously)
            # The agent will use its default system prompt and run until termination
            result = self.remote_agent.execute_step(
                session_id,
                advisor_feedback="",
                max_steps=40,
            )

            total_steps = result.get("total_steps", 0)
            cost = result.get("cost", 0.0)
            terminated = result.get("terminated", False)

            print(
                f"Agent completed: {total_steps} steps, cost: ${cost:.4f}, terminated: {terminated}"
            )
            sys.stdout.flush()

            # Get patch from remote session
            patch = self.remote_agent.get_patch(session_id)

            # Save session_id before cleanup
            completed_session_id = session_id

            # Cleanup
            self.remote_agent.cleanup_session(session_id)
            session_id = None

            info = {
                "status": "completed" if terminated else "incomplete",
                "n_steps": total_steps,
                "cost": cost,
                "session_id": completed_session_id,
            }

            return patch, info

        except Exception as e:
            print(f"Error generating patch with remote agent: {e}")
            import traceback

            traceback.print_exc()
            sys.stdout.flush()

            # Cleanup on error
            if session_id:
                try:
                    self.remote_agent.cleanup_session(session_id)
                except Exception:
                    pass

            return "", {
                "status": "error",
                "message": str(e),
                "n_steps": 0,
                "cost": 0.0,
                "session_id": session_id,
            }

    def evaluate_single(self, instance: Dict[str, Any]) -> EvalResult:
        """Evaluate a single SWE-Smith problem."""
        instance_id = instance["instance_id"]
        problem_statement = instance["problem_statement"]
        repo = instance["repo"]
        ground_truth_patch = instance["patch"]

        print(f"\n{'=' * 60}")
        print(f"Evaluating: {instance_id}")
        print(f"Repo: {repo}")
        print(f"{'=' * 60}")
        sys.stdout.flush()

        try:
            # Generate patch using remote agent (no advisor)
            print("Generating patch with remote agent (no advisor)...")
            sys.stdout.flush()
            generated_patch, agent_info = self.generate_patch(instance)
            print(f"Generated patch length: {len(generated_patch)} chars")
            print(f"Agent info: {agent_info}")
            sys.stdout.flush()

            # Evaluate using SWE-Smith harness
            reward, run_id, info = compute_score(generated_patch, instance)

            print(f"Evaluation complete: {info}")
            sys.stdout.flush()

            # Parse info string
            resolved = reward > 0.5
            status = (
                info.split("Status: ")[1].split(",")[0]
                if "Status:" in info
                else "unknown"
            )

            result_str = "RESOLVED" if resolved else "NOT RESOLVED"
            print(f"Result: {result_str} (status: {status})")

            # Add agent info to result
            agent_status = agent_info.get("status", "unknown")
            agent_steps = agent_info.get("n_steps", 0)
            agent_cost = agent_info.get("cost", 0.0)
            print(
                f"Agent status: {agent_status}, Steps: {agent_steps}, Cost: ${agent_cost:.4f}"
            )
            sys.stdout.flush()

            return EvalResult(
                instance_id=instance_id,
                problem_statement=problem_statement,
                repo=repo,
                generated_patch=generated_patch,
                ground_truth_patch=ground_truth_patch,
                resolved=resolved,
                status=f"{status} (agent: {agent_status}, steps: {agent_steps})",
                total_steps=agent_steps,
                agent_cost=agent_cost,
                session_id=agent_info.get("session_id"),
                eval_id=run_id,
                diff=generated_patch,
            )

        except Exception as e:
            print(f"ERROR: {str(e)}")
            return EvalResult(
                instance_id=instance_id,
                problem_statement=problem_statement,
                repo=repo,
                generated_patch="",
                ground_truth_patch=ground_truth_patch,
                resolved=False,
                status="error",
                total_steps=0,
                agent_cost=0.0,
                error=str(e),
            )

    def evaluate_dataset(
        self, instances: List[Dict[str, Any]], max_workers: int = 1
    ) -> List[EvalResult]:
        """Evaluate multiple instances with optional parallelization.

        Args:
            instances: List of problem instances to evaluate
            max_workers: Number of parallel workers (1 = sequential)

        Returns:
            List of evaluation results
        """
        if max_workers == 1:
            # Sequential execution
            results = []
            for instance in tqdm(instances, desc="Evaluating"):
                result = self.evaluate_single(instance)
                results.append(result)
            return results

        # Parallel execution
        results = [None] * len(instances)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self.evaluate_single, instance): idx
                for idx, instance in enumerate(instances)
            }

            # Collect results with progress bar
            with tqdm(total=len(instances), desc="Evaluating") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        results[idx] = result
                    except Exception as e:
                        print(f"\nError evaluating instance {idx}: {e}")
                        import traceback

                        traceback.print_exc()
                        # Create error result
                        instance = instances[idx]
                        results[idx] = EvalResult(
                            instance_id=instance.get("instance_id", "unknown"),
                            problem_statement=instance.get("problem_statement", ""),
                            repo=instance.get("repo", ""),
                            generated_patch="",
                            ground_truth_patch=instance.get("patch", ""),
                            resolved=False,
                            status="error",
                            total_steps=0,
                            agent_cost=0.0,
                            error=str(e),
                        )
                    pbar.update(1)

        return results


def save_session_mapping(results: List[EvalResult], output_file: str):
    """Save session ID to success/failure mapping.

    Args:
        results: List of evaluation results
        output_file: Path to save the mapping JSON file
    """
    mapping = {}
    for r in results:
        if r.session_id:
            mapping[r.session_id] = {
                "instance_id": r.instance_id,
                "eval_id": r.eval_id,
                "diff": r.diff,
                "resolved": r.resolved,
                "status": r.status,
                "total_steps": r.total_steps,
                "error": r.error,
            }

    with open(output_file, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"\nSession mapping saved to: {output_file}")
    print(f"Total sessions tracked: {len(mapping)}")


def print_summary(results: List[EvalResult], aggregate_stats: Dict[str, Any] = None):
    """Print evaluation summary.

    Args:
        results: List of evaluation results from the last run
        aggregate_stats: Optional aggregate statistics from multiple runs
    """
    total = len(results)
    resolved = sum(1 for r in results if r.resolved)
    errors = sum(1 for r in results if r.error)

    # Split results by correctness
    correct_results = [r for r in results if r.resolved]
    incorrect_results = [r for r in results if not r.resolved]

    # Calculate overall averages (for single-run mode or cost reporting)
    avg_steps = sum(r.total_steps for r in results) / total if total > 0 else 0
    avg_cost = sum(r.agent_cost for r in results) / total if total > 0 else 0
    total_cost = sum(r.agent_cost for r in results)

    # Calculate averages for correct results
    avg_steps_correct = (
        sum(r.total_steps for r in correct_results) / len(correct_results)
        if correct_results
        else 0
    )

    # Calculate averages for incorrect results
    avg_steps_incorrect = (
        sum(r.total_steps for r in incorrect_results) / len(incorrect_results)
        if incorrect_results
        else 0
    )

    print("\n" + "=" * 80)
    print("FINAL EVALUATION SUMMARY (NO ADVISOR)")
    print("=" * 80)

    # Print multi-run statistics if available
    if aggregate_stats:
        print(f"\nNumber of runs: {aggregate_stats['resolve_rate']['n']}")
        print(f"Total instances per run: {total}")
        print(f"{format_ci_string(aggregate_stats['resolve_rate'], 'Resolve Rate')}")
        print(f"{format_ci_string(aggregate_stats['step_count'], 'Average Steps')}")
        print(
            f"{format_ci_string(aggregate_stats['correct_step_count'], 'Average Steps (Correct)')}"
        )
        print(
            f"{format_ci_string(aggregate_stats['incorrect_step_count'], 'Average Steps (Incorrect)')}"
        )
    else:
        print(f"Total instances: {total}")
        print(f"Resolved: {resolved} ({resolved / total * 100:.1f}%)")
        print(f"\nAverage steps (overall): {avg_steps:.1f}")
        print(f"  - Correct: {avg_steps_correct:.1f} (n={len(correct_results)})")
        print(f"  - Incorrect: {avg_steps_incorrect:.1f} (n={len(incorrect_results)})")

    print(f"\nErrors (last run): {errors} ({errors / total * 100:.1f}%)")
    print(f"\nAverage cost: ${avg_cost:.4f}")
    print(f"Total cost: ${total_cost:.4f}")
    print("=" * 80)

    # Print per-instance breakdown
    print("\nPer-instance results (last run):")
    for r in results:
        status_icon = "+" if r.resolved else "-"
        print(
            f"  {status_icon} {r.instance_id}: {r.status} (cost: ${r.agent_cost:.2f})"
        )
    print()


def run_multi_evaluation(
    evaluator: SWESmithEvaluator,
    instances: List[Dict[str, Any]],
    num_runs: int = 1,
    max_workers: int = 1,
) -> Dict[str, Any]:
    """Run evaluation multiple times and compute aggregate statistics.

    Args:
        evaluator: SWESmithEvaluator instance
        instances: List of problem instances
        num_runs: Number of evaluation runs
        max_workers: Number of parallel workers

    Returns:
        Dictionary with all run results and aggregate statistics
    """
    # List of lists: all individual scores from each run
    all_run_resolve_scores = []
    all_run_step_counts = []
    all_run_correct_step_counts = []
    all_run_incorrect_step_counts = []
    all_run_results = []

    for run_idx in range(num_runs):
        print(f"\n{'=' * 80}")
        print(f"EVALUATION RUN {run_idx + 1}/{num_runs}")
        print(f"{'=' * 80}")

        results = evaluator.evaluate_dataset(instances, max_workers=max_workers)
        all_run_results.append(results)

        # Collect all individual scores from this run
        run_resolve_scores = [1.0 if r.resolved else 0.0 for r in results]
        run_step_counts = [float(r.total_steps) for r in results]

        # Collect correct/incorrect breakdowns
        correct_results = [r for r in results if r.resolved]
        incorrect_results = [r for r in results if not r.resolved]
        run_correct_step_counts = [float(r.total_steps) for r in correct_results]
        run_incorrect_step_counts = [float(r.total_steps) for r in incorrect_results]

        all_run_resolve_scores.append(run_resolve_scores)
        all_run_step_counts.append(run_step_counts)
        all_run_correct_step_counts.append(run_correct_step_counts)
        all_run_incorrect_step_counts.append(run_incorrect_step_counts)

        # Print run summary
        resolved = sum(1 for r in results if r.resolved)
        resolve_rate = resolved / len(results) if results else 0.0
        avg_steps = (
            sum(run_step_counts) / len(run_step_counts) if run_step_counts else 0.0
        )

        print(
            f"Run {run_idx + 1} resolve rate: {resolve_rate:.4f} ({resolved}/{len(results)})"
        )
        print(f"Run {run_idx + 1} avg steps: {avg_steps:.2f}")

    # Compute aggregate statistics across runs using all individual scores
    aggregate_stats = {
        "resolve_rate": compute_multi_run_statistics(all_run_resolve_scores),
        "step_count": compute_multi_run_statistics(all_run_step_counts),
        "correct_step_count": compute_multi_run_statistics(all_run_correct_step_counts),
        "incorrect_step_count": compute_multi_run_statistics(
            all_run_incorrect_step_counts
        ),
    }

    return {
        "run_results": all_run_results,
        "run_scores": all_run_resolve_scores,
        "aggregate_stats": aggregate_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate API model on SWE-Smith")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model to evaluate (e.g., gpt-4o-mini, gemini/gemini-2.5-flash)",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to JSONL file with evaluation data (e.g., data/swe_smith/validation_jd__tenacity.json)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None = all)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=20,
        help="Number of parallel workers (1 = sequential, >1 = parallel)",
    )
    parser.add_argument(
        "--session_mapping_file",
        type=str,
        default="session_mapping.json",
        help="Output file for session ID mapping",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of evaluation runs for confidence intervals (default: 1)",
    )

    args = parser.parse_args()

    # Load dataset from JSONL file
    print(f"Loading data from {args.data_file}...")
    instances = []
    with open(args.data_file, "r") as f:
        for line in f:
            row = json.loads(line.strip())
            reward_spec = row["reward_spec"]
            # Handle both old format (ground_truth dict) and new format (ground_truth_json string)
            if "ground_truth_json" in reward_spec:
                ground_truth = json.loads(reward_spec["ground_truth_json"])
            else:
                ground_truth = reward_spec["ground_truth"]
            instances.append(ground_truth)

    # Limit samples if specified
    if args.num_samples and len(instances) > args.num_samples:
        import random

        random.seed(42)
        instances = random.sample(instances, args.num_samples)

    print(f"Evaluating {len(instances)} instances...")
    if args.max_workers > 1:
        print(f"Using {args.max_workers} parallel workers")
    if args.num_runs > 1:
        print(f"Running {args.num_runs} evaluation runs for confidence intervals")

    # Run evaluation
    evaluator = SWESmithEvaluator(args.model)

    if args.num_runs > 1:
        # Run multi-evaluation
        multi_results = run_multi_evaluation(
            evaluator=evaluator,
            instances=instances,
            num_runs=args.num_runs,
            max_workers=args.max_workers,
        )

        # Save session mapping (from last run)
        save_session_mapping(
            multi_results["run_results"][-1], args.session_mapping_file
        )

        # Print summary
        print_summary(
            multi_results["run_results"][-1],
            aggregate_stats=multi_results["aggregate_stats"],
        )
    else:
        # Single run
        results = evaluator.evaluate_dataset(instances, max_workers=args.max_workers)

        # Save session mapping
        save_session_mapping(results, args.session_mapping_file)

        # Print summary
        print_summary(results)


if __name__ == "__main__":
    main()
