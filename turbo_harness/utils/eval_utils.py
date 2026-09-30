"""Shared utilities for evaluation scripts.

Provides:
- vLLM server management (start/stop)
- Confidence interval calculation (bootstrap + SEM)
"""

import os
import shutil
import subprocess
from typing import List, Optional

import numpy as np

def setup_vllm_server(
    model_path: str,
    served_model_name: str = "advisor_model",
    tensor_parallel_size: int = 4,
    max_model_len: int = 32768,
) -> subprocess.Popen:
    """Start a vLLM server for the given model.

    Args:
        model_path: Path to the model (local or HF identifier)
        served_model_name: Name to serve the model as
        tensor_parallel_size: Number of GPUs for tensor parallelism
        max_model_len: Maximum model length

    Returns:
        Process object for the vLLM server
    """
    print(f"Starting vLLM server for model: {model_path}")
    print(f"  Served as: {served_model_name}")
    print(f"  Tensor parallel size: {tensor_parallel_size}")
    print(f"  Max model length: {max_model_len}")

    from turbo_harness.utils.vllm import start_vllm_server

    process = start_vllm_server(
        model_to_serve_name=model_path,
        served_model_name=served_model_name,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
    )

    return process


def cleanup_vllm_server(
    process: Optional[subprocess.Popen],
    temp_dir: Optional[str] = None,
) -> None:
    """Stop vLLM server and cleanup temporary directories.

    Args:
        process: vLLM server process to terminate
        temp_dir: Temporary directory to remove
    """
    if process is not None:
        print("Stopping vLLM server...")
        process.terminate()
        try:
            process.wait(timeout=10)
            print("vLLM server stopped successfully")
        except subprocess.TimeoutExpired:
            print("vLLM server did not stop gracefully, killing...")
            process.kill()
            process.wait()

    if temp_dir is not None and os.path.exists(temp_dir):
        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)


def compute_sem(scores: List[float]) -> float:
    """Compute standard error of the mean.

    Args:
        scores: List of scores

    Returns:
        Standard error of the mean (NOT multiplied by 1.96)
    """
    if not scores or len(scores) < 2:
        return 0.0

    scores_array = np.array(scores)
    return float(np.std(scores_array, ddof=1) / np.sqrt(len(scores_array)))


def compute_multi_run_statistics(
    all_run_scores: List[List[float]],
    confidence_level: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """Compute statistics for multi-run evaluation with proper bootstrap CI.

    Bootstrap CI: Flatten all scores, resample with size of one eval set,
    compute means of resamples, get 2.5th and 97.5th percentiles.

    SEM: Compute mean of each run, then compute SEM of those run means.

    Args:
        all_run_scores: List of lists, where each inner list contains all scores
                        from one evaluation run
        confidence_level: Confidence level for bootstrap CI
        n_bootstrap: Number of bootstrap samples
        seed: Random seed for reproducibility

    Returns:
        Dictionary with mean, sem, bootstrap_ci_lower, bootstrap_ci_upper, n_runs
    """
    if not all_run_scores or not any(all_run_scores):
        return {
            "mean": 0.0,
            "sem": 0.0,
            "bootstrap_ci_lower": 0.0,
            "bootstrap_ci_upper": 0.0,
            "n": 0,
        }

    # Compute mean of each run for SEM calculation
    run_means = [np.mean(run_scores) for run_scores in all_run_scores if run_scores]
    n_runs = len(run_means)
    overall_mean = np.mean(run_means)

    # SEM is computed from the run means (standard error across runs)
    sem = compute_sem(run_means)

    # For bootstrap CI, flatten all scores and resample with size of one eval set
    all_scores_flat = []
    for run_scores in all_run_scores:
        all_scores_flat.extend(run_scores)

    all_scores_array = np.array(all_scores_flat)
    eval_set_size = (
        len(all_run_scores[0]) if all_run_scores[0] else len(all_scores_flat)
    )

    if len(all_scores_array) == 0:
        return {
            "mean": 0.0,
            "sem": 0.0,
            "bootstrap_ci_lower": 0.0,
            "bootstrap_ci_upper": 0.0,
            "n": 0,
        }

    # Set random seed for reproducibility
    rng = np.random.default_rng(seed)

    # Generate bootstrap samples - resample with size of one eval set
    bootstrap_means = []
    for _ in range(n_bootstrap):
        bootstrap_sample = rng.choice(
            all_scores_array, size=eval_set_size, replace=True
        )
        bootstrap_means.append(np.mean(bootstrap_sample))

    bootstrap_means = np.array(bootstrap_means)

    # Compute percentiles for confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return {
        "mean": overall_mean,
        "sem": sem,
        "bootstrap_ci_lower": ci_lower,
        "bootstrap_ci_upper": ci_upper,
        "n": n_runs,
    }


def format_ci_string(stats: dict, metric_name: str = "Score") -> str:
    """Format statistics into a readable string for paper reporting.

    Args:
        stats: Dictionary from compute_statistics
        metric_name: Name of the metric

    Returns:
        Formatted string with mean, ± margin, 95% CI, and SEM
    """
    mean = stats["mean"]
    ci_lower = stats["bootstrap_ci_lower"]
    ci_upper = stats["bootstrap_ci_upper"]
    sem = stats["sem"]
    n = stats["n"]

    return (
        f"{metric_name}: {mean:.4f} (± {sem:.4f} SEM 95% CI parametric, assumes normality) "
        f"[95% Bootstrapped CI: {ci_lower:.4f} - {ci_upper:.4f}] "
        f"(n={n})"
    )


def add_common_eval_args(parser) -> None:
    """Add common evaluation arguments to an argument parser.

    Args:
        parser: argparse.ArgumentParser instance
    """
    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name (HuggingFace model identifier or path to local model)",
    )

    # vLLM configuration
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=4,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=32768,
        help="Maximum model length for vLLM server",
    )

    # Evaluation configuration
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of times to run evaluation on the test set for confidence intervals",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to evaluation dataset (parquet file)",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate (None = all)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=12,
        help="Number of parallel workers for evaluation",
    )
    # Student model configuration
    parser.add_argument(
        "--student_model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use as student",
    )
