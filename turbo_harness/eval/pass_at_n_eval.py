"""Pass@N evaluation for playbook advisor.

Generates N patches per issue with temperature sampling, executes all,
and computes pass@1, pass@3, pass@5, pass@N.

Usage:
    AGENT_SERVER_URL=http://localhost:8081 \
    EVAL_SERVER_URL=http://localhost:5152 \
    VERTEXAI_PROJECT=your-gcp-project VERTEXAI_LOCATION=global \
    OPENAI_API_KEY=dummy \
    python -m turbo_harness.eval.pass_at_n_eval \
        --harness-dir artifacts/multi_repo/haiku/multi_repo_5iter/general \
        --data_file data/swe_smith/test_multi_repo.json \
        --playbook playbook_output/multi_repo_v3/playbook.json \
        --patch-model openai/Qwen/Qwen3.5-9B \
        --patch-api-base http://localhost:8000/v1 \
        --n 8 --temperature 0.8
"""

import argparse
import json
import math
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "turbo_harness/infra/mini-swe-agent/src"))

from turbo_harness.playbook.memory_advisor import format_playbook_context  # noqa: E402
from turbo_harness.playbook.schemas import Playbook  # noqa: E402
from turbo_harness import executor as EX  # noqa: E402
from turbo_harness.patch_advisor import (  # noqa: E402
    _extract_edits,
    _load_harness_context,
    apply_patch,
    PATCH_ADVISOR_SYSTEM_PROMPT,
    PATCH_ADVISOR_USER_TEMPLATE,
)
from turbo_harness.proposer import validate_artifact  # noqa: E402


def generate_n_patches(
    general_harness_dir,
    problem_statement,
    playbook_context,
    model_name,
    api_base,
    n,
    temperature,
):
    import litellm

    litellm.drop_params = True

    harness_code, summary, concept, rules_memory = _load_harness_context(
        general_harness_dir
    )
    user_content = PATCH_ADVISOR_USER_TEMPLATE.format(
        harness_code=harness_code,
        summary=summary or "(no summary available)",
        concept=concept or "(no concept document available)",
        rules_memory=rules_memory or "(no repository knowledge available)",
        playbook=playbook_context + "\n" if playbook_context else "",
        problem_statement=problem_statement,
    )

    completion_kwargs = {
        "messages": [
            {"role": "system", "content": PATCH_ADVISOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 16384,
        "temperature": temperature,
        "n": n,
    }

    if api_base:
        completion_kwargs["model"] = model_name
        completion_kwargs["api_base"] = api_base
    else:
        completion_kwargs["model"] = model_name

    response = litellm.completion(**completion_kwargs)

    all_edits = []
    for choice in response.choices:
        raw = choice.message.content.strip()
        edits = _extract_edits(raw)
        all_edits.append(edits)

    return all_edits


def apply_and_validate_n(edits_list, general_harness_dir, work_dir, issue_idx, iid):
    general_harness_dir = Path(general_harness_dir)
    work_dir = Path(work_dir)
    harness_dirs = []

    for sample_idx, edits in enumerate(edits_list):
        out_dir = work_dir / f"issue_{issue_idx}_{iid}_sample_{sample_idx}"

        if not edits:
            shutil.copytree(
                general_harness_dir, out_dir,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                dirs_exist_ok=True,
            )
            harness_dirs.append(str(out_dir))
            continue

        success, msg = apply_patch(edits, general_harness_dir, out_dir)
        if not success:
            shutil.copytree(
                general_harness_dir, out_dir,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                dirs_exist_ok=True,
            )
            harness_dirs.append(str(out_dir))
            continue

        ok, _ = validate_artifact(out_dir)
        if not ok:
            shutil.copytree(
                general_harness_dir, out_dir,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                dirs_exist_ok=True,
            )

        harness_dirs.append(str(out_dir))

    return harness_dirs


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def main():
    parser = argparse.ArgumentParser(description="Pass@N evaluation for playbook advisor")
    parser.add_argument("--harness-dir", required=True)
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--playbook", required=True)
    parser.add_argument("--patch-model", required=True)
    parser.add_argument("--patch-api-base", default=None)
    parser.add_argument("--student", default="vertex_ai/claude-haiku-4-5")
    parser.add_argument("--repo", default="multi_repo")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_workers", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--cost-limit", type=float, default=3.0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--work-dir", default=None)
    args = parser.parse_args()

    import litellm
    litellm.drop_params = True

    rows = [json.loads(l) for l in open(args.data_file) if l.strip()]
    instances = [json.loads(r["reward_spec"]["ground_truth_json"]) for r in rows]
    print(f"Loaded {len(instances)} instances")
    print(f"N={args.n}, temperature={args.temperature}, model={args.patch_model}")

    playbook = Playbook.load(args.playbook)
    playbook_context = format_playbook_context(playbook)

    work_dir = args.work_dir or f"/tmp/pass_at_n_{args.n}"
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    # Phase 1: Generate N patches per issue
    print(f"\n{'=' * 60}")
    print(f"Phase 1: Generate {args.n} patches per issue ({len(instances)} issues)")
    print(f"{'=' * 60}")
    t0 = time.time()

    all_patches = {}

    def gen_one(idx):
        gt = instances[idx]
        edits_list = generate_n_patches(
            args.harness_dir, gt["problem_statement"],
            playbook_context, args.patch_model,
            args.patch_api_base, args.n, args.temperature,
        )
        return idx, edits_list

    with ThreadPoolExecutor(max_workers=min(args.max_workers, 20)) as pool:
        futs = {pool.submit(gen_one, i): i for i in range(len(instances))}
        done = 0
        for f in as_completed(futs):
            idx = futs[f]
            try:
                _, edits_list = f.result()
                all_patches[idx] = edits_list
            except Exception as e:
                print(f"  ERROR issue {idx}: {e}")
                all_patches[idx] = [[] for _ in range(args.n)]
            done += 1
            if done % 10 == 0 or done == len(instances):
                print(f"  {done}/{len(instances)} issues generated...")

    elapsed = time.time() - t0
    print(f"Phase 1 done: {elapsed:.1f}s")

    # Phase 2: Apply patches
    print(f"\n{'=' * 60}")
    print("Phase 2: Apply and validate patches")
    print(f"{'=' * 60}")

    all_harness_dirs = {}
    for idx in range(len(instances)):
        gt = instances[idx]
        iid = gt["instance_id"].replace("/", "_")
        harness_dirs = apply_and_validate_n(
            all_patches[idx], args.harness_dir, work_dir, idx, iid,
        )
        all_harness_dirs[idx] = harness_dirs

    # Phase 3: Execute all (issue × sample) pairs
    total_runs = len(instances) * args.n
    print(f"\n{'=' * 60}")
    print(f"Phase 3: Execute {total_runs} runs ({len(instances)} issues × {args.n} samples)")
    print(f"{'=' * 60}")
    t0 = time.time()

    results = {}  # {issue_idx: [bool, bool, ...]}

    def run_one(idx, sample_idx):
        gt = instances[idx]
        harness = all_harness_dirs[idx][sample_idx]
        r = EX.run_instance(
            gt, args.student, harness=harness, repo=args.repo,
            step_limit=args.max_steps, cost_limit=args.cost_limit,
        )
        return idx, sample_idx, r.get("resolved", False)

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = {}
        for idx in range(len(instances)):
            for s in range(args.n):
                futs[pool.submit(run_one, idx, s)] = (idx, s)

        done = 0
        for f in as_completed(futs):
            idx, s = futs[f]
            try:
                _, _, resolved = f.result()
            except Exception:
                resolved = False
            results.setdefault(idx, [False] * args.n)
            results[idx][s] = resolved
            done += 1
            if done % 50 == 0 or done == total_runs:
                total_resolved = sum(
                    any(results.get(i, [])) for i in range(len(instances))
                    if i in results
                )
                print(f"  {done}/{total_runs} runs done, "
                      f"{total_resolved} issues with ≥1 pass...")

    elapsed = time.time() - t0
    print(f"Phase 3 done: {elapsed:.1f}s")

    # Compute pass@k for various k
    print(f"\n{'=' * 70}")
    print("PASS@N RESULTS")
    print(f"{'=' * 70}")

    per_issue_passes = []
    for idx in range(len(instances)):
        c = sum(results.get(idx, []))
        per_issue_passes.append(c)

    n = args.n
    for k in [1, 2, 3, 5, 8]:
        if k > n:
            break
        scores = [pass_at_k(n, c, k) for c in per_issue_passes]
        mean = sum(scores) / len(scores)
        print(f"  pass@{k}: {mean*100:.2f}%")

    naive_pass1 = sum(1 for c in per_issue_passes if c > 0) / len(per_issue_passes)
    print(f"\n  Naive pass@1 (any pass / total): {naive_pass1*100:.2f}%")
    print(f"  Issues with ≥1 pass: {sum(1 for c in per_issue_passes if c > 0)}/{len(instances)}")
    print(f"  Total resolved: {sum(per_issue_passes)}/{total_runs}")

    # Save results
    if args.out:
        out_data = {
            "n": args.n,
            "temperature": args.temperature,
            "patch_model": args.patch_model,
            "playbook": args.playbook,
            "per_issue": [
                {
                    "instance_id": instances[idx]["instance_id"],
                    "passes": results.get(idx, []),
                    "num_pass": sum(results.get(idx, [])),
                }
                for idx in range(len(instances))
            ],
            "pass_at_k": {
                str(k): sum(pass_at_k(n, c, k) for c in per_issue_passes) / len(per_issue_passes)
                for k in [1, 2, 3, 5, 8] if k <= n
            },
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out_data, indent=2))
        print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
