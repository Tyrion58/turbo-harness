"""Configuration for SWE-Smith domain.

Contains system prompts and templates for software engineering patch generation with advisor feedback.
Defines student and advisor model prompts for generating git patches to fix bugs.
"""

from typing import Tuple
import uuid

# Advisor prompt template
ADVISOR_TEMPLATE = """You are an expert software engineering advisor that will provide advice to an autonomous agent working to fix a bug in a codebase. Your role is to provide strategic, concise guidance that helps the agent solve the problem and submit a fix as quickly as possible. You will NOT be able to interact with the codebase, but the agent will have such access.

The agent can:
- Navigate the codebase
- Read and edit files (sed, cat, python scripts, etc.)
- Run tests or minimal reproduction scripts

In short, the agent has command line access. Your job is to help the agent implement and submit a fix in as few interactions with the console as possible. Do NOT suggest unnecessary steps (e.g., refactors, additional tests, cleanup) unless they directly unblock the bug. In many cases, you may need to discourage the agent from performing additional tests. The goal is to submit a fix as fast as possible.

Suggestions and guidance for advice:
- Assist the agent in rapidly localizing the error location.
- If the agent is repeatedly fighting tooling (e.g., sed), suggest a safer alternative (manual edit, Python script, direct inspection).
- Focus the agent on the error at hand: if the agent fixes adjacent issues without reducing the original symptom, point that out.
- If the agent is stuck or going in circles, help them refocus on the original problem or suggest alternatives to break out of the loop.
- Identify when extraneous tests are unnecessary: fixes ought to be tested to verify correctness, but the goal is to submit a working fix as quickly as possible. Not every edge case needs to be tested if you're confident in the fix.
- Be concise and direct. Explicitly state your advice so the agent can act on it effectively.
- Frame your advice as advisory, not authoritative, particularly when it comes to assumptions about the codebase. The agent has access to the codebase, not you. Make sure the agent understands this.

Here is the current problem statement:
{problem_statement}

Remember: Your goal is to help the agent fix the bug as quickly as possible.
"""

ADVISOR_FOLLOWUP_TEMPLATE = """## Most Recent Agent Actions and Observations (last {len_messages} messages)

{recent_text}

Continue to guide the agent to fix the issue as efficiently as possible.

Identify and address failure patterns. Actively watch for:
- Repeated errors after "fixes"
- Repeated failed edits using the same tool or method
- Drift away from the original failing behavior
- Extraneous testing when there is already high confidence in the fix

If you see a pattern that the agent doesn't seem to see or is not addressing:
- Call it out explicitly
- Recommend specific different approach (e.g., use alternative editing tools, return to work on the original issue, stop testing and submit the fix immediately)

Now, provide more advice to the agent with the goal of submitting a working fix as quickly as possible.
"""


def build_prompt(info_dict: dict) -> str:
    """Build the advisor prompt for a given problem instance.

    Args:
        info_dict: Dictionary containing problem_statement and repo

    Returns:
        Formatted prompt string
    """
    return ADVISOR_TEMPLATE.format(
        problem_statement=info_dict.get("problem_statement", ""),
    )


def build_advisor_prompt(info_dict: dict) -> str:
    """Build the advisor prompt for a given problem instance.

    Args:
        info_dict: Dictionary containing problem_statement and repo

    Returns:
        Formatted advisor prompt string
    """
    return ADVISOR_TEMPLATE.format(
        problem_statement=info_dict.get("problem_statement", ""),
    )


def summarize_agent_progress(
    new_messages: list,
    last_summarized_step: int,
    current_step: int,
    summarizer_model: str = "gpt-4.1-mini",
) -> Tuple[str, float]:
    """Summarize new agent messages since last summary.

    Args:
        new_messages: New messages since last summarization
        last_summarized_step: Step number of last summarization
        current_step: Current step number
        summarizer_model: Model to use for summarization

    Returns:
        Tuple of (summary_text, cost)
    """
    import litellm

    if not new_messages:
        return "", 0.0

    # Format messages for summarization
    message_text = []
    for msg in new_messages:
        role = msg["role"]
        content = msg["content"]
        # Truncate very long content
        if len(content) > 1000:
            content = content[:500] + "\n... [truncated] ...\n" + content[-500:]
        message_text.append(f"[{role}]: {content}")

    messages_str = "\n\n".join(message_text)

    # Summarization prompt
    prompt = f"""Summarize the following agent interaction concisely. Focus on:
1. The approach the agent is taking
2. What errors or issues occurred
3. Any patterns (e.g., repeated errors, stuck loops)
4. Current state of progress

Be concise but capture key details that would help an advisor understand what's happening. Discuss the entire trace but focus primarily on the more recent messages.

Agent messages (steps {last_summarized_step + 1} to {current_step}):
{messages_str}

Provide a concise summary (max 100 words):"""

    try:
        response = litellm.completion(
            model=summarizer_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )

        summary = response.choices[0].message.content
        cost = litellm.completion_cost(completion_response=response)

        return summary, cost

    except Exception as e:
        print(f"Error summarizing agent progress: {e}")
        # Fallback to simple truncation
        return messages_str[:500] + "...", 0.0


def build_advisor_observation(
    agent_history: list,
    rolling_summary: str,
    last_summarized_step: int,
    current_step: int,
    summarizer_model: str = "gpt-4.1-mini",
) -> Tuple[str, str, int, float]:
    """Build observation for advisor with rolling summary + recent messages.

    Args:
        agent_history: Full agent message history
        rolling_summary: Current rolling summary
        last_summarized_step: Step number of last summarization
        current_step: Current step number
        summarizer_model: Model to use for summarization

    Returns:
        Tuple of (observation_text, updated_rolling_summary, updated_last_summarized_step, cost)
    """
    summary_cost = 0.0
    # Get most recent messages for immediate context (last 10 messages)
    recent_messages = agent_history[-10:] if len(agent_history) >= 10 else agent_history

    recent_parts = []
    for msg in recent_messages:
        role = msg["role"]
        content = msg["content"]
        recent_parts.append(f"[{role}]: {content}")

    recent_text = "\n\n".join(recent_parts)

    # Combine rolling summary with recent messages
    observation_text = ADVISOR_FOLLOWUP_TEMPLATE.format(
        len_messages=len(recent_messages), recent_text=recent_text
    )

    return observation_text, rolling_summary, last_summarized_step, summary_cost


def _verified_score_impl(patch: str, instance: dict) -> Tuple[float, str, str]:
    """The actual swebench scoring for one Verified instance. Runs inside a child process
    (see _compute_score_verified) so a hung docker op / test can be killed by the parent."""
    import os

    import docker
    from swebench.harness.test_spec.test_spec import make_test_spec
    from swebench.harness.run_evaluation import run_instance as _sb_run_instance
    from swebench.harness.constants import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION

    iid = instance["instance_id"]
    # Ensure the ~3GB instance image is available from the local tar cache (or pull-once) before
    # swebench builds/runs it, so scoring never auto-pulls into the Docker-Hub limit.
    try:
        from turbo_harness.infra.image_cache import ensure_image
        ensure_image(instance.get("image_name", ""))
    except Exception:
        pass  # best-effort; swebench will still try its own pull if the cache is unavailable
    test_spec = make_test_spec(instance, namespace="swebench")
    pred = {KEY_INSTANCE_ID: iid, KEY_MODEL: "advisor_model", KEY_PREDICTION: patch or ""}
    run_id = f"verified_{uuid.uuid4().hex[:8]}"
    client = docker.from_env()
    rm_image = os.environ.get("VERIFIED_RM_IMAGE") == "1"
    result = _sb_run_instance(
        test_spec, pred, rm_image=rm_image, force_rebuild=False,
        client=client, run_id=run_id, timeout=1800,
    ) or {}
    # swebench 4.x returns a flat {"completed","resolved"}; older builds nest under the iid.
    if "resolved" in result:
        report = result
    elif isinstance(result.get(iid), dict):
        report = result[iid]
    else:
        report = {}
    resolved = bool(report.get("resolved", False))
    return (1.0 if resolved else 0.0), run_id, f"verified resolved={resolved}"


def _force_cleanup_verified(instance):
    """Force-remove any leftover scoring container for this instance after a timeout, so a wedged
    docker op cannot keep holding resources."""
    import os
    import subprocess
    dexe = os.environ.get("MSWEA_DOCKER_EXECUTABLE", "docker")
    iid = instance.get("instance_id", "")
    if not iid:
        return
    try:
        cids = subprocess.run([dexe, "ps", "-aq", "--filter", f"name={iid}"],
                              capture_output=True, text=True, timeout=30).stdout.split()
        for cid in cids:
            subprocess.run([dexe, "rm", "-f", cid], capture_output=True, timeout=30)
    except Exception:
        pass


def _compute_score_verified(patch: str, instance: dict) -> Tuple[float, str, str]:
    """Score a prediction on one SWE-bench Verified instance via the official swebench harness.

    Runs the scoring in an ISOLATED subprocess under a HARD timeout (VERIFIED_SCORE_TIMEOUT, default
    1500s). Evolution / RL / eval score in worker threads, and a transient rootless-docker
    (fuse-overlayfs) stall can wedge a docker op past swebench's own timeout; without an outer cap a
    single wedged scoring deadlocks the whole run. `subprocess.run(timeout=…)` reliably kills the
    child on timeout (and avoids fork-from-thread hazards); leftover containers are then force-removed.
    `instance` is the gt dict from build_verified_dataset.py (repo, base_commit, test_patch,
    FAIL_TO_PASS/PASS_TO_PASS, version, environment_setup_commit)."""
    import json as _json
    import os
    import subprocess
    import sys

    timeout_s = int(os.environ.get("VERIFIED_SCORE_TIMEOUT", "1500"))
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Score with a python that HAS swebench+docker: the MAIN repo venv, NOT sys.executable — under
    # RL, sys.executable is SkyRL/.venv (no swebench) so the scorer would silently return 0 for every
    # rollout (uniform reward -> zero GRPO advantage -> no learning). Override via VERIFIED_SCORE_PYTHON.
    score_py = os.environ.get("VERIFIED_SCORE_PYTHON") or os.path.join(repo_root, ".venv", "bin", "python")
    if not os.path.exists(score_py):
        score_py = sys.executable
    payload = _json.dumps({"instance": instance, "patch": patch or ""})
    try:
        r = subprocess.run(
            [score_py, "-m", "turbo_harness.infra.score_verified_cli"],
            input=payload, capture_output=True, text=True, timeout=timeout_s, cwd=repo_root,
        )
    except subprocess.TimeoutExpired:
        _force_cleanup_verified(instance)
        return 0.0, "TIMEOUT", f"verified scoring exceeded {timeout_s}s"
    if r.returncode != 0:
        _force_cleanup_verified(instance)
        return 0.0, "ERROR", f"verified scoring subprocess exited {r.returncode}: {(r.stderr or '')[-200:]}"
    try:
        out = _json.loads(r.stdout.strip().splitlines()[-1])
        return float(out["reward"]), out["run_id"], out["info"]
    except Exception as e:
        return 0.0, "ERROR", f"verified scoring parse failed: {e}; stdout tail={r.stdout[-200:]!r}"


def compute_score(patch: str, instance: dict) -> Tuple[float, str, str]:
    """Compute score by evaluating patch using SWE-Smith harness.

    This function automatically detects whether to use local Docker evaluation
    or remote evaluation based on the EVAL_SERVER_URL environment variable.

    Args:
        patch: The generated patch string
        instance: The problem instance dict from SWE-Smith dataset

    Returns:
        Tuple of (reward, run_id, info_string)
        - reward: 1.0 if patch resolves the issue, 0.0 otherwise
        - run_id: Unique ID for this evaluation
        - info_string: Status information from evaluation
    """
    import os

    # SWE-bench Verified is scored by the official swebench harness (not swesmith / eval_server).
    if instance.get("benchmark") == "verified":
        return _compute_score_verified(patch, instance)

    # Check if remote evaluation is configured
    eval_server_url = os.environ.get("EVAL_SERVER_URL")

    if eval_server_url:
        # Use remote evaluation
        try:
            from .eval_client import compute_score_remote

            return compute_score_remote(patch, instance)
        except Exception as e:
            return 0.0, "ERROR", f"Remote evaluation failed: {str(e)}"

    # Use local Docker evaluation
    try:
        from swesmith.harness.eval import run_evaluation
        from swebench.harness.constants import (
            KEY_INSTANCE_ID,
            KEY_MODEL,
            KEY_PREDICTION,
        )

        # Create prediction dict
        pred = {
            KEY_INSTANCE_ID: instance["instance_id"],
            KEY_MODEL: "advisor_model",
            KEY_PREDICTION: patch,
        }

        # Run evaluation using SWE-Smith harness
        # Use a temporary run_id to avoid conflicts
        run_id = f"eval_{uuid.uuid4().hex[:8]}"

        result = run_evaluation(
            pred=pred, instance=instance, run_id=run_id, f2p_only=False, is_gold=False
        )

        # Check if patch resolved the issue
        resolved = result.get("resolved", False)
        status = result.get("status", "unknown")

        reward = 1.0 if resolved else 0.0
        info = f"Status: {status}, Resolved: {resolved}"

        return reward, run_id, info

    except Exception as e:
        return 0.0, "ERROR", f"Evaluation failed: {str(e)}"
