"""Harness Advisor RL environment.

Single-turn: advisor generates a SEARCH/REPLACE patch → patch is applied
to the general harness → student agent runs with patched harness → reward.

Reward is shaped: 0.0 for fail, 0.5 + 0.5*(MAX_STEPS - steps)/MAX_STEPS for pass.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

REPO = Path(__file__).resolve().parents[2]

MAX_STEPS = 40
LENGTH_COEFFICIENT = 0.5


class HarnessAdvisorEnv(BaseTextEnv):

    def __init__(
        self,
        env_config: Dict[str, Any] = {},
        extras: Dict[str, Any] = {},
    ):
        super().__init__()
        self.max_turns = 1

        assert "reward_spec" in extras
        gt_json = extras["reward_spec"].get("ground_truth_json", "")
        if isinstance(gt_json, str):
            self.gt = json.loads(gt_json)
        else:
            self.gt = gt_json

        # Set via env_config or the SWE_HARNESS_DIR env var (mirrors the other domains' *_HARNESS_DIR)
        # so a run can point the reward env at a different general harness (e.g. the all-Sonnet re-run)
        # without touching code. The patch advisor's prompt already bakes this general's source (via
        # build_rl_dataset --harness-dir), so the two MUST match.
        self.harness_dir = (
            env_config.get("harness_dir")
            or os.environ.get("SWE_HARNESS_DIR")
            or str(REPO / "artifacts/swe_smith/haiku/general")
        )
        # student also env-overridable (mirror SWE_HARNESS_DIR) so a run can swap the frozen student
        # (e.g. Gemini-3.7-Flash) without touching the training config.
        self.student_model = (
            env_config.get("student_model")
            or os.environ.get("SWE_STUDENT_MODEL")
            or "vertex_ai/claude-haiku-4-5"
        )
        self.step_limit = int(env_config.get("step_limit", MAX_STEPS))
        self.cost_limit = float(env_config.get("cost_limit", 3.0))
        self.repo = env_config.get("repo", "multi_repo")
        self._tmp_dir = None

    def _out(self, reward, status, num_edits, resolved=False, steps=None, extra=None):
        meta = {
            "instance_id": self.gt.get("instance_id", ""),
            "resolved": resolved,
            "steps": steps if steps is not None else self.step_limit,
            "status": status,
            "num_edits": num_edits,
        }
        if extra:
            meta.update(extra)
        return BaseTextEnvStepOutput(
            observations=[], reward=reward, done=True, metadata=meta
        )

    def step(self, action: str) -> BaseTextEnvStepOutput:
        import json
        import os
        import signal
        import subprocess
        import sys

        sys.path.insert(0, str(REPO))
        sys.path.insert(0, str(REPO / "turbo_harness/infra/mini-swe-agent/src"))

        from turbo_harness.patch_advisor import _extract_edits, apply_patch
        from turbo_harness.proposer import validate_artifact
        from turbo_harness.infra.scoring import compute_score  # trusted parent-process scoring

        self.turns += 1

        if "</think>" in action:
            action = action.split("</think>", 1)[1].strip()

        edits = _extract_edits(action)
        is_no_patch = "NO_PATCH_NEEDED" in action

        # Reward integrity: malformed / non-applying / invalid output earns 0 and skips the
        # expensive rollout. Only a legitimate NO_PATCH_NEEDED falls back to the base harness.
        self._tmp_dir = tempfile.mkdtemp(prefix="rl_harness_")
        harness_path = self._tmp_dir

        if not edits:
            if not is_no_patch:
                return self._out(0.0, "malformed_output", 0)
            shutil.copytree(
                self.harness_dir, harness_path,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                dirs_exist_ok=True,
            )
        else:
            success, _ = apply_patch(edits, self.harness_dir, harness_path)
            if not success:
                return self._out(0.0, "patch_apply_failed", len(edits))
            ok, _ = validate_artifact(harness_path)
            if not ok:
                return self._out(0.0, "patch_invalid", len(edits))

        # Isolation + infra robustness: run the rollout (which execs policy-influenced harness
        # code) in a SEPARATE process that never scores; retry transient infra failures; the
        # trusted parent scores the returned patch so harness code cannot tamper with reward.
        payload = json.dumps({
            "gt": self.gt,
            "student_model": self.student_model,
            "harness": harness_path,
            "repo": self.repo,
            "step_limit": self.step_limit,
            "cost_limit": self.cost_limit,
        })
        INFRA_MARKERS = ("ERROR:", "SCORE_ERROR", "git_fetch_failed", "git_checkout_failed")
        roll, last_status = None, "rollout_failed"
        rollout_timeout = int(os.environ.get("RL_ROLLOUT_TIMEOUT", "1200"))
        for _ in range(3):
            # Run the rollout in its OWN process group (start_new_session) so a wedged rollout
            # can be hard-killed WHOLE. Plain subprocess.run(timeout=) is not enough: communicate()
            # blocks PAST its timeout while an orphaned `docker run` grandchild keeps the stdout
            # pipe open — a single stuck student rollout once froze the entire run for >1h.
            proc = subprocess.Popen(
                [sys.executable, "-m", "turbo_harness.rl._rollout_worker"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, cwd=str(REPO), start_new_session=True,
            )
            try:
                stdout, _ = proc.communicate(input=payload, timeout=rollout_timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)  # kill child + docker CLI
                except Exception:
                    proc.kill()
                try:
                    proc.communicate(timeout=30)
                except Exception:
                    pass
                last_status = "ERROR:rollout_timeout"
                break  # a wedged rollout will just re-hang -> treat as failed (reward 0)
            except Exception as e:
                last_status = f"ERROR:{type(e).__name__}"
                continue
            lines = [ln for ln in stdout.splitlines() if ln.startswith("ROLLOUT_JSON:")]
            if lines:
                roll = json.loads(lines[-1][len("ROLLOUT_JSON:"):])
                last_status = roll.get("status", "")
                if not any(m in last_status for m in INFRA_MARKERS):
                    break  # clean rollout
            else:
                last_status = "rollout_no_output"
            # else: transient infra failure -> retry

        if roll is None or any(m in last_status for m in INFRA_MARKERS):
            # infra/scoring failure is NOT a genuine unsolved task -> zero, flagged
            return self._out(0.0, last_status or "infra_error", len(edits),
                             extra={"infra_error": True})

        steps = roll.get("steps", self.step_limit)
        patch = roll.get("patch", "")

        reward_score, run_id, info = compute_score(patch, self.gt)
        if run_id == "ERROR":
            return self._out(0.0, f"SCORE_ERROR:{info}", len(edits), steps=steps,
                             extra={"infra_error": True})
        resolved = reward_score >= 1.0

        if resolved:
            reward = (1 - LENGTH_COEFFICIENT) + LENGTH_COEFFICIENT * (
                self.step_limit - steps
            ) / self.step_limit
        else:
            reward = 0.0

        return self._out(reward, roll.get("status", ""), len(edits),
                         resolved=resolved, steps=steps)

    def close(self):
        if self._tmp_dir and os.path.exists(self._tmp_dir):
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
