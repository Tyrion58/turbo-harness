"""Terminal-Bench-2 Harness-Advisor RL environment (single-turn GRPO) — PATCH-based.

Mirrors the ScienceWorld/SWE harness-advisor method (`turbo_harness/rl/scienceworld_env.py`): the
advisor policy emits a SEARCH/REPLACE patch to the meta-harness WINNER agent's source
(`agents/kira_poll_wait_sonnet.py`, 62.2% train); the patch is applied to a copy of the winner
harness, validated, and the FROZEN student (Sonnet 5 on Vertex) runs the PATCHED harness on ONE TB2
task via harbor. Reward = task pass 0/1.

Reward integrity: the reward is harbor's verifier reward.txt — `tests/test.sh` runs AFTER the agent
phase, inside the task container, and OVERWRITES reward.txt with the real check, so a policy-authored
patch cannot fake its own reward (unlike the SW score object). A source-scan (`reward_safe`) rejects
patches that reference the verifier path (defense-in-depth). Malformed / non-applying / invalid /
unsafe patches earn 0 and skip the expensive rollout; only an explicit NO_PATCH_NEEDED falls back to
the (unpatched) winner harness, with a selective penalty so GRPO does not collapse to always-abstain.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


class TB2HarnessPatchEnv(BaseTextEnv):
    def __init__(self, env_config: Dict[str, Any] = {}, extras: Dict[str, Any] = {}):
        super().__init__()
        self.max_turns = 1
        assert "reward_spec" in extras
        gt_json = extras["reward_spec"].get("ground_truth_json", "")
        self.gt = json.loads(gt_json) if isinstance(gt_json, str) else gt_json
        # Base = the meta-harness WINNER harness (kira_poll_wait_sonnet); the policy patches ITS code.
        self.harness_dir = (
            env_config.get("harness_dir")
            or os.environ.get("TB2_HARNESS")
            or str(REPO / "turbo_harness/terminal_bench")
        )
        self.student_model = env_config.get("student_model") or os.environ.get(
            "TB2_REWARD_STUDENT", "vertex_ai/claude-sonnet-5"
        )
        self.max_steps = int(env_config.get("max_steps", 300))  # harbor per-run turn cap
        # Anti-reward-hack: NO_PATCH_NEEDED runs the (unpatched) winner -> guarantees its reward at
        # ZERO risk, so GRPO exploits it and never explores patches. A SELECTIVE penalty on only the
        # NO_PATCH samples (not a per-instance constant, which advantage-norm would cancel) lowers
        # their within-group advantage -> pushes the policy toward attempting (good) patches.
        self.no_patch_penalty = float(
            env_config.get("no_patch_penalty")
            or os.environ.get("TB2_NO_PATCH_PENALTY", "0.05")
        )
        self._tmp_dir = None

    def _out(self, reward, status, num_edits, raw=0.0):
        return BaseTextEnvStepOutput(
            observations=[],
            reward=float(reward),
            done=True,
            metadata={
                "instance_id": self.gt.get("instance_id", self.gt.get("task", "")),
                "task": self.gt.get("task", ""),
                "num_edits": num_edits,
                "raw_reward": round(float(raw), 4),
                "status": status,
            },
        )

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        from turbo_harness.patch_advisor import _extract_edits
        from turbo_harness.eval import tb2_executor as TBX

        if "</think>" in action:
            action = action.split("</think>", 1)[1].strip()

        edits = _extract_edits(action)
        is_no_patch = "NO_PATCH_NEEDED" in action
        task = self.gt["task"]

        # Build the harness to run in a fresh temp dir (self-contained, harbor-runnable). Reward
        # integrity: malformed / non-applying / invalid / unsafe output earns 0 and skips the
        # expensive rollout; only NO_PATCH_NEEDED runs the (unpatched) winner harness.
        self._tmp_dir = tempfile.mkdtemp(prefix="tb2rl_patch_")
        TBX.make_harness_copy(self._tmp_dir)
        if not edits:
            if not is_no_patch:
                return self._out(0.0, "malformed_output", 0)
        else:
            agent_path = Path(self._tmp_dir) / "agents" / TBX.AGENT_FILE
            content = agent_path.read_text()
            applied = 0
            failed = []
            for i, (search, replace) in enumerate(edits):
                if search in content:
                    content = content.replace(search, replace, 1)
                    applied += 1
                else:
                    failed.append(i)
            if applied == 0:
                return self._out(0.0, "patch_apply_failed", len(edits))
            agent_path.write_text(content)
            ok, msg = TBX.validate_harness(self._tmp_dir)
            if not ok:
                return self._out(0.0, f"patch_invalid:{msg[:60]}", len(edits))
            safe, smsg = TBX.reward_safe(self._tmp_dir)  # defense-in-depth vs reward-hacking patches
            if not safe:
                return self._out(0.0, f"reward_unsafe:{smsg[:50]}", len(edits))

        # Run the (policy-patched) harness on ONE task via harbor + Sonnet. Reward = verifier pass 0/1.
        try:
            reward, status = TBX.eval_harness(
                self._tmp_dir, task, student=self.student_model, max_turns=self.max_steps
            )
        except Exception as e:  # noqa: BLE001
            return self._out(0.0, f"eval_error:{type(e).__name__}", len(edits))

        if status in ("no_result", "harbor_timeout", "harbor_error") or "reward_parse" in status:
            # infra miss / timeout -> 0 (harbor timeout is effectively a task failure here)
            return self._out(0.0, status, len(edits))
        # NO_PATCH (ran the winner): penalize so abstaining is not a free way to recover its reward.
        if not edits and is_no_patch and self.no_patch_penalty > 0:
            reward = max(0.0, reward - self.no_patch_penalty)
            return self._out(reward, f"no_patch_penalized:{status}", 0, raw=reward)
        return self._out(reward, status, len(edits), raw=reward)

    def close(self):
        if self._tmp_dir and os.path.exists(self._tmp_dir):
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
