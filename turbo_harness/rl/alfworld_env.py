"""ALFWorld Harness-Advisor RL environment (single-turn GRPO) — PATCH-based.

Mirrors the SWE harness-advisor method (`turbo_harness/rl/env.py`) and the ScienceWorld patch env
(`turbo_harness/rl/scienceworld_env.py`): the advisor policy emits a SEARCH/REPLACE patch to the
general harness's source (`harness.py`); the patch is applied to a copy of the general harness,
validated, and the frozen student runs the PATCHED harness on ONE ALFWorld game. Reward = the
ALFWorld outcome (won ∈ {0,1}) — a sparse GRPO signal.

Reward integrity (the patched harness is policy-authored CODE, so it must not be able to fake its
own reward): it runs in a SEPARATE process (the alfworld worker venv, via `eval_harness`), and there
it is handed a SCORING FACADE (`make_scoring_facade`) — not the real AlfEnv — so it cannot reach the
outcome; the trusted runner reads the reward from the real env's `final_won()` (`trusted_score=True`),
never from the harness's return dict. A source-scan (`reward_safe`) rejects patches that reference
outcome internals (defense-in-depth vs reflection). Malformed / non-applying / invalid / unsafe
patches earn 0 and skip the expensive rollout; only an explicit NO_PATCH_NEEDED falls back to the
(unpatched) general harness. Reward is 0.0 or 1.0.
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

ALF_DATA = Path(os.environ.get("ALFWORLD_DATA", "data/alfworld_data"))


def _load_games(split: str) -> list[str]:
    """Flatten the split json ({group: [game_relpath, ...]}) into an ordered game list (runner order)."""
    data = json.load(open(ALF_DATA / f"{split}.json"))
    out: list[str] = []
    for v in data.values():
        out.extend(v)
    return out


class ALFWorldHarnessPatchEnv(BaseTextEnv):
    def __init__(self, env_config: Dict[str, Any] = {}, extras: Dict[str, Any] = {}):
        super().__init__()
        self.max_turns = 1
        assert "reward_spec" in extras
        gt_json = extras["reward_spec"].get("ground_truth_json", "")
        self.gt = json.loads(gt_json) if isinstance(gt_json, str) else gt_json
        # Base = the meta-harness BEST GENERAL harness (evolved on train); the policy patches ITS code.
        # Set via env_config or the ALF_HARNESS_DIR env var (mirrors ALF_REWARD_STUDENT) so a run can
        # point at a specific proposer's harness (e.g. the Sonnet general) without a code change.
        self.harness_dir = (
            env_config.get("harness_dir")
            or os.environ.get("ALF_HARNESS_DIR")
            or str(REPO / "artifacts/alfworld/qwen3-5-9b/stage1_fullscaffold/general")
        )
        self.student_model = env_config.get("student_model") or os.environ.get(
            "ALF_REWARD_STUDENT", "Qwen3.5-9B"
        )
        # Each dataset row carries its own split NAME in gt (harness_r1_train for train rows,
        # harness_r1_eval for eval rows) so in-training val eval resolves the RIGHT games; fall back
        # to env_config/default. NOTE: ALFWorld's split is a NAME (not a file path) — the executor
        # reads ALF_DATA/<split>.json.
        self.split = self.gt.get("split") or env_config.get("split", "harness_r1_train")
        self.max_steps = int(env_config.get("max_steps", 50))
        # Anti-reward-hack (same lever as ScienceWorld): NO_PATCH_NEEDED runs the base and, on an
        # already-won game, guarantees reward 1.0 at ZERO risk -> GRPO can collapse to always-NO_PATCH.
        # A selective penalty on the NO_PATCH samples pushes exploration toward real patches. Binary
        # reward here (won in {0,1}) so it only bites on WON games (1 -> 1-penalty); lost games are 0
        # either way, so patching a lost game (win=1) is already strongly rewarded. Default 0 (off);
        # calibrate at the RL stage (binary may want a larger value than ScienceWorld's 0.05).
        self.no_patch_penalty = float(
            env_config.get("no_patch_penalty") or os.environ.get("ALF_NO_PATCH_PENALTY", "0")
        )
        self._tmp_dir = None
        # Cache the split's games so step() can assert gt.index really points at gt.game_relpath —
        # guards against a parquet/split mismatch silently scoring the WRONG game.
        try:
            self._games = _load_games(self.split)
        except (OSError, KeyError, json.JSONDecodeError):
            self._games = None

    def _out(self, reward, status, num_edits, won=False):
        return BaseTextEnvStepOutput(
            observations=[],
            reward=float(reward),
            done=True,
            metadata={
                "instance_id": self.gt.get("instance_id", ""),
                "game_relpath": self.gt.get("game_relpath", ""),
                "num_edits": num_edits,
                "won": bool(won),
                "status": status,
            },
        )

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        from turbo_harness.patch_advisor import _extract_edits, apply_patch
        from turbo_harness.eval import alfworld_executor as ALFX

        if "</think>" in action:
            action = action.split("</think>", 1)[1].strip()

        edits = _extract_edits(action)
        is_no_patch = "NO_PATCH_NEEDED" in action
        idx = int(self.gt["index"])

        # Guard: gt.index must map to gt.game_relpath in the reward split, else we'd score the wrong game.
        if self._games is not None and not (
            0 <= idx < len(self._games)
            and str(self._games[idx]) == str(self.gt.get("game_relpath"))
        ):
            return self._out(0.0, f"instance_mismatch:{idx}", 0)

        # Build the harness to run in a fresh temp dir. Reward integrity: malformed / non-applying /
        # invalid output earns 0 and skips the expensive rollout; only NO_PATCH_NEEDED runs the base.
        self._tmp_dir = tempfile.mkdtemp(prefix="alfrl_patch_")
        if not edits:
            if not is_no_patch:
                return self._out(0.0, "malformed_output", 0)
            shutil.copytree(
                self.harness_dir, self._tmp_dir,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                dirs_exist_ok=True,
            )
        else:
            success, _ = apply_patch(edits, self.harness_dir, self._tmp_dir)
            if not success:
                return self._out(0.0, "patch_apply_failed", len(edits))
            ok, msg = ALFX.validate_harness(self._tmp_dir)
            if not ok:
                return self._out(0.0, f"patch_invalid:{msg[:60]}", len(edits))
            safe, smsg = ALFX.reward_safe(self._tmp_dir)  # defense-in-depth vs reward-hacking patches
            if not safe:
                return self._out(0.0, f"reward_unsafe:{smsg[:50]}", len(edits))

        # Run the (policy-patched) harness in the alfworld worker venv; the outcome comes from the
        # env (trusted_score=True), never from the harness's own return dict.
        try:
            results, statuses, _resolved = ALFX.eval_harness(
                self._tmp_dir, [idx], split=self.split, student=self.student_model,
                trials=1, concurrency=1, max_steps=self.max_steps,
                harness_name=f"rl_{idx}", trusted_score=True, cleanup=True,
            )
        except Exception as e:  # noqa: BLE001
            return self._out(0.0, f"eval_error:{type(e).__name__}", len(edits))

        reward = float(results.get(idx, [0.0])[0])
        status = statuses[0] if statuses else "?"
        if "MISSING" in status:
            return self._out(0.0, status, len(edits))  # infra miss -> 0, not a real signal
        # NO_PATCH (ran the base): penalize so abstaining is not a free way to keep the base outcome.
        if not edits and is_no_patch and self.no_patch_penalty > 0:
            r = max(0.0, reward - self.no_patch_penalty)
            return self._out(r, f"no_patch_penalized:{reward:.0f}", 0, won=reward >= 1.0)
        return self._out(reward, status, len(edits), won=reward >= 1.0)

    def close(self):
        if self._tmp_dir and os.path.exists(self._tmp_dir):
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
