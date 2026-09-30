"""ALFWorld env wrapper (runs in the alfworld-worker venv). No AgentBench dependency —
just the `alfworld` package + TextWorld. One episode per AlfEnv instance.
"""

from __future__ import annotations

import os
from pathlib import Path

# MUST set ALFWORLD_DATA before importing alfworld (it resolves its data dir at import).
# Point this at your ALFWorld game-data download (see docs/DEPENDENCIES.md).
ALFWORLD_DATA = os.environ.setdefault("ALFWORLD_DATA", "data/alfworld_data")

import yaml  # noqa: E402
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv  # noqa: E402

_CONFIG_PATH = str(Path(__file__).resolve().parent / "configs" / "base_config.yaml")


def _expand(o):
    if isinstance(o, str):
        return os.path.expandvars(o)
    if isinstance(o, dict):
        return {k: _expand(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_expand(x) for x in o]
    return o


_CONFIG = _expand(yaml.safe_load(open(_CONFIG_PATH)))


class _SingleAlfredTWEnv(AlfredTWEnv):
    """One-game AlfredTWEnv (mirrors AgentBench's SingleAlfredTWEnv; regen disabled)."""

    def __init__(self, config, game_file, train_eval="eval_out_of_distribution"):
        self.config = config
        self.train_eval = train_eval
        self.goal_desc_human_anns_prob = config["env"]["goal_desc_human_anns_prob"]
        self.get_game_logic()
        self.random_seed = 42
        self.game_files = [game_file]
        self.num_games = 1


class AlfEnv:
    """reset() -> (obs, admissible); step(action) -> (obs, admissible, done, won)."""

    def __init__(self, game_relpath: str):
        self.game_relpath = game_relpath
        gf = os.path.join(ALFWORLD_DATA, game_relpath)
        self._env = _SingleAlfredTWEnv(_CONFIG, gf).init_env(batch_size=1)
        self._done = False
        # Trusted outcome, latched straight from the env on every step. The runner can read this
        # (via final_won()) instead of the harness's self-reported dict, so a policy-authored
        # (patched) harness cannot fake its reward by returning {"won": True} without actually
        # completing the task. Latched (never un-set) — once truly won, stays won.
        self._last_won = False

    def reset(self):
        obs, info = self._env.reset()
        # drop TextWorld's preamble block (AgentBench: '\n'.join(ob.split('\n\n')[1:]))
        obs0 = "\n".join(obs[0].split("\n\n")[1:]) if obs and obs[0] else ""
        admissible = info.get("admissible_commands", [[]])[0]
        self._done = False
        self._last_won = False
        return obs0, admissible

    def step(self, action):
        obs, _scores, dones, info = self._env.step([action])
        won = bool(info.get("won", [False])[0])
        done = bool(dones[0])
        admissible = info.get("admissible_commands", [[]])[0]
        self._done = done
        self._last_won = self._last_won or won
        return (obs[0] if obs else ""), admissible, done, won

    def final_won(self):
        """Whether the episode actually reached the goal (trusted; not the harness's return value)."""
        return self._last_won

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass


def make_scoring_facade(real_env: "AlfEnv"):
    """Return a stand-in env to hand to POLICY-AUTHORED (patched) harness code.

    It forwards reset()/step() to a hidden real AlfEnv (captured in the methods' closure, NOT stored
    as a reachable attribute) and deliberately exposes NO outcome api (no _last_won / final_won).
    The trusted runner reads the reward from the REAL env (`real_env.final_won()`), which the harness
    never has a named reference to — so a patched harness cannot inflate its own reward by setting
    `env._last_won`, replacing `env.step`, or faking its return dict. (This blocks the natural/simple
    reward hacks; a source-scan in the RL path adds defense against deeper reflection.)
    """

    class _ScoringFacade:
        def __init__(self):
            self.game_relpath = real_env.game_relpath

        def reset(self):
            return real_env.reset()

        def step(self, action):
            return real_env.step(action)

        def close(self):
            pass

    return _ScoringFacade()
