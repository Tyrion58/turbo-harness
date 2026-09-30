"""GRPO training entrypoint for harness patch advisor.

Usage:
    cd SkyRL && uv run --isolated --extra fsdp \
        python -m turbo_harness.rl.main [config overrides...]
"""

import sys

import ray
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils import initialize_ray
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl_gym.envs import register


@ray.remote(num_cpus=1)
def rl_entrypoint(cfg: SkyRLTrainConfig):
    register(
        id="harness_advisor",
        entry_point="turbo_harness.rl.env:HarnessAdvisorEnv",
    )
    register(
        id="alfworld_harness_patch",
        entry_point="turbo_harness.rl.alfworld_env:ALFWorldHarnessPatchEnv",
    )
    register(
        id="tb2_harness_patch",
        entry_point="turbo_harness.rl.tb2_env:TB2HarnessPatchEnv",
    )
    exp = BasePPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(rl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
