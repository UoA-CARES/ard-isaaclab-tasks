# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Humanoid running environment (state observations).

Migration of the official ``Isaac-Humanoid-Direct-v0`` benchmark (IsaacLab
2.3.X, ``isaaclab_tasks.direct.humanoid``). The cfg, agent hyperparameters and
env machinery are copied unchanged; the reward now lives in
``humanoid_env.py`` (``LocomotionEnv.compute_reward``) as the ARD edit target,
and the gym ID is prefixed with ``Isaac-ARD-`` to avoid clashing with the
``isaaclab_tasks`` registration that ``train.py`` also imports.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Isaac-ARD-Humanoid-Direct-v0",
    entry_point=f"{__name__}.humanoid_env:HumanoidEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_env_cfg:HumanoidEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)