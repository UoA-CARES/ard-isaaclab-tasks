# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Shadow Hand cube-repose environment (state observations).

Migration of the official ``Isaac-Repose-Cube-Shadow-Direct-v0`` benchmark
(IsaacLab 2.3.X, ``isaaclab_tasks.direct.shadow_hand``). The cfg, agent
hyperparameters and env machinery are copied unchanged; the reward now lives in
``shadow_hand_env.py`` (``ShadowHandEnv._get_rewards``) as the ARD edit target,
and the gym ID is prefixed with ``Isaac-ARD-`` to avoid clashing with the
``isaaclab_tasks`` registration that ``train.py`` also imports.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-ARD-Repose-Cube-Shadow-Direct-v0",
    entry_point=f"{__name__}.shadow_hand_env:ShadowHandEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shadow_hand_env_cfg:ShadowHandEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
