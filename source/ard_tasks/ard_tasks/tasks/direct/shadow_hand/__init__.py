# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Shadow Hand environments.

Verbatim migration of the official ``Isaac-Repose-Cube-Shadow-Direct-v0`` (state-based)
and ``Isaac-Repose-Cube-Shadow-Vision-Direct-v0`` benchmarks (IsaacLab 2.3.X,
``isaaclab_tasks.direct.shadow_hand``). The env, cfg, feature extractor and agent
hyperparameters are copied unchanged; only the gym registration ID is prefixed with
``Isaac-ARD-`` to avoid clashing with the ``isaaclab_tasks`` registration (which
``train.py`` also imports).
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

inhand_task_entry = "ard_tasks.tasks.direct.inhand_manipulation"

# ------------
# State-based
# ------------

gym.register(
    id="Isaac-ARD-Repose-Cube-Shadow-Direct-v0",
    entry_point=f"{inhand_task_entry}.inhand_manipulation_env:InHandManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shadow_hand_env_cfg:ShadowHandEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# -------
# Vision
# -------

gym.register(
    id="Isaac-ARD-Repose-Cube-Shadow-Vision-Direct-v0",
    entry_point=f"{__name__}.shadow_hand_vision_env:ShadowHandVisionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shadow_hand_vision_env:ShadowHandVisionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandVisionFFPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_vision_cfg.yaml",
    },
)

gym.register(
    id="Isaac-ARD-Repose-Cube-Shadow-Vision-Direct-Play-v0",
    entry_point=f"{__name__}.shadow_hand_vision_env:ShadowHandVisionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shadow_hand_vision_env:ShadowHandVisionEnvPlayCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandVisionFFPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_vision_cfg.yaml",
    },
)
