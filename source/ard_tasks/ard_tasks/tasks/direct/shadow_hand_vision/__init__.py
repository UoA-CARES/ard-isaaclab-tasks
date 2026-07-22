# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Shadow Hand cube-repose environment (vision: TiledCamera + CNN feature extractor).

Migration of the official ``Isaac-Repose-Cube-Shadow-Vision-Direct-v0`` benchmark
(IsaacLab 2.3.X, ``isaaclab_tasks.direct.shadow_hand``). The cfg, feature
extractor, agent hyperparameters and env machinery are copied unchanged; the
reward now lives in ``shadow_hand_vision_env.py``
(``ShadowHandVisionEnv._get_rewards``) as the ARD edit target, and the gym ID is
prefixed with ``Isaac-ARD-`` to avoid clashing with the ``isaaclab_tasks``
registration that ``train.py`` also imports.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

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
