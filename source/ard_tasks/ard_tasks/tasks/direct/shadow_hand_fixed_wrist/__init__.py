# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Shadow Hand OpenAI cube-repose environment with the wrist locked and a motor speed cap.

IsaacLab's ``Isaac-Repose-Cube-Shadow-OpenAI-FF/LSTM-Direct-v0`` (same env code, reward,
randomization and agent configs), changed only where the MASA hand differs: the MASA hand
has no wrist, so the two wrist joints are locked and left out of the actions, and its
measured motor speed cap is applied to every actuated joint. For a fair comparison with
``Isaac-ARD-Repose-Cube-Masa-OpenAI-FF/LSTM-Direct-v0``.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-ARD-Repose-Cube-Shadow-OpenAI-FixedWrist-FF-Direct-v0",
    entry_point=f"{__name__}.shadow_hand_fixed_wrist_env:ShadowHandFixedWristEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shadow_hand_fixed_wrist_env_cfg:ShadowHandOpenAIFixedWristEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
    },
)

gym.register(
    id="Isaac-ARD-Repose-Cube-Shadow-OpenAI-FixedWrist-LSTM-Direct-v0",
    entry_point=f"{__name__}.shadow_hand_fixed_wrist_env:ShadowHandFixedWristEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shadow_hand_fixed_wrist_env_cfg:ShadowHandOpenAIFixedWristEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)
