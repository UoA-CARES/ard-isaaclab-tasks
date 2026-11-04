# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
MASA Hand OpenAI cube-repose environment (reduced policy observations, full-state critic).

IsaacLab's Shadow Hand OpenAI settings (``Isaac-Repose-Cube-Shadow-OpenAI-FF/LSTM-Direct-v0``)
on the MASA hand. This task has its own env, cfg and agent configs, apart from
``Isaac-ARD-Repose-Cube-Masa-Direct-v0`` (``masa_hand/``). Only the robot definition is shared:
``masa_hand/hand_v2_left_config.py`` and its USD.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-ARD-Repose-Cube-Masa-OpenAI-FF-Direct-v0",
    entry_point=f"{__name__}.masa_hand_openai_env:MasaHandOpenAIEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.masa_hand_openai_env_cfg:MasaHandOpenAIEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
    },
)

gym.register(
    id="Isaac-ARD-Repose-Cube-Masa-OpenAI-LSTM-Direct-v0",
    entry_point=f"{__name__}.masa_hand_openai_env:MasaHandOpenAIEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.masa_hand_openai_env_cfg:MasaHandOpenAIEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)
