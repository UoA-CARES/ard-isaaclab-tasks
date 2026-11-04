# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
MASA Hand cube-repose environment (state observations).

The MASA hand version of ``Isaac-ARD-Repose-Cube-Shadow-Direct-v0``: same task,
env machinery, cube/goal layout and ARD reward contract (``masa_hand_env.py``,
``MasaHandEnv.compute_reward``), with the MASA hand (``hand_v2_left``, see
``hand_v2_left_config.py``) in place of the Shadow Hand. The network size comes from
the original MASA hand IsaacLab project (best sweep run, unit_8).
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-ARD-Repose-Cube-Masa-Direct-v0",
    entry_point=f"{__name__}.masa_hand_env:MasaHandEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.masa_hand_env_cfg:MasaHandEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
