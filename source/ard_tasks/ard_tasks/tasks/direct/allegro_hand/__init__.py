# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Allegro Inhand Manipulation environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

inhand_task_entry = "ard_tasks.tasks.direct.inhand_manipulation"

gym.register(
    id="Isaac-ARD-Allegro-Repose-v0",
    entry_point=f"{inhand_task_entry}.inhand_manipulation_env:InHandManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.allegro_hand_env_cfg:AllegroHandEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
