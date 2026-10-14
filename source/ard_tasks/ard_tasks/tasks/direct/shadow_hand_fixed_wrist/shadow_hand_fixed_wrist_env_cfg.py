# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.shadow_hand.shadow_hand_env_cfg import EventCfg as ShadowEventCfg
from isaaclab_tasks.direct.shadow_hand.shadow_hand_env_cfg import ShadowHandOpenAIEnvCfg

from ..masa_hand.hand_v2_left_config import HAND_V2_LEFT_MOTOR_MAX_VELOCITY_DEG_S


@configclass
class EventCfg(ShadowEventCfg):
    """IsaacLab's Shadow OpenAI randomization, with no joint-limit noise on the locked wrist.

    The limit noise (std 0.01 rad) is larger than the locked range, so it could put a lower
    limit above the upper one, which IsaacLab stops with an error.
    """

    robot_joint_pos_limits = EventTerm(
        func=mdp.randomize_joint_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="robot0_(FF|MF|RF|LF|TH)J.*"),
            "lower_limit_distribution_params": (0.00, 0.01),
            "upper_limit_distribution_params": (0.00, 0.01),
            "operation": "add",
            "distribution": "gaussian",
        },
    )


@configclass
class ShadowHandOpenAIFixedWristEnvCfg(ShadowHandOpenAIEnvCfg):
    """IsaacLab's ShadowHandOpenAIEnvCfg with the wrist locked and the MASA hand's motor speed cap."""

    # env: the policy drives the 18 finger joints only
    action_space = 18
    observation_space = 40  # (openai: fingertip pos 15 + object pos 3 + relative rot 4 + actions 18)
    state_space = 185  # (full obs 155 + fingertip forces 30)
    actuated_joint_names = [
        "robot0_FFJ3",
        "robot0_FFJ2",
        "robot0_FFJ1",
        "robot0_MFJ3",
        "robot0_MFJ2",
        "robot0_MFJ1",
        "robot0_RFJ3",
        "robot0_RFJ2",
        "robot0_RFJ1",
        "robot0_LFJ4",
        "robot0_LFJ3",
        "robot0_LFJ2",
        "robot0_LFJ1",
        "robot0_THJ4",
        "robot0_THJ3",
        "robot0_THJ2",
        "robot0_THJ1",
        "robot0_THJ0",
    ]
    # the wrist (the MASA hand has none): locked to +-fixed_joint_range (rad) around 0
    fixed_joint_names = ["robot0_WRJ1", "robot0_WRJ0"]
    fixed_joint_range = 0.001
    # the MASA hand's motor speed cap (HAND_V2_LEFT_MOTOR_MAX_VELOCITY_DEG_S, measured). Shadow has no
    # tendon routing matrix, so each actuated joint counts as one motor at 1:1.
    limit_motor_speed: bool = True
    motor_max_velocity_deg_s: float = HAND_V2_LEFT_MOTOR_MAX_VELOCITY_DEG_S
    # domain randomization config
    events: EventCfg = EventCfg()
