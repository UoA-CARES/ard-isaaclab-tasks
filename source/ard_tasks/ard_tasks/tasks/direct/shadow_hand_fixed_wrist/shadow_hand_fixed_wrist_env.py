# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IsaacLab's Shadow Hand cube-repose env (``InHandManipulationEnv``) with locked joints and a motor speed cap.

Only ``__init__`` (lock the wrist) and ``_apply_action`` (speed cap) change; the reward,
observations and resets are IsaacLab's.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import saturate

from isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env import InHandManipulationEnv, scale

if TYPE_CHECKING:
    from .shadow_hand_fixed_wrist_env_cfg import ShadowHandOpenAIFixedWristEnvCfg


class ShadowHandFixedWristEnv(InHandManipulationEnv):
    cfg: ShadowHandOpenAIFixedWristEnvCfg

    def __init__(self, cfg: ShadowHandOpenAIFixedWristEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # lock the fixed joints (not actuated): shrink their range to +-fixed_joint_range. Also into the
        # defaults, which IsaacLab's joint-limit randomization starts from at every reset.
        fixed_ids = [self.hand.joint_names.index(name) for name in self.cfg.fixed_joint_names]
        self.hand_dof_lower_limits[:, fixed_ids] = -self.cfg.fixed_joint_range
        self.hand_dof_upper_limits[:, fixed_ids] = self.cfg.fixed_joint_range
        fixed_limits = torch.stack(
            (self.hand_dof_lower_limits[:, fixed_ids], self.hand_dof_upper_limits[:, fixed_ids]), dim=-1
        )
        self.hand.write_joint_position_limit_to_sim(fixed_limits, joint_ids=fixed_ids)
        self.hand.data.default_joint_pos_limits[:, fixed_ids] = fixed_limits

        # motor speed cap as a target step per physics step
        self.motor_max_step = (
            math.radians(self.cfg.motor_max_velocity_deg_s) * self.physics_dt if self.cfg.limit_motor_speed else None
        )

    def _apply_action(self) -> None:
        self.cur_targets[:, self.actuated_dof_indices] = scale(
            self.actions,
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        self.cur_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.cur_targets[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
        )
        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            self.cur_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        # Same rule as the MASA hand task (robot_hand VelocityGuard): scale the whole target step down
        # by one factor so no motor moves faster than its speed cap. Here one motor per joint, at 1:1.
        if self.motor_max_step is not None:
            step = self.cur_targets[:, self.actuated_dof_indices] - self.prev_targets[:, self.actuated_dof_indices]
            overshoot = step.abs().amax(dim=-1) / self.motor_max_step
            self.cur_targets[:, self.actuated_dof_indices] = (
                self.prev_targets[:, self.actuated_dof_indices] + step / torch.clamp(overshoot, min=1.0).unsqueeze(-1)
            )

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        self.hand.set_joint_position_target(
            self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )
