# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the MASA hand, hand_v2_left design.

The following configurations are available:

* :obj:`HAND_V2_LEFT_CONFIG`: left MASA hand (21 actuated joints) with implicit actuators,
  root link fixed in space, palm facing up.

The USD is built from the robot-hand-control-stack URDF by ``scripts/convert_masa_hand.py``.
Rerun it when the URDF changes.

* From the URDF: links, frames, joint axes, 1 Nm effort limit, no joint friction.
* From the stack's ``model.json`` (applied by the env): joint limits and tendon routing.
* Set here: PD gains, rigid-body settings. Link masses come from PhysX (1000 kg/m^3).
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

HAND_V2_LEFT_USD_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "hand_v2_left", "hand_v2_left.usd"
)

HAND_V2_LEFT_SIM_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "hand_v2_left", "model.json"
)
"""Joint limits, frame signs and tendon routing of the real hand (from the stack's export tool)."""

HAND_V2_LEFT_MOTOR_MAX_VELOCITY_DEG_S = 2400.0
"""Motor speed cap (deg/s), from the stack's XC330 motor model.
Used only when ``MasaHandEnvCfg.limit_motor_speed`` is True (off by default)."""

HAND_V2_LEFT_CONFIG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=HAND_V2_LEFT_USD_PATH,
        # VERY IMPORTANT
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # gravity on, like the real hand (fingers sag a little at stiffness 1.0)
            disable_gravity=False,
            retain_accelerations=True,
            enable_gyroscopic_forces=False,
            angular_damping=0.01,
            max_depenetration_velocity=1000.0,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # fingers cannot pass through each other. Overlapping palm-finger pairs are
            # filtered in the USD (scripts/convert_masa_hand.py), or the hand jams.
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
            fix_root_link=True,  # Fix the base in space
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # palm up, fingers along -y, fingers (not palm) under the cube spawn point
        pos=(0.005, -0.225, 0.54),
        rot=(0.0, 0.70711, -0.70711, 0.0),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            # PhysX default, as in Shadow Hand. The URDF's 1 rad/s breaks self-collision.
            velocity_limit_sim=100.0,
            # URDF value, set explicitly
            effort_limit_sim=1.0,
            # Shadow Hand finger gains
            stiffness=1.0,
            damping=0.1,
            # the URDF defines no joint friction
            friction=0.0,
            dynamic_friction=0.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
