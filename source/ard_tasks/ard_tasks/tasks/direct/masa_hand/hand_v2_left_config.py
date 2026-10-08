# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the MASA hand, hand_v2_left design.

The following configurations are available:

* :obj:`HAND_V2_LEFT_CONFIG`: left MASA hand (21 actuated joints) with implicit actuators,
  root link fixed in space, palm facing up.

The hand is also a digital twin, so the USD (``assets/hand_v2_left/``) is built from the
URDF in robot-hand-control-stack (``robot_hand/models/hand_v2_left/hand_v2_left.urdf``) by
``scripts/convert_masa_hand.py``, which also checks the USD against the URDF (link frames,
joint limits, effort and velocity limits). Rebuild it with that script whenever the URDF changes.

What comes from the URDF: link names and frames, joint axes, effort limit (1 Nm, also
set explicitly below), and no joint friction. What comes from the control stack
(``model.json``, applied by the env): the joint limits, which differ from the URDF on
ring_mcp1, little_mcp1, middle_mcp1 and thumb_pip, and an optional command speed limit
(off by default), through the motor speed cap and the tendon routing. What neither
defines and is set here instead: PD gains (from the v1 MASA hand config), rigid-body
settings, and link masses (computed by PhysX from collider volume at 1000 kg/m^3).
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
"""Joint limits, frame signs and tendon routing of the real hand, written by the stack's
``tools/export_sim_model.py`` (``scripts/convert_masa_hand.py --robot_config left_hand_v2.yaml``)."""

HAND_V2_LEFT_MOTOR_MAX_VELOCITY_DEG_S = 1200.0
"""Speed cap of every motor in actuator-space deg/s: ``max_velocity_deg_s`` in the stack's
``config/actuators/motor_models/XC330.yaml`` times ``output_ratio`` (1.0; ``left_hand_v2_motors.yaml``
sets no other value). The real driver scales each command step so no motor exceeds it.
Used only when ``MasaHandEnvCfg.limit_motor_speed`` is True; it is off by default, like
the Shadow Hand task, which has no speed limit in sim. With the stack's 240 a full fist takes
about 1.17 s (each finger's shared extensor motor turns 280 deg), against about 0.3 s for the
Shadow Hand. The stack calls 240 a conservative cap (hardware maximum 316)."""

HAND_V2_LEFT_CONFIG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=HAND_V2_LEFT_USD_PATH,
        # VERY IMPORTANT
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # gravity on, as on the real hand (Shadow and Allegro turn it off). With stiffness 1.0
            # the fingers sag a little: random-target tracking error 0.061 -> 0.066 rad, most on
            # palm_abd_add (0.09 -> 0.14 rad), which carries the ring and little fingers
            disable_gravity=False,
            retain_accelerations=True,
            enable_gyroscopic_forces=False,
            angular_damping=0.01,
            max_depenetration_velocity=1000.0,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # On, so a policy cannot move fingers through each other (unsafe on the real hand).
            # The palm-to-finger-base pairs whose convex hulls overlap at the open pose are
            # filtered in the USD (see scripts/convert_masa_hand.py); without that filter the
            # hand jams itself. Every finger-to-finger pair collides.
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
            fix_root_link=True,  # Fix the base in space
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # In the USD the fingers point along +x and the palm faces -z. This rotation
        # (180 deg about (1, -1, 0)) turns the palm up and points the fingers along -y,
        # like the Shadow Hand, and the position puts the palm under the cube's spawn
        # point (0.0, -0.39, 0.6), the same spot as in the Shadow Hand task.
        pos=(0.005, -0.315, 0.54),
        rot=(0.0, 0.70711, -0.70711, 0.0),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            # 100 rad/s, the Shadow Hand USD's value (PhysX default): effectively no joint speed
            # limit. The URDF's 1 rad/s, applied as a hard PhysX clamp, fights the contact solver
            # once self-collision is on (joints were pushed radians past their limits). A command
            # speed limit like the real driver is optional: MasaHandEnvCfg.limit_motor_speed.
            velocity_limit_sim=100.0,
            # 1 Nm, the URDF value; set explicitly so a URDF change cannot alter it silently
            effort_limit_sim=1.0,
            # Shadow Hand finger values (isaaclab_assets shadow_hand.py). The real hand's gains
            # are in mA/deg (stack controllers); converting them needs the XC330 torque constant.
            stiffness=1.0,
            damping=0.1,
            # the URDF defines no joint friction
            friction=0.0,
            dynamic_friction=0.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
