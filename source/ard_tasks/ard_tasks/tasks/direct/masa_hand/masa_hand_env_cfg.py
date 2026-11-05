# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from ard_tasks.utils.events import randomize_rigid_body_material_cached

from .hand_v2_left_config import (
    HAND_V2_LEFT_CONFIG,
    HAND_V2_LEFT_MOTOR_MAX_VELOCITY_DEG_S,
    HAND_V2_LEFT_SIM_MODEL_PATH,
)


@configclass
class EventCfg:
    """Hand friction by material, like the real hand and the OpenAI variant: rubber-like pads on the
    palm and phalanges, smooth 3D-printed plastic everywhere else.

    Restitution stays 0, this task's value before friction was added. The OpenAI variant uses 1, from Shadow.
    """

    # Friction ranges are estimates until measured on the real hand (slide angle: static = tan(angle)).
    # PhysX averages the two materials in a contact, so hand-cube friction = (hand + cube) / 2.
    robot_rubber_material = EventTerm(
        func=randomize_rigid_body_material_cached,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["palm_.*", ".*_(pp|ip|dp)"]),
            "static_friction_range": (1.0, 1.5),
            "dynamic_friction_range": (0.8, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
            "make_consistent": True,
        },
    )
    robot_plastic_material = EventTerm(
        func=randomize_rigid_body_material_cached,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_mcp", "arm_.*"]),
            "static_friction_range": (0.25, 0.45),
            "dynamic_friction_range": (0.2, 0.35),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
            "make_consistent": True,
        },
    )


@configclass
class MasaHandEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    action_space = 21
    observation_space = 152  # (full)
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
        ),
    )
    # robot: tilted 20 deg forward with the fingertips down, same as the OpenAI variant
    # (the base rotation (0, 0.70711, -0.70711, 0) turned +20 deg about world x)
    robot_cfg: ArticulationCfg = HAND_V2_LEFT_CONFIG.replace(
        init_state=HAND_V2_LEFT_CONFIG.init_state.replace(rot=(-0.12279, 0.69636, -0.69636, -0.12279))
    )
    # real hand's joint limits (robot-hand-control-stack export)
    sim_model_path: str = HAND_V2_LEFT_SIM_MODEL_PATH
    # Optional command speed limit like the real hand's driver (the motor speed cap, applied per
    # joint, see MasaHandEnv._apply_action). Off by default, like the Shadow Hand task: finger speed then
    # comes only from the PD drives and the link masses. Turn it on with `env.limit_motor_speed=True`
    # on train.py; the cap can be overridden the same way (`env.motor_max_velocity_deg_s=240.0`).
    limit_motor_speed: bool = False
    motor_max_velocity_deg_s: float = HAND_V2_LEFT_MOTOR_MAX_VELOCITY_DEG_S
    actuated_joint_names = [
        "palm_abd_add",
        "index_mcp1",
        "index_mcp2",
        "index_pip",
        "index_dip",
        "middle_mcp1",
        "middle_mcp2",
        "middle_pip",
        "middle_dip",
        "ring_mcp1",
        "ring_mcp2",
        "ring_pip",
        "ring_dip",
        "little_mcp1",
        "little_mcp2",
        "little_pip",
        "little_dip",
        "thumb_mcp1",
        "thumb_mcp2",
        "thumb_pip",
        "thumb_dip",
    ]
    fingertip_body_names = [
        "index_dp",
        "middle_dp",
        "ring_dp",
        "little_dp",
        "thumb_dp",
    ]

    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
            semantic_tags=[("class", "cube")],
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.39, 0.6), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.0, 1.0, 1.0),
            )
        },
    )
    # scene
    # clone_in_fabric off (IsaacLab's default; Shadow uses on): with this hand's USD, the
    # Fabric copies of env_0 get no mesh points, so every env but env_0 is invisible in the
    # viewer (physics is unaffected). Scene creation at 8192 envs took 17 s vs 18 s with it on.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192, env_spacing=0.75, replicate_physics=True, clone_in_fabric=False
    )

    # Cameras
    enable_cameras: bool = False

    # reset
    reset_position_noise = 0.01  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # joints reset exactly to the default pose (no noise): they move fingers sideways, and
    # neighbouring fingers are only 2.1 mm apart, so noise there starts fingers interlocked
    reset_noise_free_joint_names = ["palm_abd_add", "thumb_mcp1", "index_mcp1", "middle_mcp1", "ring_mcp1", "little_mcp1"]

    # task / observation settings
    vel_obs_scale = 0.2
    act_moving_average = 1.0
    fall_dist = 0.24
    success_tolerance = 0.1
    max_consecutive_success = 0
    av_factor = 0.1

    # domain randomization config
    events: EventCfg = EventCfg()
