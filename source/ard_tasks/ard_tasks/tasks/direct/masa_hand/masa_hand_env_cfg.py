# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import isaaclab.envs.mdp as mdp
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
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

from .hand_v2_left_config import (
    HAND_V2_LEFT_CONFIG,
    HAND_V2_LEFT_MOTOR_MAX_VELOCITY_DEG_S,
    HAND_V2_LEFT_SIM_MODEL_PATH,
)


@configclass
class MasaHandEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    action_space = 21
    observation_space = 152  # (full)
    state_space = 0
    asymmetric_obs = False
    obs_type = "full"

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
    # robot
    robot_cfg: ArticulationCfg = HAND_V2_LEFT_CONFIG
    # real hand's joint limits and tendon routing (robot-hand-control-stack export)
    sim_model_path: str = HAND_V2_LEFT_SIM_MODEL_PATH
    # Optional command speed limit like the real hand's driver (tendon routing + motor speed cap,
    # see MasaHandEnv._apply_action). Off by default, like the Shadow Hand task: finger speed then
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
    # reward scales
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250
    fall_penalty = 0
    fall_dist = 0.24
    vel_obs_scale = 0.2
    success_tolerance = 0.1
    max_consecutive_success = 0
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0


@configclass
class EventCfg:
    """Configuration for randomization (IsaacLab's Shadow Hand OpenAI EventCfg).

    Shadow's ``robot_tendon_properties`` term is left out: the MASA USD has no PhysX tendons
    (its tendon coupling is the routing matrix in MasaHandEnv._apply_action).
    """

    # -- robot
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    robot_joint_pos_limits = EventTerm(
        func=mdp.randomize_joint_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "lower_limit_distribution_params": (0.00, 0.01),
            "upper_limit_distribution_params": (0.00, 0.01),
            "operation": "add",
            "distribution": "gaussian",
        },
    )

    # -- object
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # -- scene
    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )


@configclass
class MasaHandOpenAIEnvCfg(MasaHandEnvCfg):
    """IsaacLab's ShadowHandOpenAIEnvCfg settings on the MASA hand, tilted like the real setup."""

    # env
    decimation = 3
    episode_length_s = 8.0
    action_space = 21
    observation_space = 43  # (openai: fingertip pos 15 + object pos 3 + relative rot 4 + actions 21)
    state_space = 182  # (full obs 152 + fingertip forces 30)
    asymmetric_obs = True
    obs_type = "openai"
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
    )
    # robot: tilted 20 deg forward with the fingertips down, like the real hand's setup
    # (the base rotation (0, 0.70711, -0.70711, 0) turned +20 deg about world x)
    robot_cfg: ArticulationCfg = HAND_V2_LEFT_CONFIG.replace(
        init_state=HAND_V2_LEFT_CONFIG.init_state.replace(rot=(-0.12279, 0.69636, -0.69636, -0.12279))
    )
    # the real hand's motor speed limit (HAND_V2_LEFT_MOTOR_MAX_VELOCITY_DEG_S, measured)
    limit_motor_speed: bool = True
    # reset
    reset_position_noise = 0.01  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # reward scales
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250
    fall_penalty = -50
    fall_dist = 0.24
    vel_obs_scale = 0.2
    success_tolerance = 0.4
    max_consecutive_success = 50
    av_factor = 0.1
    act_moving_average = 0.3
    force_torque_obs_scale = 10.0
    # domain randomization config
    events: EventCfg = EventCfg()
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    )
