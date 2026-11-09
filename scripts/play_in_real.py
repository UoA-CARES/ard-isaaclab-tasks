# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play an RL-Games checkpoint in simulation and stream its joint positions to the real MASA hand.

This is scripts/play.py plus the streaming. It is streaming, not deployment: the policy runs in the
sim and reads only sim state. After every physics step (60 Hz), the actual joint positions of env 0
also go to the real hand over ROS 2 as its targets, on the same topics as scripts/real_hand_pose.py.
The real hand's measured pose never goes back into the policy.

On the robot computer: the stack running, controller started, a control law other than compliant,
torque on, plus the URDF frame adapter:

    ros2 run robot_hand urdf_target_adapter --ros-args -p config_file_path:=<robot.yaml>

On this computer, in the Isaac conda env. Do not run `jazzy`: it leaves the conda env, and the system
rclpy is built for Python 3.12. This script uses the rclpy that Isaac ships for Python 3.11.

    BRIDGE=$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge
    export ROS_DISTRO=jazzy RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    export ROS_DISCOVERY_SERVER=172.24.192.43:11811 ROS_SUPER_CLIENT=TRUE ROS_DOMAIN_ID=42
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$BRIDGE/jazzy/lib
    export PYTHONPATH=$PWD/source/ard_tasks:$BRIDGE/jazzy/rclpy

    python scripts/play_in_real.py --task Isaac-ARD-Repose-Cube-Masa-OpenAI-FF-Direct-v0 --checkpoint <.pth>

After a y/N prompt the hand follows the sim. This script has no speed cap, so the first move and each
episode reset run at the speed the stack's driver allows. The script always runs one env in real time.
On any problem (torque off, controller stopped, ...) it stops sending for the rest of the run and the
sim keeps playing. The stack holds the last target.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import faulthandler

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL-Games checkpoint and stream its joint positions to the real hand.")
parser.add_argument("--video", action="store_true", default=False, help="Record multi camera views during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
    # The task cameras record env 0 only, and their reset crashes with more than one env
    # (CUDA "index out of bounds" in Camera.reset), so video always runs a single env.
    if args_cli.num_envs != 1:
        print(f"[WARN] --video records env 0 only and needs --num_envs 1; using 1 instead of {args_cli.num_envs}.")
        args_cli.num_envs = 1

# LD_LIBRARY_PATH is read only when the process starts, so check it here, before the long Isaac launch.
if not (
    "isaacsim.ros2.bridge/jazzy/lib" in os.environ.get("LD_LIBRARY_PATH", "")
    and "isaacsim.ros2.bridge/jazzy/rclpy" in os.environ.get("PYTHONPATH", "")
):
    sys.exit("Isaac's ROS 2 libraries are not set up: run the export lines at the top of scripts/play_in_real.py first.")
# The real hand follows one env, at the sim's real speed: faster than real time would rush the hand.
args_cli.real_time = True
if args_cli.num_envs not in (None, 1):
    print(f"[WARN] The real hand follows one env; using 1 instead of {args_cli.num_envs}.")
args_cli.num_envs = 1

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import math
import random
import time

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import ard_tasks.tasks  # noqa: F401
from multi_video_wrapper import MultiCameraRecordVideo
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.signals import SignalHandlerOptions
from real_hand_pose import (
    CONTROLLER_TOPIC,
    ROBOT_YAML,
    STATE_TOPIC,
    TARGET_TOPIC,
    TORQUE_TOPIC,
    WAIT_S,
    load_limits,
)
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
# PLACEHOLDER: Extension template (do not remove this comment)

# In GUI mode, Isaac Sim's omni.kit.pip_archive extension loads numpy 1.26, which cannot unpickle
# checkpoints saved with numpy 2 ("No module named 'numpy._core'"). Map the numpy 2 names to numpy 1.
try:
    import numpy._core  # noqa: F401
except ModuleNotFoundError:
    import numpy

    sys.modules["numpy._core"] = numpy.core
    sys.modules["numpy._core.multiarray"] = numpy.core.multiarray

CONTROLLER_STALE_S = 1.0  # no controller message for this long counts as a stopped controller
SPINS_PER_SEND = 2  # ROS messages read per send; enough to keep the controller and torque status fresh


def add_joint_pos_probe(base_env):
    """Print once per second how often hand.data.joint_pos changes, checked at every physics step."""
    # env.step calls _apply_action just before each physics step. A check in the play loop runs only
    # once per env.step, so it could never read above the policy rate.
    sim_apply_action = base_env._apply_action
    last_pos = base_env.hand.data.joint_pos.clone()
    changes, steps, start = 0, 0, time.monotonic()

    def apply_action_and_probe():
        nonlocal last_pos, changes, steps, start
        sim_apply_action()
        pos = base_env.hand.data.joint_pos
        steps += 1
        if not torch.equal(pos, last_pos):
            changes += 1
            last_pos = pos.clone()
        elapsed = time.monotonic() - start
        if elapsed >= 1.0:
            print(
                f"[PROBE] joint_pos updates: {changes / elapsed:.1f} Hz real time, changed in {changes}/{steps}"
                f" physics steps (sim rate: {1 / base_env.physics_dt:.1f} Hz)"
            )
            changes, steps, start = 0, 0, time.monotonic()

    base_env._apply_action = apply_action_and_probe



class RealHandStream:
    """Publish one env's joint positions (radians, URDF names) to the real hand, as its targets.

    The checks from scripts/real_hand_pose.py run before the first move and at every step. A target
    sent while the controller is stopped, the torque is off or the law ignores targets is stored,
    and the hand jumps to it later when that changes. So after any problem this class stops sending
    for the rest of the run and never resumes by itself.
    """

    def __init__(self, names: list[str]):
        limits = load_limits(ROBOT_YAML)
        unknown = [n for n in names if n not in limits]
        if unknown:
            sys.exit(f"Nothing sent: these sim joints are not on the real hand: {unknown}")
        self.names = names
        self.lower = [math.radians(limits[n][0]) for n in names]
        self.upper = [math.radians(limits[n][1]) for n in names]
        self.active = False  # True from the end of start() until the first problem

        if not os.environ.get("ROS_DISCOVERY_SERVER"):
            print("[WARN] ROS_DISCOVERY_SERVER is not set. Run the export lines if the hand is on another computer.")
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)  # Isaac keeps its own Ctrl+C handling
        self.node = Node("real_hand_stream")
        self.measured: dict[str, float] = {}  # URDF name -> rad
        self.status: dict = {}  # "law": active control law, "law_time": when last seen, "torque": driver state
        self.state_sub = self.node.create_subscription(
            JointState, STATE_TOPIC, lambda m: self.measured.update(zip(m.name, m.position)), 10
        )
        self.node.create_subscription(
            JointState, CONTROLLER_TOPIC, lambda m: self.status.update(law=m.header.frame_id, law_time=time.monotonic()), 10
        )
        self.node.create_subscription(
            Bool, TORQUE_TOPIC, lambda m: self.status.update(torque=m.data),
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.pub = self.node.create_publisher(JointState, TARGET_TOPIC, 10)

    def start(self, targets: list[float]) -> None:
        """Check the hand and ask, while the sim waits. The first target sent after this moves the hand."""
        deadline = time.monotonic() + WAIT_S
        while self._problems() and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        if found := self._problems():
            self._stop("Not ready, nothing sent", found)
            return

        targets = self._clamp(targets)
        print(f"{'joint':14s} {'now':>8s} {'target':>8s} {'move':>8s}   (deg, URDF frame)")
        for name, target in zip(self.names, targets):
            now, deg = math.degrees(self.measured[name]), math.degrees(target)
            print(f"{name:14s} {now:8.1f} {deg:8.1f} {deg - now:+8.1f}")
        answer = input("The hand goes to this pose at once, with no speed cap from this script. Stream the policy? [y/N] ")
        if answer.strip().lower() != "y":
            self._stop("Cancelled, nothing sent", [])
            return

        self._spin_until(time.monotonic() + 0.2)  # refresh the status after the prompt
        if found := self._problems():
            self._stop("Not ready, nothing sent", found)
            return

        self.node.destroy_subscription(self.state_sub)  # not needed from here; saves reading it at every send
        self.active = True
        print("[real hand] Streaming.")

    def publish(self, targets: list[float]) -> None:
        """Send one target from the sim to the real hand, unless streaming has stopped."""
        if not self.active:
            return
        for _ in range(SPINS_PER_SEND):
            rclpy.spin_once(self.node, timeout_sec=0)
        found = self._problems()
        if not all(map(math.isfinite, targets)):
            found.append(f"the sim gave a target that is not a finite number: {targets}")
        if found:
            self._stop("Stopped streaming", found)
            return
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = self.names
        msg.position = self._clamp(targets)
        self.pub.publish(msg)

    def close(self) -> None:
        self.node.destroy_node()
        rclpy.try_shutdown()
        print("[real hand] Closed. The hand holds the last target sent.")

    def _problems(self) -> list[str]:
        out = []
        missing = [n for n in self.names if n not in self.measured]
        if missing:
            out.append(f"no measured pose on {STATE_TOPIC} for {missing}: is the stack running?")
        if self.pub.get_subscription_count() == 0:
            out.append(f"nobody listens on {TARGET_TOPIC}: start urdf_target_adapter on the robot computer")
        if time.monotonic() - self.status.get("law_time", -math.inf) > CONTROLLER_STALE_S:
            out.append(f"no {CONTROLLER_TOPIC}: the controller is not running")
        elif self.status["law"] == "compliant":
            out.append("the active control law is 'compliant', which ignores targets")
        if self.status.get("torque") is not True:
            out.append(f"torque is not on ({TORQUE_TOPIC} = {self.status.get('torque', 'not received')})")
        return out

    def _stop(self, reason: str, details: list[str]) -> None:
        self.active = False
        print(f"[real hand] {reason}. The sim keeps playing; restart this script to stream again.")
        for detail in details:
            print(f"  {detail}")

    def _clamp(self, targets: list[float]) -> list[float]:
        """Hold the targets inside the real hand's limits: model.json, which the sim uses, can lag behind them."""
        return [min(max(t, lo), hi) for t, lo, hi in zip(targets, self.lower, self.upper)]

    def _spin_until(self, t: float) -> None:
        while (left := t - time.monotonic()) > 0:
            rclpy.spin_once(self.node, timeout_sec=left)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with RL-Games agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # Enable cameras for env
    if args_cli.video:
        env_cfg.enable_cameras = True

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        # env = gym.wrappers.RecordVideo(env, **video_kwargs)
        env = MultiCameraRecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
    # link the real hand. actuated_dof_indices is sorted, so take the names from it, not from cfg.actuated_joint_names
    hand_env = env.unwrapped
    hand_joint_ids = hand_env.actuated_dof_indices
    real_hand = RealHandStream([hand_env.hand.joint_names[i] for i in hand_joint_ids])
    real_hand.start(hand_env.hand.data.joint_pos[0, hand_joint_ids].tolist())
    # Send the simulated hand's actual pose: env.step calls _apply_action just before each physics step, so
    # joint_pos here is the pose from the step that just finished. The wait spreads the sends out evenly,
    # instead of 3 in a burst per policy step.
    sim_apply_action = hand_env._apply_action
    next_send = time.monotonic()

    def apply_action_and_send():
        nonlocal next_send
        sim_apply_action()
        time.sleep(max(0.0, next_send - time.monotonic()))
        real_hand.publish(hand_env.hand.data.joint_pos[0, hand_joint_ids].tolist())
        next_send = max(next_send + hand_env.physics_dt, time.monotonic())

    hand_env._apply_action = apply_action_and_send
    add_joint_pos_probe(env.unwrapped)
    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # convert obs to agent format
            obs = agent.obs_to_torch(obs)
            # agent stepping
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            # env stepping
            obs, _, dones, _ = env.step(actions)

            # perform operations for terminated episodes
            if len(dones) > 0:
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    real_hand.close()
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    try:
        main()
    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")
        raise e
    finally:
        # close sim app
        faulthandler.dump_traceback_later(600, exit=True)
        simulation_app.close()
