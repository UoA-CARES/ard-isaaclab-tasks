#!/usr/bin/env python3
"""Move the real MASA hand (hand_v2_left) to a chosen pose over ROS 2.

Name the joints and their angles in degrees, in the URDF frame: the same frame as the sim
joint positions. The limits come from the real hand's robot config (left_hand_v2.yaml in
robot-hand-control-stack), not from the sim's model.json, which can lag behind it. Without
--send the script only checks the pose and prints it; it does not touch ROS.

    python3 scripts/real_hand_pose.py --list                             # joint names and limits
    python3 scripts/real_hand_pose.py index_pip=45 index_dip=30          # check only
    python3 scripts/real_hand_pose.py index_pip=45 index_dip=30 --send   # move the real hand

--send needs ROS 2 Jazzy in this shell (run `jazzy` first, not inside the Isaac conda env).
On the robot computer: the stack running, controller started, torque on, plus the URDF
frame adapter:

    ros2 run robot_hand urdf_target_adapter --ros-args -p config_file_path:=<robot.yaml>

The script publishes to /robot_hand/target_joint_states_urdf (radians, URDF names), the
stack's input for sim-frame targets. The adapter converts them to the stack's internal
frame, and the stack's own limits and velocity guard still apply. Only the named joints
move, in a straight line from the measured pose at --speed; the others keep their current
target. The stack holds the last target after the script exits.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import yaml

ROBOT_YAML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "robot-hand-control-stack", "robot_hand", "config", "robots", "left_hand_v2.yaml",
)
STATE_TOPIC = "/robot_hand/joint_states_urdf"  # measured pose, radians, URDF names
TARGET_TOPIC = "/robot_hand/target_joint_states_urdf"  # targets, radians, URDF names (read by urdf_target_adapter)
CONTROLLER_TOPIC = "/robot_hand/controller_state"  # published every controller tick; frame_id = active control law
TORQUE_TOPIC = "/robot_hand/torque_enabled"  # latched Bool from the driver
RATE_HZ = 50.0  # target publish rate during the move
MAX_SPEED_DEG_S = math.inf  # upper bound for --speed (the motors reach about 240)
WAIT_S = 10.0  # how long to wait for the robot's topics


def load_limits(path: str) -> dict[str, tuple[float, float]]:
    """URDF joint name -> (min_deg, max_deg), from the robot config the real hand's controller clamps to."""
    try:
        with open(path) as f:
            doc = yaml.safe_load(f) or {}
    except FileNotFoundError:
        sys.exit(f"Robot config not found: {path}\nPass its path with --robot-yaml.")
    params = doc.get("/**", {}).get("ros__parameters", doc)  # same unwrapping as the stack's config loader
    aliases = params.get("urdf_joint_aliases") or {}  # stack name -> URDF name
    limits = {}
    for finger in (params.get("fingers") or {}).values():
        for name, joint in (finger.get("joints") or {}).items():
            lo, hi = joint["range_deg"]
            limits[aliases.get(name, name)] = (float(lo), float(hi))
    if not limits:
        sys.exit(f"No joints found in {path}")
    return limits


def parse_targets(pairs: list[str], limits: dict[str, tuple[float, float]]) -> dict[str, float]:
    """'index_pip=45' strings -> {joint: deg}. Exits if any joint is unknown, repeated or out of its limits."""
    targets, errors = {}, []
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep:
            errors.append(f"{pair!r}: write it as joint=degrees, e.g. index_pip=45")
            continue
        if name not in limits:
            errors.append(f"{name!r}: unknown joint (see --list)")
            continue
        if name in targets:
            errors.append(f"{name}: given twice")
            continue
        try:
            deg = float(value)
        except ValueError:
            errors.append(f"{pair!r}: {value!r} is not a number")
            continue
        lo, hi = limits[name]
        if not (math.isfinite(deg) and lo <= deg <= hi):
            errors.append(f"{name}={value}: outside its limits {lo:g}..{hi:g} deg")
            continue
        targets[name] = deg
    if errors:
        sys.exit("Pose rejected, nothing sent:\n  " + "\n  ".join(errors))
    return targets


def send(targets: dict[str, float], speed_deg_s: float) -> None:
    """Ramp the named joints from the measured pose to the targets, then report the reached pose."""
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool
    except ImportError:
        sys.exit("rclpy not found: run `jazzy` first (system Python, not the Isaac conda env).")
    if not os.environ.get("ROS_DISCOVERY_SERVER"):
        print("Warning: ROS_DISCOVERY_SERVER is not set. Run `jazzy` first if the hand is on another computer.")

    rclpy.init()
    node = Node("real_hand_pose")
    measured: dict[str, float] = {}  # URDF name -> rad
    status: dict[str, object] = {}  # "law": active control law, "torque": driver torque state
    node.create_subscription(JointState, STATE_TOPIC, lambda m: measured.update(zip(m.name, m.position)), 10)
    node.create_subscription(JointState, CONTROLLER_TOPIC, lambda m: status.update(law=m.header.frame_id), 10)
    node.create_subscription(
        Bool, TORQUE_TOPIC, lambda m: status.update(torque=m.data),
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
    )
    pub = node.create_publisher(JointState, TARGET_TOPIC, 10)

    def spin_until(t: float) -> None:
        while (left := t - time.monotonic()) > 0:
            rclpy.spin_once(node, timeout_sec=left)

    def problems() -> list[str]:
        # A target sent while the controller is stopped, the torque is off or the law ignores
        # targets is stored, and the hand jumps to it later when that changes. So refuse instead.
        out = []
        missing = [n for n in targets if n not in measured]
        if missing:
            out.append(f"no measured pose on {STATE_TOPIC} for {missing}: is the stack running?")
        if pub.get_subscription_count() == 0:
            out.append(f"nobody listens on {TARGET_TOPIC}: start urdf_target_adapter on the robot computer")
        if "law" not in status:
            out.append(f"no {CONTROLLER_TOPIC}: the controller is not started")
        elif status["law"] == "compliant":
            out.append("the active control law is 'compliant', which ignores targets")
        if status.get("torque") is not True:
            out.append(f"torque is not on ({TORQUE_TOPIC} = {status.get('torque', 'not received')})")
        return out

    try:
        deadline = time.monotonic() + WAIT_S
        while problems() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if found := problems():
            sys.exit("Not ready, nothing sent:\n  " + "\n  ".join(found))

        print(f"{'joint':14s} {'now':>8s} {'target':>8s} {'move':>8s}   (deg, URDF frame)")
        for n, deg in targets.items():
            now = math.degrees(measured[n])
            print(f"{n:14s} {now:8.1f} {deg:8.1f} {deg - now:+8.1f}")
        if input(f"Move the real hand at {speed_deg_s:g} deg/s? [y/N] ").strip().lower() != "y":
            print("Cancelled, nothing sent.")
            return

        spin_until(time.monotonic() + 0.2)  # refresh the measured pose after the prompt
        names = list(targets)
        start = {n: math.degrees(measured[n]) for n in names}
        duration = max(abs(targets[n] - start[n]) for n in names) / speed_deg_s
        steps = max(1, math.ceil(duration * RATE_HZ))
        t0 = time.monotonic()
        # print(f"STEP: {steps}", f"DURATION: {duration:.2f}s", f"RATE: {RATE_HZ:.1f}Hz", sep=", ")
        for k in range(1, steps + 1):
            a = k / steps
            msg = JointState()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.name = names
            msg.position = [math.radians(start[n] + a * (targets[n] - start[n])) for n in names]
            pub.publish(msg)
            spin_until(t0 + k / RATE_HZ)

        spin_until(time.monotonic() + 2.0)  # let the hand settle
        print(f"\n{'joint':14s} {'target':>8s} {'reached':>8s} {'error':>8s}   (deg)")
        for n in names:
            reached = math.degrees(measured[n])
            print(f"{n:14s} {targets[n]:8.1f} {reached:8.1f} {reached - targets[n]:+8.1f}")
    except KeyboardInterrupt:
        print("\nStopped. The hand holds the last target sent.")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="*", help="joint=degrees pairs, e.g. index_pip=45 (URDF frame)")
    parser.add_argument("--send", action="store_true", help="move the real hand (default: check the pose only)")
    parser.add_argument("--speed", type=float, default=math.inf, help="joint speed of the move in deg/s (default 30)")
    parser.add_argument("--list", action="store_true", help="print the joint names and limits, then exit")
    parser.add_argument("--robot-yaml", default=ROBOT_YAML, help="the real hand's robot config, for the joint limits")
    args = parser.parse_args()

    limits = load_limits(args.robot_yaml)
    if args.list:
        for name, (lo, hi) in limits.items():
            print(f"{name:14s} {lo:g} .. {hi:g} deg")
        return
    if not args.targets:
        parser.error("name at least one joint, e.g. index_pip=45 (see --list)")
    if not 0 < args.speed <= MAX_SPEED_DEG_S:
        parser.error(f"--speed must be in (0, {MAX_SPEED_DEG_S:g}] deg/s")

    targets = parse_targets(args.targets, limits)
    if not args.send:
        for name, deg in targets.items():
            lo, hi = limits[name]
            print(f"{name:14s} {deg:7.1f} deg   (limits {lo:g}..{hi:g})")
        print(f"Limits from {os.path.normpath(args.robot_yaml)}")
        print("Check only, nothing sent. Add --send to move the real hand.")
        return
    send(targets, args.speed)


if __name__ == "__main__":
    main()
