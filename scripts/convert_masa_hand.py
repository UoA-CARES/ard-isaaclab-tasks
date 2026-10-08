# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Build the MASA hand (hand_v2_left) USD used by Isaac-ARD-Repose-Cube-Masa-Direct-v0.

The hand is also used as a digital twin, so the USD is built from the URDF
(``robot_hand/models/hand_v2_left/hand_v2_left.urdf`` in robot-hand-control-stack) and
then checked against it:

1. Decimate every mesh (quadric decimation) so the hand is about as light as the Shadow
   Hand asset (~40k triangles). Only the surface detail changes; the script prints how far
   each decimated mesh moves from the original.
2. Convert with IsaacLab's UrdfConverter. Fixed joints are not merged, so every URDF link
   is a USD body with the same name and frame. The URDF has no <collision> tags, so the
   colliders (convex hulls) are made from the visual meshes, which are byte-identical to
   the collision meshes in the hand's MJCF.
3. Fix the importer's empty joint axis. For an axis that is not along x, y or z
   (palm_abd_add), the importer rotates the joint frame so the axis becomes its local x,
   then leaves the axis token empty. The script sets it, after checking that the rotated
   frame really maps x onto the URDF axis.
4. Filter the false self-contacts of the convex hulls. At the open pose (all joints 0)
   the real hand has no self-contact, but the palm hulls fill the hollow where the
   finger bases sit, and PhysX only ignores contacts between a link and its direct
   parent. Pairs whose hulls overlap or come within 1 mm there (the margin covers
   PhysX's 64-vertex hull cooking) get a filtered pair; with the current model these
   are all palm-to-finger-base pairs. Neighbouring fingers are 2.1 mm apart at the
   open pose, so every finger-to-finger pair keeps colliding.
5. Check the USD against the URDF: link frames (forward kinematics at random joint
   positions), joint limits, and effort and velocity limits. The script exits with an
   error if anything differs.
6. With ``--robot_config``, also write ``model.json`` next to the USD with the stack's own
   exporter (``tools/export_sim_model.py``): joint limits, frame signs and tendon routing
   of the real hand, which the task uses to limit target speed like the real driver.

Needs ``pip install fast_simplification`` (used by trimesh for the decimation).

Usage:

    python scripts/convert_masa_hand.py <robot-hand-control-stack>/robot_hand/models/hand_v2_left/hand_v2_left.urdf \
        --robot_config left_hand_v2.yaml
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "source/ard_tasks/ard_tasks/tasks/direct/masa_hand/assets/hand_v2_left/hand_v2_left.usd",
)

parser = argparse.ArgumentParser(description="Convert the MASA hand URDF into the USD used by the MASA task.")
parser.add_argument("input", type=str, help="Path to the MASA hand URDF (hand_v2_left.urdf).")
parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Path of the USD file to write.")
parser.add_argument(
    "--keep_ratio", type=float, default=0.14, help="Fraction of mesh triangles to keep (1.0 = no decimation)."
)
parser.add_argument("--num_checks", type=int, default=200, help="Random joint positions used to check the frames.")
parser.add_argument(
    "--robot_config", type=str, default=None, help="robot-hand-control-stack robot config to export model.json from."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import torch
import trimesh
from pxr import Sdf, Usd, UsdPhysics
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.sim.utils.stage import create_new_stage


def _floats(text, default):
    return np.array([float(v) for v in text.split()]) if text else np.array(default, dtype=float)


def parse_urdf_joints(robot: ET.Element) -> dict:
    joints = {}
    for joint in robot.findall("joint"):
        origin, axis, limit = joint.find("origin"), joint.find("axis"), joint.find("limit")
        joints[joint.get("name")] = {
            "type": joint.get("type"),
            "parent": joint.find("parent").get("link"),
            "child": joint.find("child").get("link"),
            "xyz": _floats(origin.get("xyz") if origin is not None else None, [0, 0, 0]),
            "rpy": _floats(origin.get("rpy") if origin is not None else None, [0, 0, 0]),
            "axis": _floats(axis.get("xyz") if axis is not None else None, [1, 0, 0]),
            "limit": None if limit is None else [float(limit.get(k, 0.0)) for k in ("lower", "upper", "effort", "velocity")],
        }
    return joints


def rpy_matrix(rpy):
    (cr, cp, cy), (sr, sp, sy) = np.cos(rpy), np.sin(rpy)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def axis_angle_matrix(axis, angle):
    k = axis / np.linalg.norm(axis)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx


def urdf_forward_kinematics(joints: dict, q: dict) -> dict:
    """Pose (4x4) of every link relative to the root link, as the URDF defines it."""
    children = {j["child"] for j in joints.values()}
    root = next(j["parent"] for j in joints.values() if j["parent"] not in children)
    poses, todo = {root: np.eye(4)}, [root]
    while todo:
        parent = todo.pop()
        for name, j in joints.items():
            if j["parent"] != parent:
                continue
            t = np.eye(4)
            t[:3, :3], t[:3, 3] = rpy_matrix(j["rpy"]), j["xyz"]
            if j["type"] in ("revolute", "continuous"):
                motion = np.eye(4)
                motion[:3, :3] = axis_angle_matrix(j["axis"], q[name])
                t = t @ motion
            poses[j["child"]] = poses[parent] @ t
            todo.append(j["child"])
    return poses


def quat_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def resolve_mesh(filename: str, urdf_dir: str) -> str:
    """Resolve ``package://<pkg>/<path>`` by finding the <pkg> folder above the URDF."""
    if not filename.startswith("package://"):
        return os.path.join(urdf_dir, filename)
    package, rel_path = filename[len("package://") :].split("/", 1)
    folder = urdf_dir
    while os.path.basename(folder) != package:
        if os.path.dirname(folder) == folder:
            raise FileNotFoundError(f"Package '{package}' not found above {urdf_dir}")
        folder = os.path.dirname(folder)
    return os.path.join(folder, rel_path)


def build_decimated_urdf(urdf_path: str, keep_ratio: float) -> str:
    """Write a copy of the URDF whose meshes are decimated copies; return its path."""
    tree = ET.parse(urdf_path)
    build_dir = tempfile.mkdtemp(prefix="masa_hand_urdf_")
    os.makedirs(os.path.join(build_dir, "meshes"))
    total_before = total_after = 0
    print(f"[INFO] Decimating meshes (keep {keep_ratio:.0%} of triangles):")
    for mesh_el in tree.getroot().iter("mesh"):
        src = resolve_mesh(mesh_el.get("filename"), os.path.dirname(os.path.abspath(urdf_path)))
        dst = os.path.join(build_dir, "meshes", os.path.basename(src))
        mesh = trimesh.load(src, force="mesh")
        small = mesh
        if keep_ratio < 1.0:
            small = mesh.simplify_quadric_decimation(face_count=max(int(len(mesh.faces) * keep_ratio), 100))
        small.export(dst)
        # how far the decimated surface moved: distance from original vertices to it
        sample = mesh.vertices[np.random.default_rng(0).choice(len(mesh.vertices), min(3000, len(mesh.vertices)))]
        _, dist, _ = trimesh.proximity.closest_point(small, sample)
        # colliders are convex hulls, so this is the change that matters for contacts
        _, hull_dist, _ = trimesh.proximity.closest_point(small.convex_hull, mesh.convex_hull.vertices)
        scale = float(mesh_el.get("scale", "1 1 1").split()[0]) * 1000  # mesh units -> mm
        print(
            f"         {os.path.basename(src):24s} {len(mesh.faces):6d} -> {len(small.faces):5d} triangles,"
            f" surface moved max {dist.max() * scale:.2f} mm (mean {dist.mean() * scale:.3f} mm),"
            f" collision hull moved max {hull_dist.max() * scale:.2f} mm"
        )
        total_before, total_after = total_before + len(mesh.faces), total_after + len(small.faces)
        mesh_el.set("filename", os.path.join("meshes", os.path.basename(src)))
    print(f"[INFO] Triangles in total: {total_before} -> {total_after}")
    out = os.path.join(build_dir, os.path.basename(urdf_path))
    tree.write(out)
    return out


def fix_empty_joint_axes(usd_path: str, joints: dict):
    """Set the axis the importer left empty, after checking the joint frame maps x onto it."""
    stage = Usd.Stage.Open(usd_path)
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        attr = UsdPhysics.RevoluteJoint(prim).GetAxisAttr()
        if attr.Get() in ("X", "Y", "Z"):
            continue
        urdf_axis = joints[prim.GetName()]["axis"] / np.linalg.norm(joints[prim.GetName()]["axis"])
        quat = UsdPhysics.Joint(prim).GetLocalRot1Attr().Get()
        local_x = quat_matrix(*quat.GetImaginary(), quat.GetReal())[:, 0]
        if np.dot(local_x, urdf_axis) < 1.0 - 1e-5:
            raise RuntimeError(f"Joint '{prim.GetName()}': empty axis and its frame does not map x onto {urdf_axis}")
        spec = attr.GetPropertyStack()[0]
        spec.default = "X"
        spec.layer.Save()
        print(f"[INFO] Joint '{prim.GetName()}': set empty axis to X (joint frame maps x onto URDF axis {urdf_axis})")


def remove_dangling_references(usd_path: str):
    """Drop references to prims that do not exist (the importer adds one for links without geometry)."""
    config_dir = os.path.join(os.path.dirname(usd_path), "configuration")
    layers = [Sdf.Layer.FindOrOpen(os.path.join(config_dir, f)) for f in sorted(os.listdir(config_dir))]
    for layer in layers:
        dangling = []

        def visit(path, layer=layer, dangling=dangling):
            spec = layer.GetObjectAtPath(path)
            if not isinstance(spec, Sdf.PrimSpec):
                return
            for ref in spec.referenceList.GetAddedOrExplicitItems():
                targets = layers if not ref.assetPath else [Sdf.Layer.FindOrOpen(layer.ComputeAbsolutePath(ref.assetPath))]
                if not any(t.GetPrimAtPath(ref.primPath) for t in targets):
                    dangling.append((spec, ref))

        layer.Traverse(Sdf.Path.absoluteRootPath, visit)
        for spec, ref in dangling:
            spec.referenceList.RemoveItemEdits(ref)
            print(f"[INFO] Removed reference to missing prim {ref.primPath} from {spec.path}")
        if dangling:
            layer.Save()


def filter_open_pose_hull_overlaps(usd_path: str, build_urdf: str, joints: dict, margin: float = 0.001):
    """Filter collisions between links whose convex hulls overlap (or are within ``margin``) at the open pose."""
    build_dir = os.path.dirname(build_urdf)
    poses = urdf_forward_kinematics(joints, {n: 0.0 for n, j in joints.items() if j["type"] == "revolute"})
    hulls = {}
    for link in ET.parse(build_urdf).getroot().findall("link"):
        visual = link.find("visual")
        if visual is None:
            continue
        mesh_el, origin = visual.find("geometry/mesh"), visual.find("origin")
        pts = trimesh.load(os.path.join(build_dir, mesh_el.get("filename")), force="mesh").convex_hull.vertices
        pts = pts * float(mesh_el.get("scale", "1 1 1").split()[0])
        if origin is not None:
            pts = pts @ rpy_matrix(_floats(origin.get("rpy"), [0, 0, 0])).T + _floats(origin.get("xyz"), [0, 0, 0])
        pose = poses[link.get("name")]
        hulls[link.get("name")] = ConvexHull(pts @ pose[:3, :3].T + pose[:3, 3])

    def depth(ha, hb):
        # deepest point inside both hulls: max t s.t. n.x + c + t <= 0 for every face; t < 0 is a gap
        eq = np.vstack([ha.equations, hb.equations])
        res = linprog(
            c=[0, 0, 0, -1],
            A_ub=np.hstack([eq[:, :3], np.ones((len(eq), 1))]),
            b_ub=-eq[:, 3],
            bounds=[(None, None)] * 3 + [(None, 0.05)],
            method="highs",
        )
        return res.x[3] if res.success else -np.inf

    jointed = {frozenset((j["parent"], j["child"])) for j in joints.values()}
    names = sorted(hulls)
    pairs = [
        (a, b, depth(hulls[a], hulls[b]))
        for k, a in enumerate(names)
        for b in names[k + 1 :]
        if frozenset((a, b)) not in jointed  # PhysX already ignores parent-child contacts
    ]
    pairs = [(a, b, d) for a, b, d in pairs if d > -margin]

    name = os.path.splitext(os.path.basename(usd_path))[0]
    stage = Usd.Stage.Open(os.path.join(os.path.dirname(usd_path), "configuration", f"{name}_physics.usd"))
    for a, b, d in pairs:
        api = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(f"/{name}/{a}"))
        api.GetFilteredPairsRel().AddTarget(f"/{name}/{b}")
        print(f"[INFO] Filtered self-contact {a} - {b} (hulls {'overlap' if d > 0 else 'gap'} {abs(d) * 1000:.2f} mm at the open pose)")
    stage.GetRootLayer().Save()


def check_against_urdf(usd_path: str, joints: dict, num_checks: int):
    """Compare link frames and joint limits of the USD with the URDF; raise on mismatch."""
    create_new_stage()
    sim = SimulationContext(SimulationCfg(dt=1 / 120, device="cpu"))
    hand = Articulation(
        ArticulationCfg(
            prim_path="/World/Hand",
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd_path, rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)
            ),
            actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
        )
    )
    sim.reset()
    view = hand.root_physx_view
    revolute = {n: j for n, j in joints.items() if j["type"] == "revolute"}
    missing = (set(revolute) ^ set(hand.joint_names)) | ({j["child"] for j in joints.values()} - set(hand.body_names))
    if missing:
        raise RuntimeError(f"USD and URDF names differ: {sorted(missing)}")

    # joint limits, effort and velocity, in the articulation's joint order
    urdf_limits = torch.tensor([revolute[n]["limit"] for n in hand.joint_names])
    usd_limits = torch.cat(
        [view.get_dof_limits()[0], view.get_dof_max_forces()[0, :, None], view.get_dof_max_velocities()[0, :, None]],
        dim=-1,
    ).cpu()
    limit_err = (usd_limits - urdf_limits).abs().max(dim=0).values
    print(
        "[CHECK] joint limits vs URDF, max difference: "
        f"lower {limit_err[0]:.1e} rad, upper {limit_err[1]:.1e} rad, effort {limit_err[2]:.1e} Nm,"
        f" velocity {limit_err[3]:.1e} rad/s"
    )

    # link frames relative to the root link, at random joint positions
    rng = np.random.default_rng(0)
    pos_err = rot_err = 0.0
    root_id = hand.body_names.index(next(iter(urdf_forward_kinematics(joints, {n: 0.0 for n in revolute}))))
    for _ in range(num_checks):
        q = {n: rng.uniform(j["limit"][0], j["limit"][1]) for n, j in revolute.items()}
        q_sim = torch.tensor([[q[n] for n in hand.joint_names]], dtype=torch.float32, device=sim.device)
        hand.write_joint_state_to_sim(q_sim, torch.zeros_like(q_sim))
        sim.physics_sim_view.update_articulations_kinematic()
        tf = view.get_link_transforms()[0].cpu().numpy().astype(np.float64)  # xyz + quat (x, y, z, w)
        sim_poses = []
        for t in tf:
            m = np.eye(4)
            m[:3, :3] = quat_matrix(*t[3:7])
            m[:3, 3] = t[:3]
            sim_poses.append(m)
        root_inv = np.linalg.inv(sim_poses[root_id])
        for link, urdf_pose in urdf_forward_kinematics(joints, q).items():
            rel = root_inv @ sim_poses[hand.body_names.index(link)]
            pos_err = max(pos_err, np.linalg.norm(rel[:3, 3] - urdf_pose[:3, 3]))
            # angle between the two rotations; stable near 0, unlike arccos of the trace
            chord = np.linalg.norm(rel[:3, :3] - urdf_pose[:3, :3]) / (2.0 * np.sqrt(2.0))
            rot_err = max(rot_err, np.degrees(2.0 * np.arcsin(min(chord, 1.0))))
    print(
        f"[CHECK] link frames vs URDF forward kinematics ({num_checks} random joint positions, {len(hand.body_names)}"
        f" links): max position error {pos_err * 1000:.4f} mm, max rotation error {rot_err:.4f} deg"
    )
    if pos_err > 1e-4 or rot_err > 0.05 or limit_err[:2].max() > 1e-4 or limit_err[2:].max() > 1e-3:
        raise RuntimeError("The USD does not match the URDF (see the [CHECK] lines above).")
    print("[CHECK] USD matches the URDF.")


def main():
    urdf_path = os.path.abspath(args_cli.input)
    joints = parse_urdf_joints(ET.parse(urdf_path).getroot())
    build_urdf = build_decimated_urdf(urdf_path, args_cli.keep_ratio)

    output = os.path.abspath(args_cli.output)
    out_dir = os.path.dirname(output)
    UrdfConverter(
        UrdfConverterCfg(
            asset_path=build_urdf,
            usd_dir=out_dir,
            usd_file_name=os.path.basename(output),
            force_usd_conversion=True,
            fix_base=True,
            merge_fixed_joints=False,
            collision_from_visuals=True,
            collider_type="convex_hull",
            self_collision=False,
            # keep the URDF's effort and velocity limits; gains are set by the task config
            joint_drive=None,
        )
    )
    # converter cache files; config.yaml also holds this machine's absolute paths
    for cache_file in (".asset_hash", "config.yaml"):
        if os.path.exists(os.path.join(out_dir, cache_file)):
            os.remove(os.path.join(out_dir, cache_file))
    print(f"[INFO] Wrote {output}")

    if args_cli.robot_config:
        # the stack's exporter, found by walking up from the URDF to the repo root
        stack = os.path.dirname(urdf_path)
        while not os.path.isfile(os.path.join(stack, "tools", "export_sim_model.py")):
            if os.path.dirname(stack) == stack:
                raise FileNotFoundError("tools/export_sim_model.py not found above the URDF")
            stack = os.path.dirname(stack)
        exporter = os.path.join(stack, "tools", "export_sim_model.py")
        model_json = os.path.join(out_dir, "model.json")
        subprocess.run([sys.executable, exporter, "--config", args_cli.robot_config, "-o", model_json], check=True)
        print(f"[INFO] Wrote {model_json} (robot-hand-control-stack {args_cli.robot_config})")

    remove_dangling_references(output)
    fix_empty_joint_axes(output, joints)
    filter_open_pose_hull_overlaps(output, build_urdf, joints)
    check_against_urdf(output, joints, args_cli.num_checks)


if __name__ == "__main__":
    main()
    simulation_app.close()
