import torch
import isaaclab.utils.math as math_utils

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

# Helper to convert deg to quat notation
def deg_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    r = torch.tensor([roll * torch.pi / 180.0])
    p = torch.tensor([pitch * torch.pi / 180.0])
    y = torch.tensor([yaw * torch.pi / 180.0])
    quat_tensor = math_utils.quat_from_euler_xyz(r, p, y)
    
    return tuple(quat_tensor[0].tolist())

#Change convention to world if the cameras dont work

# Top view camera
TOP_VIEW_CAMERA = CameraCfg(
    prim_path="/World/envs/env_.*/topview_camera",
    width=640,
    height=480,
    offset = CameraCfg.OffsetCfg(
        pos = (0, -0.35, 1.5),
        rot = deg_to_quat(0, 0, 0),
        convention = "world"
    ),
    data_types = ["rgb"],
    spawn = sim_utils.PinholeCameraCfg(
        focal_length = 20,
    )
)

# Isometric view camera 0
CAMERA_0 = CameraCfg(
    prim_path = "/World/envs/env_.*/camera_0",
    width=640,
    height=480,
    offset = CameraCfg.OffsetCfg(
        pos = (0.5, 0, 0.8),
        rot = deg_to_quat(45, 45, 145),
        convention = "world"
    ),
    data_types = ["rgb"],
    spawn = sim_utils.PinholeCameraCfg(
        focal_length = 18,
    )
)

# Isometric view camera 1
CAMERA_1 = CameraCfg(
    prim_path = "/World/envs/env_.*/camera_1",
    width=640,
    height=480,
    offset = CameraCfg.OffsetCfg(
        pos = (-0.3, 0, 0.9),
        rot = deg_to_quat(-35, -35, 140),
        convention = "world"
    ),
    data_types = ["rgb"],
    spawn = sim_utils.PinholeCameraCfg(
        focal_length = 18,
    )
)

# Isometric view camera 2
CAMERA_2 = CameraCfg(
    prim_path = "/World/envs/env_.*/camera_2",
    width=640,
    height=480,
    offset = CameraCfg.OffsetCfg(
        pos = (0, -1, 0.6),
        rot = deg_to_quat(80, 0, 0),
        convention = "world"
    ),
    data_types = ["rgb"],
    spawn = sim_utils.PinholeCameraCfg(
        focal_length = 18,
    )
)

