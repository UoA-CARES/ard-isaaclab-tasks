import torch
import isaaclab.utils.math as math_utils

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

# Set this to opengl convention to match the camera orientation in Isaac Sim viewport. The default USD convention is Y-up, which is different from the OpenGL convention (Z-up).
COORD_SYS = "opengl"

# Top view cameras
TOP_VIEW_CAMERA = CameraCfg(
    prim_path="/World/envs/env_.*/topview_camera",
    width=640,
    height=480,
    offset = CameraCfg.OffsetCfg(
        pos = (0, -0.35, 1.5),
        rot = (1, 0, 0, 0),
        convention = COORD_SYS
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
        pos = (0.6, 0.135, 0.94),
        rot = (0.40607, 0.24949, 0.45663, 0.75123),
        convention = COORD_SYS
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
        rot = (-0.22883, -0.11546, 0.45061, 0.85514),
        convention = COORD_SYS
    ),
    data_types = ["rgb"],
    spawn = sim_utils.PinholeCameraCfg(
        focal_length = 18,
    )
)

# Front view camera
CAMERA_2 = CameraCfg(
    prim_path = "/World/envs/env_.*/frontview_camera",
    width=640,
    height=480,
    offset = CameraCfg.OffsetCfg(
        pos = (-0.1, -1, 0.65),
        rot = (0.73313, 0.68008, -0.00243, -0.00226),
        convention = COORD_SYS
    ),
    data_types = ["rgb"],
    spawn = sim_utils.PinholeCameraCfg(
        focal_length = 18,
    )
)

