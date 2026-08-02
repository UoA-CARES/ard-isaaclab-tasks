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
        pos = (0.5, 0, 0.8),
        rot = (0.39634, 0.23087, 0.44351, 0.77001),
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
        rot = (0.39606, 0.17141, -0.36758, -0.8238),
        convention = COORD_SYS
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
        rot = (0.76604, 0.64279, 0, 0),
        convention = COORD_SYS
    ),
    data_types = ["rgb"],
    spawn = sim_utils.PinholeCameraCfg(
        focal_length = 18,
    )
)

