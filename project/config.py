"""Robot command settings.

The pad and the camera-to-base map come from the four taught corners.
TCP Z stays on the plane measured on the pendant (about -0.227 m),
which is 30 cm above the table in this cell. It is never commanded lower.
"""

from __future__ import annotations

import numpy as np

# Safety plane. TCP Z is always TABLE_Z_ROBOT + this, never lower.
# The four pendant readings sit on Z = -0.2272 m, so the table in the
# base frame is 30 cm below that.
HOVER_ABOVE_TABLE_M = 0.30
TABLE_Z_ROBOT = -0.5272

# Camera (X, Z) → robot (X, Y), meters. Fit to the four taught corners (2.7 mm).
# robot_xy = R @ [cam_x, cam_z] + t
CALIBRATED = True
CAM_XZ_TO_ROBOT_R = np.array(
    [[-0.73210148, 0.69091082], [0.65942076, 0.7020882]], dtype=float
)
CAM_XZ_TO_ROBOT_T = np.array([-0.05686844, -0.72815036], dtype=float)

# Tool orientation from the pendant, radians. Kept fixed while tracking.
TCP_ORIENTATION = [2.1705, -2.2980, 0.0465]

# Motion limits
MAX_STEP_M = 0.015       # max XY change sent per camera frame
SERVO_SPEED = 0.01
SERVO_ACCEL = 0.01
ROBOT_IP = "100.65.63.94"

MIN_OBJECT_AREA_PX = 400
