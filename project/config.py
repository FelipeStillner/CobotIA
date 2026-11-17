"""Camera frame and the robot command that the next step will send.

The pad on the table comes from calibration.json (the four clicked corners).
RealSense camera frame (meters): X right, Y down, Z forward.

The TCP stays on a plane 30 cm above the table, directly over the object.
"""

from __future__ import annotations

import numpy as np

# Safety plane. TCP Z is always TABLE_Z_ROBOT + this, never lower.
HOVER_ABOVE_TABLE_M = 0.30

# Height of the table surface in the UR base frame (meters).
# Jog the TCP until it just touches the table and paste that Z here.
# With the base origin on the table, 0.0 is correct and the hover is Z = 0.30 m.
TABLE_Z_ROBOT = 0.0

# Camera (X, Z) → robot (X, Y), meters.
# robot_xy = R @ [cam_x, cam_z] + t
# Used when the robot link is turned on. The camera loop does not read this yet.
CALIBRATED = False
CAM_XZ_TO_ROBOT_R = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
CAM_XZ_TO_ROBOT_T = np.array([0.35, -0.15], dtype=float)

# Motion limits
MAX_STEP_M = 0.015       # max XY change sent per camera frame
SERVO_SPEED = 0.01
SERVO_ACCEL = 0.01
ROBOT_IP = "100.65.63.94"

MIN_OBJECT_AREA_PX = 400
