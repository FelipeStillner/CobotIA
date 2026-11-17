"""Desk workspace and camera→robot mapping.

Place the camera at the center of the short side of the pad (40 cm)
and aim it along the long side (70 cm). The lens axis is the middle
of that short edge, so the pad extends 20 cm to the left and 20 cm
to the right.

RealSense camera frame (meters):
    X right, Y down, Z forward (away from the lens).

The robot TCP is commanded on a plane 30 cm above the table,
directly over the detected object. It is never sent below that plane.
"""

from __future__ import annotations

import numpy as np

# Rectangle in front of the camera, in the camera frame (meters).
# Short side is 40 cm, centered on the lens. Long side is 70 cm forward.
WORK_WIDTH_X = 0.40     # 40 cm, the short side
WORK_DEPTH_Z = 0.70     # 70 cm, the long side
WORK_X0 = -WORK_WIDTH_X / 2   # 20 cm to the left of the lens
WORK_Z0 = 0.20          # near edge: 20 cm in front of the lens

# Safety plane. TCP Z is always TABLE_Z_ROBOT + this, never lower.
HOVER_ABOVE_TABLE_M = 0.30

# Height of the table surface in the UR base frame (meters).
# Jog the TCP until it just touches the table and paste that Z here.
# With the base origin on the table, 0.0 is correct and the hover is Z = 0.30 m.
TABLE_Z_ROBOT = 0.0

# Camera (X, Z) → robot (X, Y), meters.
# robot_xy = R @ [cam_x, cam_z] + t
# Replace with a real calibration (jog the TCP over the pad corners).
# Until CALIBRATED is True, follow_object.py will preview only.
CALIBRATED = False
CAM_XZ_TO_ROBOT_R = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
CAM_XZ_TO_ROBOT_T = np.array([0.35, -0.15], dtype=float)

# Motion limits
MAX_STEP_M = 0.015       # max XY change sent per camera frame
SERVO_SPEED = 0.02
SERVO_ACCEL = 0.02
ROBOT_IP = "127.0.0.1"

# Depth blob: object must be this much closer than the background
CLOSER_THAN_M = 0.02
MIN_OBJECT_AREA_PX = 400
