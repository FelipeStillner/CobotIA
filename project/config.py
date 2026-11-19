"""Robot command settings.

The pad, the camera-to-base map, and the hover height are written by
calibrate.py into calibration.json. The tool stays at least MIN_TCP_Z
above the base, and CLEARANCE_M above the detected object center.
"""

from __future__ import annotations

import math

# Table surface in the robot BASE frame, meters. Measured 2026-11-18 with
# the TCP_1 tip touching the pad, read over RTDE (getActualTCPPose):
# [0.0258, -0.5759, -0.0018, ...]. The pendant's "Vue"/View feature on this
# cell is rotated 45 deg and offset ~406 mm in Z, so never copy Z from it.
# The hover plane must sit at least HOVER_ABOVE_TABLE_M above this.
# Set to None to block all motion. Re-measure if the table, the robot mount
# or the TCP changes.
TABLE_Z_ROBOT: float | None = -0.0018
# The saved hover plane must sit at least this far above the table.
HOVER_ABOVE_TABLE_M = 0.20

# Runtime limits. The interface may change the values below, and safety.py
# clamps them again before a pose is built.
SPEED_MIN_M_S = 0.03
SPEED_MAX_M_S = 0.30
HEIGHT_MIN_M = 0.10
HEIGHT_MAX_M = 0.50
# How far above the detected object center the TCP is sent.
CLEARANCE_M = 0.20
# Lowest base-frame Z the TCP may be commanded to.
MIN_TCP_Z = 0.200

# The tool orientation kept while tracking is read from the robot by
# calibrate.py and saved as "tcp_orientation" in calibration.json.

# Tracking (servoL). servoL ignores its speed and acceleration arguments:
# the speed is the step to the target divided by the servo time. Each command
# uses the real time since the previous one (dt, capped at MAX_SERVO_TIME) as
# its servo time and moves at most MAX_SPEED_M_S * dt from the actual TCP, so
# the TCP cannot go faster than MAX_SPEED_M_S however fast the loop runs, and
# a slow frame cannot turn into a big jump (max step MAX_SPEED_M_S * MAX_SERVO_TIME).
MAX_SPEED_M_S = 0.10
SERVO_TIME = 1.0 / 30.0     # dt for the first command after a start or hold
MIN_SERVO_TIME = 0.008
MAX_SERVO_TIME = 0.2
SERVO_LOOKAHEAD = 0.1       # range 0.03–0.2 s
SERVO_GAIN = 200            # range 100–2000

# Tracking only starts once the tool orientation matches and Z is at least
# the floor; it holds instead of servoing if it drifts off either.
HOVER_Z_TOL_M = 0.005
ORIENTATION_TOL_RAD = math.radians(3.0)

# Slow straight-line move onto the hover plane before tracking starts.
APPROACH_SPEED = 0.1      # m/s
APPROACH_ACCEL = 0.05      # m/s^2

ROBOT_IP = "100.65.63.94"

MIN_OBJECT_AREA_PX = 400

# Several objects: the one closest to the camera (straight-line distance from
# the lens) is tracked. To stop depth noise flipping between two objects at
# about the same distance, the tracked one is only dropped for another that is
# at least this much closer. 0 = always the closest, every frame.
OBJECT_SWITCH_MARGIN_M = 0.02
# A detection within this many pixels of last frame's target is the same object.
OBJECT_MATCH_PX = 50
