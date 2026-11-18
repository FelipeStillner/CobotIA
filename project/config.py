"""Robot command settings.

The pad, the camera-to-base map, and the hover height are written by
calibrate.py into calibration.json from the four corners and the
X, Y, Z you type for each one. TCP Z stays on that height.
"""

from __future__ import annotations

# The Z typed at each corner should already be this far above the table.
HOVER_ABOVE_TABLE_M = 0.30

# Tool orientation from the pendant, radians. Kept fixed while tracking.
TCP_ORIENTATION = [2.1705, -2.2980, 0.0465]

# Motion limits
MAX_STEP_M = 0.015       # max XY change sent per camera frame
SERVO_SPEED = 0.01
SERVO_ACCEL = 0.01
ROBOT_IP = "100.65.63.94"

MIN_OBJECT_AREA_PX = 400
