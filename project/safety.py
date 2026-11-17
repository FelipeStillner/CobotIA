"""Pad check and the pose the robot will be sent.

enforce_hover and limit_step are for the robot link. The camera loop
does not call them yet.
"""

from __future__ import annotations

import numpy as np

from calibration import xz_inside
from config import (
    CAM_XZ_TO_ROBOT_R,
    CAM_XZ_TO_ROBOT_T,
    HOVER_ABOVE_TABLE_M,
    TABLE_Z_ROBOT,
)


def hover_z() -> float:
    return float(TABLE_Z_ROBOT + HOVER_ABOVE_TABLE_M)


def in_camera_workspace(cam_x: float, cam_z: float) -> bool:
    """True when the point is inside the calibrated pad."""
    return bool(np.asarray(xz_inside(cam_x, cam_z)).reshape(-1)[0])


def camera_xz_to_robot_xy(cam_x: float, cam_z: float) -> tuple[float, float]:
    v = CAM_XZ_TO_ROBOT_R @ np.array([cam_x, cam_z], dtype=float) + CAM_XZ_TO_ROBOT_T
    return float(v[0]), float(v[1])


def target_over_object(cam_x: float, cam_z: float) -> list[float] | None:
    """Robot [x, y, z] above the object, or None if it is outside the pad.

    Z is the hover plane. The object's own height is not used, so the
    TCP cannot be pulled down onto the table.
    """
    if not in_camera_workspace(cam_x, cam_z):
        return None
    x, y = camera_xz_to_robot_xy(cam_x, cam_z)
    return [x, y, hover_z()]


def enforce_hover(pose: list[float]) -> list[float]:
    """Copy of pose with Z on or above the hover plane.

    Raises if a caller tries to command anything below the 30 cm floor.
    """
    if len(pose) < 3:
        raise ValueError("pose needs at least x, y, z")
    floor = hover_z()
    if pose[2] < floor - 1e-6:
        raise RuntimeError(
            f"Refusing Z={pose[2]:.4f} m — below the {HOVER_ABOVE_TABLE_M:.2f} m "
            f"hover floor (Z >= {floor:.4f} m)"
        )
    out = list(pose)
    out[2] = max(float(out[2]), floor)
    return out


def limit_step(current_xy: tuple[float, float], target_xy: tuple[float, float], max_step: float) -> tuple[float, float]:
    """Move at most max_step meters from current toward target."""
    dx = target_xy[0] - current_xy[0]
    dy = target_xy[1] - current_xy[1]
    dist = float(np.hypot(dx, dy))
    if dist <= max_step or dist == 0.0:
        return target_xy
    scale = max_step / dist
    return (current_xy[0] + dx * scale, current_xy[1] + dy * scale)
