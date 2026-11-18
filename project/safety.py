"""Pad check and the pose sent to the robot.

Z is locked to the average of the four corner heights typed in calibrate.py.
"""

from __future__ import annotations

import numpy as np

from calibration import load_calibration, xz_inside
from config import HOVER_ABOVE_TABLE_M


def _robot_map() -> tuple[np.ndarray, np.ndarray] | None:
    calib = load_calibration()
    if not calib or "cam_xz_to_robot_R" not in calib or "cam_xz_to_robot_t" not in calib:
        return None
    rotation = np.asarray(calib["cam_xz_to_robot_R"], dtype=float)
    translation = np.asarray(calib["cam_xz_to_robot_t"], dtype=float)
    if rotation.shape != (2, 2) or translation.shape != (2,):
        return None
    return rotation, translation


def has_robot_map() -> bool:
    return _robot_map() is not None and hover_z() is not None


def hover_z() -> float | None:
    """Typed corner height in meters, or None until calibrate.py saves it."""
    calib = load_calibration()
    if not calib or calib.get("hover_z") is None:
        return None
    return float(calib["hover_z"])


def in_camera_workspace(cam_x: float, cam_z: float) -> bool:
    """True when the point is inside the calibrated pad."""
    return bool(np.asarray(xz_inside(cam_x, cam_z)).reshape(-1)[0])


def camera_xz_to_robot_xy(cam_x: float, cam_z: float) -> tuple[float, float] | None:
    mapped = _robot_map()
    if mapped is None:
        return None
    rotation, translation = mapped
    v = rotation @ np.array([cam_x, cam_z], dtype=float) + translation
    return float(v[0]), float(v[1])


def target_over_object(cam_x: float, cam_z: float) -> list[float] | None:
    """Robot [x, y, z] above the object, or None if it cannot be commanded.

    Z is the hover plane from the four typed corner heights. The object's
    own height is not used, so the TCP cannot be pulled down onto the table.
    """
    if not in_camera_workspace(cam_x, cam_z):
        return None
    xy = camera_xz_to_robot_xy(cam_x, cam_z)
    z = hover_z()
    if xy is None or z is None:
        return None
    return [xy[0], xy[1], z]


def enforce_hover(pose: list[float]) -> list[float]:
    """Copy of pose with Z on or above the hover plane.

    Raises if a caller tries to command anything below the typed corner height.
    """
    if len(pose) < 3:
        raise ValueError("pose needs at least x, y, z")
    floor = hover_z()
    if floor is None:
        raise RuntimeError(
            "No hover height yet. Run calibrate.py and enter X Y Z for each corner."
        )
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
