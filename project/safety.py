"""Pad check and the pose sent to the robot.

Z is locked to the average of the four corner heights typed in calibrate.py,
and that height must be at least HOVER_ABOVE_TABLE_M above TABLE_Z_ROBOT.
"""

from __future__ import annotations

import numpy as np

from calibration import load_calibration, xz_inside
from config import HOVER_ABOVE_TABLE_M, MIN_TCP_Z, TABLE_Z_ROBOT


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


def tcp_orientation() -> list[float]:
    """Tool rotation vector for tracking, read from the robot by calibrate.py."""
    calib = load_calibration()
    rv = calib.get("tcp_orientation") if calib else None
    if rv is None or len(rv) != 3:
        raise RuntimeError(
            "No tool orientation in calibration.json. Re-run calibrate.py "
            "(it reads the corner poses from the robot)."
        )
    return [float(v) for v in rv]


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


def tcp_z_for_height(height_m: float) -> float:
    """Base Z 20 cm above the object center, never below MIN_TCP_Z.

    height_m is how far the detected center sits above the table.
    """
    if TABLE_Z_ROBOT is None:
        raise RuntimeError(
            "TABLE_Z_ROBOT is not measured. Touch the TCP to the pad, read Z on the "
            "pendant with Feature = Base, and set it in config.py."
        )
    z = float(TABLE_Z_ROBOT) + max(0.0, float(height_m)) + HOVER_ABOVE_TABLE_M
    return max(z, MIN_TCP_Z)


def target_over_object(cam_x: float, cam_z: float, height_m: float = 0.0) -> list[float] | None:
    """Robot [x, y, z] 20 cm above the object center, or None outside the pad.

    Z follows the center's height above the table and never goes below MIN_TCP_Z.
    """
    if not in_camera_workspace(cam_x, cam_z):
        return None
    xy = camera_xz_to_robot_xy(cam_x, cam_z)
    if xy is None or hover_z() is None:
        return None
    return [xy[0], xy[1], tcp_z_for_height(height_m)]


def table_floor_z() -> float:
    """Table plus 20 cm. The saved hover plane must clear this before the approach."""
    if TABLE_Z_ROBOT is None:
        raise RuntimeError(
            "TABLE_Z_ROBOT is not measured. Touch the TCP to the pad, read Z on the "
            "pendant with Feature = Base, and set it in config.py."
        )
    return TABLE_Z_ROBOT + HOVER_ABOVE_TABLE_M


def check_hover_plane() -> float:
    """The hover Z, after checking it clears the table by HOVER_ABOVE_TABLE_M.

    Raises if calibrate.py has not saved a height, or if the typed corner
    heights put the hover plane too close to the table.
    """
    z = hover_z()
    if z is None:
        raise RuntimeError(
            "No hover height yet. Run calibrate.py and enter X Y Z for each corner."
        )
    floor = table_floor_z()
    if z < floor - 1e-6:
        raise RuntimeError(
            f"Hover Z={z:.4f} m is only {z - TABLE_Z_ROBOT:.3f} m above the table "
            f"(TABLE_Z_ROBOT={TABLE_Z_ROBOT:.4f}); need {HOVER_ABOVE_TABLE_M:.2f} m. "
            "Re-check the corner Z values in calibrate.py or TABLE_Z_ROBOT in config.py."
        )
    return z


def enforce_hover(pose: list[float]) -> list[float]:
    """Copy of pose. Raises if Z is below MIN_TCP_Z."""
    if len(pose) < 3:
        raise ValueError("pose needs at least x, y, z")
    if pose[2] < MIN_TCP_Z - 1e-6:
        raise RuntimeError(
            f"Refusing Z={pose[2]:.4f} m — below the floor (Z >= {MIN_TCP_Z:.4f} m)"
        )
    return list(pose)


def limit_step(current, target, max_step: float) -> tuple[float, ...]:
    """Move at most max_step meters from current toward target.

    Works for XY or XYZ; the limit is on the straight-line distance.
    """
    cur = np.asarray(current, dtype=float)
    tgt = np.asarray(target, dtype=float)
    delta = tgt - cur
    dist = float(np.linalg.norm(delta))
    if dist <= max_step or dist == 0.0:
        return tuple(float(v) for v in tgt)
    return tuple(float(v) for v in cur + delta * (max_step / dist))


def _rotvec_to_matrix(rotvec) -> np.ndarray:
    r = np.asarray(rotvec, dtype=float)
    angle = float(np.linalg.norm(r))
    if angle < 1e-12:
        return np.eye(3)
    k = r / angle
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * (kx @ kx)


def orientation_error_rad(rotvec_a, rotvec_b) -> float:
    """Angle between two UR rotation vectors (axis-angle), radians."""
    rel = _rotvec_to_matrix(rotvec_a).T @ _rotvec_to_matrix(rotvec_b)
    cos = (float(np.trace(rel)) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))
