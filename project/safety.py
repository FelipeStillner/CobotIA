"""Pad check and the pose sent to the robot.

Z is locked to the average of the four corner heights typed in calibrate.py,
and that height must be at least HOVER_ABOVE_TABLE_M above TABLE_Z_ROBOT.
"""

from __future__ import annotations

import numpy as np

import config
from calibration import load_calibration, xz_inside
from config import HOVER_ABOVE_TABLE_M, TABLE_Z_ROBOT


def clamp_speed(speed_m_s: float) -> float:
    """Tracking speed inside 0.03–0.30 m/s."""
    return float(min(config.SPEED_MAX_M_S, max(config.SPEED_MIN_M_S, float(speed_m_s))))


def clamp_height(meters: float) -> float:
    """A height setting inside 10–50 cm."""
    return float(min(config.HEIGHT_MAX_M, max(config.HEIGHT_MIN_M, float(meters))))


def tracking_speed() -> float:
    """Speed used for follow steps, after the safety clamp."""
    return clamp_speed(config.MAX_SPEED_M_S)


def set_tracking_speed(speed_m_s: float) -> float:
    config.MAX_SPEED_M_S = clamp_speed(speed_m_s)
    return config.MAX_SPEED_M_S


def clearance_m() -> float:
    """Meters kept above the object center, after the safety clamp."""
    return clamp_height(config.CLEARANCE_M)


def set_clearance(meters: float) -> float:
    config.CLEARANCE_M = clamp_height(meters)
    return config.CLEARANCE_M


def min_tcp_z() -> float:
    """Lowest base Z that may be commanded, after the safety clamp."""
    return clamp_height(config.MIN_TCP_Z)


def set_min_tcp_z(meters: float) -> float:
    config.MIN_TCP_Z = clamp_height(meters)
    return config.MIN_TCP_Z


def approach_speed_m_s() -> float:
    """Straight-line approach speed, never above the tracking cap."""
    return clamp_speed(min(config.APPROACH_SPEED, config.SPEED_MAX_M_S))


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
    """Base Z one clearance above the top of the object, never below the floor.

    height_m is how far the top of the object sits above the table.
    """
    if TABLE_Z_ROBOT is None:
        raise RuntimeError(
            "TABLE_Z_ROBOT is not measured. Touch the TCP to the pad, read Z on the "
            "pendant with Feature = Base, and set it in config.py."
        )
    z = float(TABLE_Z_ROBOT) + max(0.0, float(height_m)) + clearance_m()
    return max(z, min_tcp_z())


def target_over_object(cam_x: float, cam_z: float, height_m: float = 0.0) -> list[float] | None:
    """Robot [x, y, z] one clearance above the top of the object, or None outside the pad.

    Z follows the object's height above the table and never goes below the floor.
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
    """Copy of pose. Raises if Z is below the clamped floor."""
    if len(pose) < 3:
        raise ValueError("pose needs at least x, y, z")
    floor = min_tcp_z()
    if pose[2] < floor - 1e-6:
        raise RuntimeError(
            f"Refusing Z={pose[2]:.4f} m — below the floor (Z >= {floor:.4f} m)"
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
