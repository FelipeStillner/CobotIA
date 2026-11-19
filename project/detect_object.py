"""Find the calibrated objects inside the clicked pad.

detect_objects() returns every region of the calibrated color inside the pad,
closest to the camera first. ClosestObjectPicker chooses which one to track.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from calibration import load_calibration, xz_inside
from config import MIN_OBJECT_AREA_PX, OBJECT_MATCH_PX, OBJECT_SWITCH_MARGIN_M


@dataclass
class Detection:
    u: float
    v: float
    cam_x: float
    cam_z: float
    contour: np.ndarray
    distance_m: float   # straight-line distance from the camera lens
    height_m: float     # center's height above the table, meters


def deproject(u: float, v: float, depth_m: float, fx: float, fy: float, ppx: float, ppy: float):
    z = float(depth_m)
    x = (float(u) - ppx) / fx * z
    y = (float(v) - ppy) / fy * z
    return x, y, z


def workspace_mask(depth_u16: np.ndarray, depth_scale_m: float, intrinsics: dict) -> np.ndarray:
    us = np.arange(depth_u16.shape[1], dtype=np.float32)[None, :]
    z = depth_u16.astype(np.float32) * float(depth_scale_m)
    x = (us - intrinsics["ppx"]) / intrinsics["fx"] * z
    return (depth_u16 > 0) & xz_inside(x, z)


def color_mask(color_bgr: np.ndarray, calib: dict) -> np.ndarray:
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.int16)
    delta = np.abs(hue - int(round(calib["hue_center"])))
    delta = np.minimum(delta, 180 - delta)
    return (
        (delta <= int(calib["hue_tol"]))
        & (hsv[:, :, 1] >= int(calib["s_low"]))
        & (hsv[:, :, 2] >= int(calib["v_low"]))
    )


def fit_table_plane(depth_m, valid, on_pad, object_u8, intrinsics):
    """cam_y = a*cam_x + b*cam_z + c for pad pixels that are not the object.

    Camera Y points down, so a higher point has a smaller cam_y.
    Returns None when the table is not a clear plane.
    """
    table = valid & on_pad & (object_u8 == 0)
    idx = np.flatnonzero(table)
    if idx.size < 200:
        return None
    step = max(1, idx.size // 2500)
    idx = idx[::step]
    width = depth_m.shape[1]
    rows, cols = np.divmod(idx, width)
    z = depth_m[rows, cols].astype(np.float64)
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cam_x = (cols.astype(np.float64) - float(intrinsics["ppx"])) / fx * z
    cam_y = (rows.astype(np.float64) - float(intrinsics["ppy"])) / fy * z
    design = np.column_stack([cam_x, z, np.ones(cam_x.shape[0])])
    coeff, *_ = np.linalg.lstsq(design, cam_y, rcond=None)
    if float(np.median(np.abs(design @ coeff - cam_y))) > 0.02:
        return None
    return coeff


def height_above_table(cam_x: float, cam_y: float, cam_z: float, plane) -> float:
    """Meters the point sits above the fitted table. Zero when the plane is missing."""
    if plane is None:
        return 0.0
    table_y = float(plane[0]) * cam_x + float(plane[1]) * cam_z + float(plane[2])
    return float(np.clip(table_y - cam_y, 0.0, 0.50))


def detect_objects(
    depth_u16: np.ndarray,
    depth_scale_m: float,
    intrinsics: dict,
    color_bgr: np.ndarray | None = None,
) -> list[Detection]:
    """Every object of the calibrated color inside the pad, closest to the camera first."""
    if depth_scale_m <= 0:
        return []
    depth_m = depth_u16.astype(np.float32) * float(depth_scale_m)
    valid = (depth_m > 0.15) & (depth_m < 2.0)
    if int(valid.sum()) < 200:
        return []

    calib = load_calibration()
    if not calib or color_bgr is None or "hue_center" not in calib:
        return []
    on_pad = workspace_mask(depth_u16, depth_scale_m, intrinsics)
    foreground = valid & on_pad & color_mask(color_bgr, calib)
    mask = (foreground.astype(np.uint8)) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    plane = fit_table_plane(depth_m, valid, on_pad, mask, intrinsics)
    found = []
    inside = np.zeros(depth_m.shape, dtype=np.uint8)
    for contour in contours:
        if cv2.contourArea(contour) < MIN_OBJECT_AREA_PX:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        u = moments["m10"] / moments["m00"]
        v = moments["m01"] / moments["m00"]

        inside[:] = 0
        cv2.drawContours(inside, [contour], -1, 1, thickness=-1)
        samples = depth_m[(inside == 1) & valid]
        if samples.size == 0:
            continue
        z = float(np.median(samples))
        cam_x, cam_y, cam_z = deproject(
            u, v, z, intrinsics["fx"], intrinsics["fy"], intrinsics["ppx"], intrinsics["ppy"]
        )
        distance = float(np.sqrt(cam_x**2 + cam_y**2 + cam_z**2))
        height = height_above_table(cam_x, cam_y, cam_z, plane)
        found.append(Detection(u, v, cam_x, cam_z, contour, distance, height))
    found.sort(key=lambda d: d.distance_m)
    return found


def detect_object(
    depth_u16: np.ndarray,
    depth_scale_m: float,
    intrinsics: dict,
    color_bgr: np.ndarray | None = None,
) -> Detection | None:
    """The object closest to the camera, or None."""
    found = detect_objects(depth_u16, depth_scale_m, intrinsics, color_bgr)
    return found[0] if found else None


class ClosestObjectPicker:
    """Chooses the object to track: the closest to the camera.

    The object tracked last frame is kept unless another one is at least
    OBJECT_SWITCH_MARGIN_M closer, so two objects at about the same distance
    do not make the target flip between them on depth noise.
    """

    def __init__(self, margin_m: float = OBJECT_SWITCH_MARGIN_M, match_px: float = OBJECT_MATCH_PX):
        self.margin_m = margin_m
        self.match_px = match_px
        self._last: tuple[float, float] | None = None   # pixel of last frame's target

    def reset(self) -> None:
        self._last = None

    def pick(self, detections: list[Detection]) -> Detection | None:
        if not detections:
            self._last = None
            return None
        closest = min(detections, key=lambda d: d.distance_m)
        chosen = closest
        if self._last is not None:
            lu, lv = self._last
            previous = min(detections, key=lambda d: (d.u - lu) ** 2 + (d.v - lv) ** 2)
            near = np.hypot(previous.u - lu, previous.v - lv) <= self.match_px
            if near and previous.distance_m - closest.distance_m < self.margin_m:
                chosen = previous
        self._last = (chosen.u, chosen.v)
        return chosen
