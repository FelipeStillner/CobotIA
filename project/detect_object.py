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

# Depth the detector accepts. The interface draws exactly this set of pixels.
DEPTH_NEAR_M = 0.15
DEPTH_FAR_M = 2.0


@dataclass
class Detection:
    u: float
    v: float
    cam_x: float
    cam_z: float
    contour: np.ndarray
    distance_m: float   # straight-line distance from the camera lens
    height_m: float     # top of the object above the table, meters


def deproject(u: float, v: float, depth_m: float, fx: float, fy: float, ppx: float, ppy: float):
    z = float(depth_m)
    x = (float(u) - ppx) / fx * z
    y = (float(v) - ppy) / fy * z
    return x, y, z


def table_pixel_mask(shape, calib, margin_px: int = 15) -> np.ndarray:
    """The clicked table corners, grown a little so the object on the edge is included.

    This is the table in the image. It does not include the robot standing
    above that rectangle.
    """
    mask = np.zeros(shape[:2], np.uint8)
    corners = calib.get("corners_px") if calib else None
    if not corners or len(corners) < 3:
        return mask.astype(bool)
    pts = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1)
    if margin_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (margin_px * 2 + 1, margin_px * 2 + 1)
        )
        mask = cv2.dilate(mask, kernel)
    return mask.astype(bool)


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


def _cam_points(rows, cols, depth_m, intrinsics):
    """Camera X, Y, Z for image pixels. Y points down, so higher is a smaller Y."""
    z = depth_m[rows, cols].astype(np.float64)
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cam_x = (cols.astype(np.float64) - float(intrinsics["ppx"])) / fx * z
    cam_y = (rows.astype(np.float64) - float(intrinsics["ppy"])) / fy * z
    return cam_x, cam_y, z


def _plane_from_points(cam_x, cam_y, cam_z):
    design = np.column_stack([cam_x, cam_z, np.ones(cam_x.shape[0])])
    coeff, *_ = np.linalg.lstsq(design, cam_y, rcond=None)
    residual = np.abs(design @ coeff - cam_y)
    return coeff, residual


def fit_table_plane(depth_m, valid, on_pad, intrinsics):
    """cam_y = a*cam_x + b*cam_z + c for the pad.

    Camera Y points down, so a higher point has a smaller cam_y.
    Two fits drop pixels more than 1.5 cm above the plane, so the object
    itself does not tilt the table. Returns None when the pad is not a
    clear plane.
    """
    table = valid & on_pad
    idx = np.flatnonzero(table)
    if idx.size < 200:
        return None
    step = max(1, idx.size // 2500)
    idx = idx[::step]
    rows, cols = np.divmod(idx, depth_m.shape[1])
    cam_x, cam_y, cam_z = _cam_points(rows, cols, depth_m, intrinsics)
    coeff, _residual = _plane_from_points(cam_x, cam_y, cam_z)
    keep = np.ones(cam_x.shape[0], dtype=bool)
    for _ in range(2):
        height = coeff[0] * cam_x + coeff[1] * cam_z + coeff[2] - cam_y
        keep = (height < 0.015) & (height > -0.04) & np.isfinite(height)
        if int(keep.sum()) < 80:
            return None
        coeff, residual = _plane_from_points(cam_x[keep], cam_y[keep], cam_z[keep])
    if float(np.median(residual)) > 0.02:
        return None
    return coeff


def _above_table(depth_m, selected, intrinsics, plane, min_height=0.008):
    """Pixels of `selected` that stand at least min_height above the table.

    None when too few pixels clear the table, so the caller keeps the
    color region (a flat object has no raised pixels).
    """
    if plane is None:
        return None
    rows, cols = np.nonzero(selected)
    if rows.size == 0:
        return None
    cam_x, cam_y, cam_z = _cam_points(rows, cols, depth_m, intrinsics)
    height = float(plane[0]) * cam_x + float(plane[1]) * cam_z + float(plane[2]) - cam_y
    keep = np.isfinite(height) & (height > min_height)
    if int(keep.sum()) < 30:
        return None
    out = np.zeros(selected.shape, dtype=bool)
    out[rows[keep], cols[keep]] = True
    return out


def height_above_table(cam_x: float, cam_y: float, cam_z: float, plane) -> float:
    """Meters the point sits above the fitted table. Zero when the plane is missing."""
    if plane is None:
        return 0.0
    table_y = float(plane[0]) * cam_x + float(plane[1]) * cam_z + float(plane[2])
    return float(np.clip(table_y - cam_y, 0.0, 0.50))


def height_of_mask(depth_m, selected, intrinsics, plane) -> float:
    """Height of the top of a region above the table plane.

    Each pixel is measured with its own depth. The result is the median of
    the upper quarter, so the table showing through the mask does not pull
    the height down to zero, and a few wild depth pixels do not set it.
    """
    if plane is None:
        return 0.0
    rows, cols = np.nonzero(selected)
    if rows.size < 20:
        return 0.0
    cam_x, cam_y, cam_z = _cam_points(rows, cols, depth_m, intrinsics)
    height = float(plane[0]) * cam_x + float(plane[1]) * cam_z + float(plane[2]) - cam_y
    height = height[np.isfinite(height)]
    if height.size < 20:
        return 0.0
    # Pixels on the object stand above the table. The table seen through the
    # mask stays near zero and must not be part of the average.
    above = height[height > 0.008]
    if above.size >= 15:
        band = above[above >= np.percentile(above, 80)]
        return float(np.clip(np.median(band), 0.0, 0.50))
    return float(np.clip(np.median(height), 0.0, 0.50))


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
    valid = (depth_m > DEPTH_NEAR_M) & (depth_m < DEPTH_FAR_M)
    if int(valid.sum()) < 200:
        return []

    calib = load_calibration()
    if not calib or color_bgr is None or "hue_center" not in calib:
        return []
    on_pad = workspace_mask(depth_u16, depth_scale_m, intrinsics)
    # On the table in the image, or standing above it. A taller object leaves
    # the clicked rectangle, but its ground position stays inside the pad.
    colored = valid & color_mask(color_bgr, calib)
    on_table = table_pixel_mask(depth_u16.shape, calib)
    foreground = colored & (on_table | on_pad)
    plane = fit_table_plane(depth_m, valid, on_pad, intrinsics)
    mask = (foreground.astype(np.uint8)) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        samples = (inside == 1) & valid
        if int(samples.sum()) < 20:
            continue
        rows, cols = np.nonzero(samples)
        cam_xs, cam_ys, cam_zs = _cam_points(rows, cols, depth_m, intrinsics)
        cam_x = float(np.median(cam_xs))
        cam_y = float(np.median(cam_ys))
        cam_z = float(np.median(cam_zs))
        if not bool(np.asarray(xz_inside(cam_x, cam_z)).reshape(-1)[0]):
            continue
        distance = float(np.sqrt(cam_x**2 + cam_y**2 + cam_z**2))
        height = height_of_mask(depth_m, samples, intrinsics, plane)
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
