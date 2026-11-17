"""Find the nearest object on the desk from an aligned depth frame.

The object is the largest region that is closer to the camera than the
background (the table). Its 3D point is in the RealSense camera frame:
X right, Y down, Z forward, meters.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import CLOSER_THAN_M, MIN_OBJECT_AREA_PX


@dataclass
class Detection:
    u: float
    v: float
    depth_m: float
    cam_x: float
    cam_y: float
    cam_z: float
    contour: np.ndarray


def deproject(u: float, v: float, depth_m: float, fx: float, fy: float, ppx: float, ppy: float):
    z = float(depth_m)
    x = (float(u) - ppx) / fx * z
    y = (float(v) - ppy) / fy * z
    return x, y, z


def detect_object(depth_u16: np.ndarray, depth_scale_m: float, intrinsics: dict) -> Detection | None:
    if depth_scale_m <= 0:
        return None
    depth_m = depth_u16.astype(np.float32) * float(depth_scale_m)
    valid = (depth_m > 0.15) & (depth_m < 2.0)
    if int(valid.sum()) < 200:
        return None

    background = float(np.percentile(depth_m[valid], 70))
    foreground = valid & (depth_m < background - CLOSER_THAN_M)
    mask = (foreground.astype(np.uint8)) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < MIN_OBJECT_AREA_PX:
        return None

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    u = moments["m10"] / moments["m00"]
    v = moments["m01"] / moments["m00"]

    inside = np.zeros(depth_m.shape, dtype=np.uint8)
    cv2.drawContours(inside, [contour], -1, 1, thickness=-1)
    samples = depth_m[(inside == 1) & valid]
    if samples.size == 0:
        return None
    z = float(np.median(samples))
    cam_x, cam_y, cam_z = deproject(
        u, v, z, intrinsics["fx"], intrinsics["fy"], intrinsics["ppx"], intrinsics["ppy"]
    )
    return Detection(u, v, z, cam_x, cam_y, cam_z, contour)
