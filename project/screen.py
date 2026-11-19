"""Camera-window drawing for calibration and tracking.

The scripts decide what happened. This file only paints it: pad tint,
clicked corners, object contours, and the status lines.
"""

from __future__ import annotations

import os
import sys

if sys.platform.startswith("linux"):
    # opencv-python only ships the Qt xcb plugin; "wayland" in QT_QPA_PLATFORM
    # makes window creation fail. Run through XWayland instead.
    os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import numpy as np

from calibration import load_calibration, xz_inside
from realsense_overlay import blend_overlay
from safety import in_camera_workspace

AREA_COLOR = (0, 255, 255)  # BGR yellow: the calibrated pad outline
OTHER_COLOR = (255, 0, 255)  # BGR magenta: objects seen but not tracked
TEXT_COLOR = (0, 255, 0)


def _text(image, lines, scale=0.5, gap=22) -> None:
    y = 24
    for line in lines:
        cv2.putText(
            image, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, TEXT_COLOR, 2, cv2.LINE_AA
        )
        y += gap


def draw_calibrated_area(image):
    """Outline the four pad corners clicked in calibrate.py, numbered in click order."""
    calib = load_calibration()
    corners = calib.get("corners_px") if calib else None
    if not corners or len(corners) < 3:
        return image
    out = image.copy()
    pts = np.array(corners, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], True, AREA_COLOR, 2, cv2.LINE_AA)
    for i, (u, v) in enumerate(corners, start=1):
        cv2.circle(out, (int(u), int(v)), 5, AREA_COLOR, -1, cv2.LINE_AA)
        cv2.putText(
            out, str(i), (int(u) + 7, int(v) - 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, AREA_COLOR, 2, cv2.LINE_AA,
        )
    return out


def paint_workspace(image, depth_u16, depth_scale_m, intrinsics):
    """Tint the table pixels that lie inside the calibrated pad."""
    if intrinsics is None or not depth_scale_m:
        return image
    us = np.arange(depth_u16.shape[1], dtype=np.float32)[None, :]
    z = depth_u16.astype(np.float32) * float(depth_scale_m)
    x = (us - intrinsics["ppx"]) / intrinsics["fx"] * z
    inside = (depth_u16 > 0) & xz_inside(x, z)
    out = image.copy()
    tint = np.zeros_like(out)
    tint[:] = (0, 180, 0)
    blended = cv2.addWeighted(out, 0.6, tint, 0.4, 0)
    out[inside] = blended[inside]
    mask = (inside.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
    return out


def draw_status(overlay, det, target, note: str, others=()) -> None:
    """Draw tracked and other objects, then the status lines, on the left image."""
    for other in others:
        cv2.drawContours(overlay, [other.contour], -1, OTHER_COLOR, 1)
        cv2.putText(
            overlay, f"{other.distance_m:.2f} m", (int(other.u) + 8, int(other.v)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, OTHER_COLOR, 1, cv2.LINE_AA,
        )
    if det is not None:
        cv2.drawContours(overlay, [det.contour], -1, (0, 255, 0), 2)
        cv2.putText(
            overlay, f"TARGET {det.distance_m:.2f} m", (int(det.u) + 10, int(det.v) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA,
        )
        cv2.drawMarker(
            overlay,
            (int(det.u), int(det.v)),
            (0, 255, 0),
            cv2.MARKER_CROSS,
            16,
            2,
        )
        where = "INSIDE pad" if in_camera_workspace(det.cam_x, det.cam_z) else "OUTSIDE pad"
        lines = [
            f"{len(others) + 1} object(s), tracking the closest",
            f"object cam X={det.cam_x:.3f} Z={det.cam_z:.3f} m  "
            f"top {det.height_m * 1000:.0f} mm above table  {where}",
        ]
        if target is not None:
            lines.append(
                f"TCP  X={target[0]:.3f} Y={target[1]:.3f} Z={target[2]:.3f} m"
            )
        else:
            lines.append("TCP  none (outside the pad)")
    else:
        lines = ["no object", "TCP  none"]
    lines.append(note)
    _text(overlay, lines)


def draw_follow_view(color, depth, depth_scale_m, intrinsics, det, target, status, detections):
    """Side-by-side follow image: annotated color on the left, depth on the right."""
    scale = depth_scale_m if depth_scale_m else 0.001
    overlay, heat = blend_overlay(color, depth, 0.45, scale)
    if intrinsics is not None:
        overlay = paint_workspace(overlay, depth, scale, intrinsics)
        heat = paint_workspace(heat, depth, scale, intrinsics)
    overlay = draw_calibrated_area(overlay)
    heat = draw_calibrated_area(heat)
    others = [item for item in detections if item is not det]
    draw_status(overlay, det, target, status, others)
    return np.hstack([overlay, heat])


def draw_calibrate_view(color, depth, depth_scale_m, corners, phase, message):
    """Side-by-side calibration image with the clicked corners and the instructions."""
    scale = depth_scale_m if depth_scale_m else 0.001
    _overlay, heat = blend_overlay(color, depth, 0.45, scale)
    view_left = color.copy()
    if len(corners) >= 2:
        pts = np.array([[c["u"], c["v"]] for c in corners], dtype=np.int32)
        closed = len(corners) == 4
        cv2.polylines(view_left, [pts], closed, (0, 255, 0), 2)
    for i, corner in enumerate(corners, start=1):
        cv2.circle(view_left, (corner["u"], corner["v"]), 6, (0, 255, 0), -1)
        cv2.putText(
            view_left, str(i), (corner["u"] + 8, corner["v"] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
        )
    if phase == "corners":
        done = sum(1 for c in corners if "robot" in c)
        lines = [
            f"TCP over corner {min(done + 1, 4)}/4, then click it",
            "n next   u undo   q quit",
        ]
    else:
        lines = [
            "Click the object a few times",
            "s save   u undo   q quit",
        ]
    if message:
        lines.append(message)
    _text(view_left, lines, scale=0.55, gap=24)
    return np.hstack([view_left, heat])
