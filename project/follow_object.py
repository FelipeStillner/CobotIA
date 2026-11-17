"""Track the object with the UR5 TCP on the taught hover plane.

Preview (no motion):
    python project/follow_object.py

Move the robot (Remote Control, workspace clear):
    python project/follow_object.py --execute

q quits and stops the robot.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys

if sys.platform.startswith("linux"):
    # opencv-python only ships the Qt xcb plugin; "wayland" in QT_QPA_PLATFORM
    # makes window creation fail. Run through XWayland instead.
    os.environ["QT_QPA_PLATFORM"] = "xcb"
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration import xz_inside
from detect_object import detect_object
from realsense_overlay import RealSense, RsError, blend_overlay
from robot import Robot
from safety import hover_z, in_camera_workspace, target_over_object

robot = Robot()


def on_sigint(signum, frame):
    print("\nCtrl+C — stopping")
    robot.disconnect()
    raise SystemExit(0)


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


def draw_status(overlay, det, target, note: str) -> None:
    if det is not None:
        cv2.drawContours(overlay, [det.contour], -1, (0, 255, 0), 2)
        cv2.drawMarker(
            overlay,
            (int(det.u), int(det.v)),
            (0, 255, 0),
            cv2.MARKER_CROSS,
            16,
            2,
        )
        where = "INSIDE pad" if in_camera_workspace(det.cam_x, det.cam_z) else "OUTSIDE pad"
        lines = [f"object cam X={det.cam_x:.3f} Z={det.cam_z:.3f} m  {where}"]
        if target is not None:
            lines.append(
                f"TCP  X={target[0]:.3f} Y={target[1]:.3f} Z={target[2]:.3f} m"
            )
        else:
            lines.append("TCP  none (outside the pad)")
    else:
        lines = ["no object", "TCP  none"]
    lines.append(note)
    y = 24
    for line in lines:
        cv2.putText(
            overlay, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA
        )
        y += 22


def main() -> int:
    parser = argparse.ArgumentParser(description="Follow the object with the UR5")
    parser.add_argument("--execute", action="store_true", help="Send TCP poses to the robot")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, on_sigint)
    print(f"Hover plane Z = {hover_z():.4f} m (never lower)")
    note = "preview — robot not moving"
    if args.execute:
        try:
            robot.connect()
        except Exception as exc:
            print("Robot connection failed:", exc)
            return 1
        note = "sending to robot"

    cam = RealSense()
    try:
        cam.start()
    except RsError as exc:
        print(f"Camera failed: {exc}")
        robot.disconnect()
        return 1

    win = "Follow — overlay | depth"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    status = note
    try:
        while True:
            color, depth, _center_z = cam.wait_aligned()
            det = None
            target = None
            status = note
            if cam.intrinsics is not None and cam.depth_scale_m:
                det = detect_object(depth, cam.depth_scale_m, cam.intrinsics, color)
                if det is not None:
                    target = target_over_object(det.cam_x, det.cam_z)
                    if args.execute and target is not None:
                        try:
                            status = robot.track(target)
                        except Exception as exc:
                            status = f"move rejected: {exc}"
                    elif args.execute:
                        status = "outside pad — holding"
            scale = cam.depth_scale_m if cam.depth_scale_m else 0.001
            overlay, heat = blend_overlay(color, depth, 0.45, scale)
            if cam.intrinsics is not None:
                overlay = paint_workspace(overlay, depth, scale, cam.intrinsics)
                heat = paint_workspace(heat, depth, scale, cam.intrinsics)
            draw_status(overlay, det, target, status)
            cv2.imshow(win, np.hstack([overlay, heat]))
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        robot.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
