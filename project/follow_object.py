"""Track the object by sending servoL to the UR5.

The robot must be in Remote Control and the workspace clear.

    sudo ./.venv/bin/python project/follow_object.py                 # URSim at 127.0.0.1
    sudo ./.venv/bin/python project/follow_object.py --real          # real UR5 at config.ROBOT_IP
    sudo ./.venv/bin/python project/follow_object.py --real 10.0.0.5 # real UR5 at another IP

Before tracking, the TCP makes one slow straight-line move onto the hover
plane with the fixed tool orientation; you confirm it in the terminal.
Tracking then moves at most the clamped speed, in X, Y, and Z.

q or Ctrl+C stops the robot and disconnects.
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import sys
import time

if sys.platform.startswith("linux"):
    # opencv-python only ships the Qt xcb plugin; "wayland" in QT_QPA_PLATFORM
    # makes window creation fail. Run through XWayland instead.
    os.environ["QT_QPA_PLATFORM"] = "xcb"
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import HOVER_ABOVE_TABLE_M, ROBOT_IP, TABLE_Z_ROBOT
from detect_object import ClosestObjectPicker, detect_objects
from realsense_overlay import RealSense, RsError
from robot import Robot
from safety import (
    check_hover_plane,
    has_robot_map,
    in_camera_workspace,
    target_over_object,
    tcp_orientation,
    tracking_speed,
)
from screen import draw_follow_view

robot = Robot()


def on_sigint(signum, frame):
    print("\nCtrl+C — stopping")
    robot.disconnect()
    raise SystemExit(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Follow the object with the UR5 TCP")
    parser.add_argument(
        "ip",
        nargs="?",
        help=f"robot IP (default: 127.0.0.1, or {ROBOT_IP} with --real)",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help=f"drive the real robot (default IP {ROBOT_IP}); without it only URSim on localhost is allowed",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ip = args.ip or (ROBOT_IP if args.real else "127.0.0.1")
    if ip not in ("127.0.0.1", "localhost", "::1") and not args.real:
        print(f"{ip} is not URSim on localhost. Add --real to drive the real robot.")
        return 1
    robot.ip = ip
    signal.signal(signal.SIGINT, on_sigint)
    if not has_robot_map():
        print("No robot corners yet. Run calibrate.py and type X Y Z for each corner.")
        return 1
    try:
        z_hover = check_hover_plane()
        tcp_orientation()
    except RuntimeError as exc:
        print(exc)
        return 1
    print(
        f"Hover plane Z = {z_hover:.4f} m, {z_hover - TABLE_Z_ROBOT:.3f} m above the "
        f"table (need >= {HOVER_ABOVE_TABLE_M:.2f}); tracking speed <= {tracking_speed() * 100:.1f} cm/s"
    )
    # Camera and window first: if either fails (e.g. no display access under
    # sudo, which aborts the whole process inside Qt), the robot has not moved.
    cam = RealSense()
    try:
        cam.start()
    except RsError as exc:
        print(f"Camera failed: {exc}")
        return 1
    win = "Follow - overlay | depth"  # ASCII only: Qt highgui fails on non-ASCII names
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    color, depth, _center_z = cam.wait_aligned()
    cv2.imshow(win, draw_follow_view(
        color, depth, cam.depth_scale_m, cam.intrinsics, None, None, "opening camera", [],
    ))
    cv2.waitKey(1)

    try:
        robot.connect()
    except Exception as exc:
        print("Robot connection failed:", exc)
        cam.stop()
        cv2.destroyAllWindows()
        return 1

    try:
        print("Approach: one straight-line moveL onto the hover plane")
        for line in robot.describe_approach():
            print(line)
        if input("Workspace clear? Type yes to move: ").strip().lower() != "yes":
            print("Not moving.")
            raise SystemExit
        robot.approach()
    except (Exception, SystemExit) as exc:
        if not isinstance(exc, SystemExit):
            print("Approach stopped:", exc)
        robot.disconnect()
        cam.stop()
        cv2.destroyAllWindows()
        return 1
    holding = False

    log_dir = Path(__file__).resolve().parent / "captures"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / time.strftime("track_%Y%m%d_%H%M%S.csv")
    log_file = open(log_path, "w", newline="")
    log = csv.writer(log_file)
    log.writerow([
        "t_s", "status", "n_objects", "det_u", "det_v", "cam_x", "cam_z", "distance_m", "inside_pad",
        "target_x", "target_y", "actual_x", "actual_y", "actual_z", "sent_x", "sent_y", "sent_z",
    ])
    print(f"Logging every frame to {log_path}")
    t_start = time.monotonic()
    last_status = None
    picker = ClosestObjectPicker()

    try:
        while True:
            color, depth, _center_z = cam.wait_aligned()
            det = None
            detections = []
            target = None
            inside = None
            robot.last_actual = robot.last_sent = None
            status = "no object - not moving"
            if cam.intrinsics is not None and cam.depth_scale_m:
                detections = detect_objects(depth, cam.depth_scale_m, cam.intrinsics, color)
                det = picker.pick(detections)
                if det is not None:
                    inside = in_camera_workspace(det.cam_x, det.cam_z)
                    target = (
                        target_over_object(det.cam_x, det.cam_z, det.height_m) if inside else None
                    )
                    if target is not None:
                        holding = False
                        try:
                            status = robot.track(target)
                        except Exception as exc:
                            status = f"move rejected: {exc}"
                            robot.hold()
                    elif not holding:
                        holding = True
                        robot.hold()
                        print("holding — outside pad")
                        status = "outside pad — holding"
                    else:
                        status = "outside pad — holding"
            frame = draw_follow_view(
                color, depth, cam.depth_scale_m, cam.intrinsics,
                det, target, status, detections,
            )
            kind = status.split(" (")[0]
            if kind != last_status:
                print(f"[{time.monotonic() - t_start:7.1f} s] {status}")
                last_status = kind
            actual = robot.last_actual or [None] * 3
            sent = robot.last_sent or [None] * 3
            log.writerow([
                round(time.monotonic() - t_start, 3), status, len(detections),
                det.u if det else None, det.v if det else None,
                det.cam_x if det else None, det.cam_z if det else None,
                round(det.distance_m, 4) if det else None, inside,
                target[0] if target else None, target[1] if target else None,
                actual[0], actual[1], actual[2], sent[0], sent[1], sent[2],
            ])
            cv2.imshow(win, frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        log_file.close()
        cam.stop()
        cv2.destroyAllWindows()
        robot.disconnect()
        print(f"Log: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
