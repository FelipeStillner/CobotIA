"""PC loop: RealSense → object position → UR5 TCP above it.

The camera watches a 40 cm × 70 cm pad in front of it (camera in the
corner of the desk). This script finds the closest object on that pad
and, when --execute is set, servos the UR5 so the TCP stays directly
above that point on a plane 30 cm over the table.

Nothing below that plane is ever sent.

Preview (no robot motion):
    sudo ./venv/bin/python project/follow_object.py

URSim:
    sudo ./venv/bin/python project/follow_object.py --execute

Real robot (after editing project/config.py calibration):
    sudo ./venv/bin/python project/follow_object.py --execute --ip 192.168.0.10 --i-understand

Keys: q quit
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CALIBRATED, MAX_STEP_M, ROBOT_IP, SERVO_ACCEL, SERVO_SPEED
from detect_object import detect_object
from realsense_overlay import RealSense, RsError, blend_overlay
from safety import enforce_hover, hover_z, in_camera_workspace, limit_step, target_over_object

rtde_c = None
rtde_r = None


def disconnect():
    global rtde_c, rtde_r
    if rtde_c is not None:
        try:
            rtde_c.servoStop()
        except Exception:
            pass
    for name, client in (("control", rtde_c), ("receive", rtde_r)):
        if client is None:
            continue
        try:
            client.disconnect()
            print(f"Disconnected {name}")
        except Exception as exc:
            print(f"Error disconnecting {name}: {exc}")
    rtde_c = None
    rtde_r = None


def on_sigint(signum, frame):
    print("\nCtrl+C — stopping")
    disconnect()
    sys.exit(0)


def connect_robot(ip: str):
    global rtde_c, rtde_r
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface

    print(f"Connecting to UR5 at {ip} ...")
    rtde_c = RTDEControlInterface(ip)
    rtde_r = RTDEReceiveInterface(ip)
    print("Robot connected. TCP Z floor:", round(hover_z(), 3), "m")
    return rtde_c, rtde_r


def send_target(target_xyz: list[float]) -> str:
    """Servo toward target_xyz. Orientation stays as it is. Z is re-checked."""
    if rtde_r.isProtectiveStopped() or rtde_r.isEmergencyStopped():
        return "robot stopped (protective/emergency) — not moving"

    current = list(rtde_r.getActualTCPPose())
    x, y = limit_step((current[0], current[1]), (target_xyz[0], target_xyz[1]), MAX_STEP_M)
    pose = enforce_hover([x, y, target_xyz[2], current[3], current[4], current[5]])

    if not rtde_c.isPoseWithinSafetyLimits(pose):
        return "target outside UR safety limits — held"

    # time=0.1 s, lookahead=0.1 s, gain=200 — gentle tracking
    rtde_c.servoL(pose, SERVO_SPEED, SERVO_ACCEL, 0.1, 0.1, 200)
    return "tracking"


def draw_status(overlay, det, target, note: str):
    if det is not None:
        cv2.drawContours(overlay, [det.contour], -1, (0, 255, 0), 2)
        center = (int(det.u), int(det.v))
        cv2.drawMarker(overlay, center, (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
        inside = in_camera_workspace(det.cam_x, det.cam_z)
        where = "INSIDE pad" if inside else "OUTSIDE pad — robot holds"
        lines = [
            f"object cam X={det.cam_x:.3f} Z={det.cam_z:.3f} m  {where}",
        ]
        if target is not None:
            lines.append(
                f"TCP target  X={target[0]:.3f} Y={target[1]:.3f} Z={target[2]:.3f} m (hover)"
            )
        else:
            lines.append("TCP target  none (outside 40x70 cm pad)")
    else:
        lines = ["no object", "TCP target  none"]
    lines.append(note)
    y = 24
    for line in lines:
        cv2.putText(overlay, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        y += 22


def main() -> int:
    parser = argparse.ArgumentParser(description="Follow an object with the UR5 TCP at a 30 cm hover")
    parser.add_argument("--execute", action="store_true", help="Send poses to the robot")
    parser.add_argument("--ip", default=ROBOT_IP)
    parser.add_argument(
        "--i-understand",
        action="store_true",
        help="Required to move a robot that is not URSim (not 127.0.0.1)",
    )
    parser.add_argument(
        "--allow-uncalibrated",
        action="store_true",
        help="Allow --execute before CALIBRATED=True in config.py",
    )
    args = parser.parse_args()

    local = args.ip in ("127.0.0.1", "localhost", "::1")
    if args.execute and not local and not args.i_understand:
        print("Refusing to move a non-localhost robot without --i-understand")
        return 2
    if args.execute and not CALIBRATED and not args.allow_uncalibrated:
        print("Camera→robot XY is not calibrated (config.py CALIBRATED=False).")
        print("Preview still works. To move anyway (URSim): add --allow-uncalibrated")
        print("The 30 cm Z floor is still enforced either way.")
        return 2

    signal.signal(signal.SIGINT, on_sigint)

    if args.execute:
        try:
            connect_robot(args.ip)
        except Exception as exc:
            print("Robot connection failed:", exc)
            return 1
    else:
        print("Preview only — no motion. Pass --execute to drive the robot.")
        print(f"Hover plane Z = {hover_z():.3f} m (30 cm above the table)")

    cam = RealSense()
    try:
        cam.start()
    except RsError as exc:
        print(f"Camera failed: {exc}")
        print("On macOS: sudo ./venv/bin/python project/follow_object.py")
        disconnect()
        return 1

    win = "Follow — overlay | depth"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    note = "sending to robot" if args.execute else "preview (robot not moving)"

    try:
        while True:
            color, depth, _center_z = cam.wait_aligned()
            intr = cam.intrinsics
            scale = cam.depth_scale_m
            det = None
            target = None
            status = note
            if intr is not None and scale:
                det = detect_object(depth, scale, intr)
                if det is not None:
                    target = target_over_object(det.cam_x, det.cam_z)
                    if args.execute and target is not None:
                        try:
                            status = send_target(target)
                        except Exception as exc:
                            status = f"move rejected: {exc}"
                    elif args.execute:
                        status = "object outside pad — holding"
            overlay, heat = blend_overlay(color, depth, 0.45)
            draw_status(overlay, det, target, status)
            cv2.imshow(win, np.hstack([overlay, heat]))
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
