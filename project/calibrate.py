"""Click each pad corner with the robot TCP held over it, then sample the object.

Each corner click reads the TCP pose straight from the robot over RTDE
(rtde_receive only: read-only, nothing is ever sent to the robot), so the
pose is in the Base frame that the motion code uses. The pendant's
Feature setting ("Vue", "Base", ...) does not matter.

    sudo ./.venv/bin/python project/calibrate.py               # robot at config.ROBOT_IP
    sudo ./.venv/bin/python project/calibrate.py 127.0.0.1     # URSim

1. Jog the TCP to the hover height (at least HOVER_ABOVE_TABLE_M above the pad) over corner 1,
   with the tool pointing down the way it should track. Let it stop.
2. Click corner 1 on the left image. The line shows where the click landed
   in the camera, and the tool X, Y read from the robot.
3. Jog the tool to the next corner and let it stop. Repeat around the edge,
   at the same height and tool orientation. The tool X, Y stay the same
   until the tool itself moves.
4. Press n.
5. Click the object a few times (the color is sampled around each click).
6. Press s to save.

u undoes the last click. q quits without saving.
Clicks count on the left image only.
"""

from __future__ import annotations

import argparse
import os
import sys

if sys.platform.startswith("linux"):
    # opencv-python only ships the Qt xcb plugin; "wayland" in QT_QPA_PLATFORM
    # makes window creation fail. Run through XWayland instead.
    os.environ["QT_QPA_PLATFORM"] = "xcb"
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration import circular_mean_hue, fit_camera_to_robot, save_calibration
from config import HOVER_ABOVE_TABLE_M, ROBOT_IP, TABLE_Z_ROBOT
from detect_object import deproject
from realsense_overlay import RealSense, RsError
from safety import orientation_error_rad
from screen import draw_calibrate_view

# Reject a corner if the TCP is still moving faster than this.
STILL_SPEED_M_S = 0.002
# Warn when the four corner orientations differ by more than this.
ORIENTATION_SPREAD_WARN_RAD = np.radians(5.0)
# Two corners closer than this are the same place. The map would then send
# every object to one robot X, Y.
MIN_CORNER_SEPARATION_M = 0.02
MIN_PAD_SPAN_M = 0.05

WIN = "Calibrate - click the left image"  # ASCII only: Qt highgui fails on non-ASCII names
state = {
    "phase": "corners",
    "corners": [],
    "colors": [],
    "color": None,
    "depth": None,
    "scale": None,
    "intrinsics": None,
    "message": "",
    "receive": None,
    "robot_ip": "",
}


def read_robot_pose() -> tuple[list[float] | None, str]:
    """Actual TCP pose [x, y, z, rx, ry, rz] in the Base frame, or None and why."""
    receive = state["receive"]
    try:
        if not receive.isConnected():
            receive.reconnect()
        pose = [float(v) for v in receive.getActualTCPPose()]
        speed = float(np.linalg.norm(receive.getActualTCPSpeed()[:3]))
    except Exception as exc:
        return None, f"robot read failed: {exc}"
    if len(pose) != 6:
        return None, "robot returned no pose"
    if speed > STILL_SPEED_M_S:
        return None, f"robot still moving ({speed * 1000:.0f} mm/s) - wait and click again"
    return pose, ""


def patch_median(image, u, v, radius, valid=None):
    h, w = image.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    y0, y1 = max(0, vi - radius), min(h, vi + radius + 1)
    x0, x1 = max(0, ui - radius), min(w, ui + radius + 1)
    patch = image[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    if valid is None:
        return np.median(patch.reshape(-1, patch.shape[-1]), axis=0)
    good = patch[valid[y0:y1, x0:x1]]
    if good.size == 0:
        return None
    return float(np.median(good))


def reset_calibration() -> None:
    """Clear the clicks so a new calibration can start."""
    state["phase"] = "corners"
    state["corners"] = []
    state["colors"] = []
    state["message"] = "Hold the TCP over corner 1, then click it"


def advance_calibration() -> None:
    """Move from the four corners to the object-color clicks."""
    if state["phase"] != "corners":
        return
    if len(state["corners"]) == 4:
        state["phase"] = "object"
        state["message"] = "click the object"
    else:
        state["message"] = "need 4 corners"


def _clicked_camera_xz(x: int, y: int) -> tuple[float, float] | None:
    """Camera X, Z (meters) of one click, from the depth at that pixel."""
    raw = patch_median(state["depth"], x, y, 4, valid=state["depth"] > 0)
    if raw is None:
        return None
    intr = state["intrinsics"]
    depth_m = raw * state["scale"]
    cam_x, _cam_y, cam_z = deproject(
        x, y, depth_m, intr["fx"], intr["fy"], intr["ppx"], intr["ppy"]
    )
    return cam_x, cam_z


def _separated(a, b, tol: float) -> bool:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 >= tol * tol


def click_at(x: int, y: int) -> None:
    """Record a corner or an object-color sample at a pixel on the left image."""
    color = state["color"]
    depth = state["depth"]
    if color is None or depth is None or state["intrinsics"] is None or not state["scale"]:
        state["message"] = "camera not ready"
        return
    if x < 0 or y < 0 or x >= color.shape[1] or y >= color.shape[0]:
        state["message"] = "click the left image"
        return
    point = _clicked_camera_xz(x, y)
    if point is None:
        state["message"] = "no depth there — click the table again"
        return
    cam_x, cam_z = point
    seen = f"camera X {cam_x * 1000:.0f} Z {cam_z * 1000:.0f} mm"
    if state["phase"] == "corners":
        if len(state["corners"]) >= 4:
            state["message"] = "4 corners already — press n"
            return
        for i, earlier in enumerate(state["corners"], start=1):
            if not _separated((cam_x, cam_z), (earlier["x"], earlier["z"]), MIN_CORNER_SEPARATION_M):
                state["message"] = (
                    f"Same place as corner {i} ({seen}). Click the next corner of the pad."
                )
                print(state["message"])
                return
        pose, why = read_robot_pose()
        if pose is None:
            state["message"] = why
            print(why)
            return
        tool = pose[:2]
        for i, earlier in enumerate(state["corners"], start=1):
            if not _separated(tool, earlier["robot"][:2], MIN_CORNER_SEPARATION_M):
                mm = [round(v * 1000.0, 1) for v in pose[:3]]
                where = state.get("robot_ip") or "the connected robot"
                state["message"] = (
                    f"Tool still at X {mm[0]:.0f} Y {mm[1]:.0f} mm on {where}. "
                    "Jog the tool to this corner, let it stop, then click."
                )
                print(state["message"])
                print(f"           click measured {seen}")
                return
        state["corners"].append(
            {"u": x, "v": y, "x": cam_x, "z": cam_z, "robot": pose[:3], "orientation": pose[3:]}
        )
        n = len(state["corners"])
        mm = [round(v * 1000.0, 1) for v in pose[:3]]
        height = ""
        if TABLE_Z_ROBOT is not None:
            height = f", {(pose[2] - TABLE_Z_ROBOT) * 1000:.0f} mm above the table"
        print(
            f"corner {n}/4: {seen}, pixel ({x}, {y}) | "
            f"tool X {mm[0]} Y {mm[1]} Z {mm[2]} mm (Base){height}"
        )
        print(f"           orientation {[round(v, 4) for v in pose[3:]]} rad")
        state["message"] = (
            f"corner {n}/4  {seen}   tool X {mm[0]:.0f} Y {mm[1]:.0f} mm"
            + (" — press n" if n == 4 else "")
        )
        return

    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    sample = patch_median(hsv, x, y, 6)
    if sample is None:
        return
    state["colors"].append([float(sample[0]), float(sample[1]), float(sample[2])])
    state["message"] = f"object samples: {len(state['colors'])}  {seen} — press s to save"


def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    click_at(x, y)


def undo():
    if state["phase"] == "object" and state["colors"]:
        state["colors"].pop()
        state["message"] = f"object samples: {len(state['colors'])}"
        return
    if state["corners"]:
        state["corners"].pop()
        state["phase"] = "corners"
        state["message"] = f"corner {len(state['corners'])}/4"


def save_from_clicks() -> bool:
    corners = state["corners"]
    if len(corners) != 4 or any("robot" not in c for c in corners) or not state["colors"]:
        state["message"] = "need 4 corners and at least one object click"
        return False
    hues = [c[0] for c in state["colors"]]
    sats = [c[1] for c in state["colors"]]
    vals = [c[2] for c in state["colors"]]
    corners_xz = [[c["x"], c["z"]] for c in corners]
    corners_robot = [c["robot"] for c in corners]
    cam_span_x = max(p[0] for p in corners_xz) - min(p[0] for p in corners_xz)
    cam_span_z = max(p[1] for p in corners_xz) - min(p[1] for p in corners_xz)
    tool_span_x = max(p[0] for p in corners_robot) - min(p[0] for p in corners_robot)
    tool_span_y = max(p[1] for p in corners_robot) - min(p[1] for p in corners_robot)
    if cam_span_x < MIN_PAD_SPAN_M or cam_span_z < MIN_PAD_SPAN_M:
        state["message"] = (
            "The four clicks landed on the same spot in the camera. "
            "Click the four corners of the pad."
        )
        print(state["message"])
        return False
    if tool_span_x < MIN_PAD_SPAN_M or tool_span_y < MIN_PAD_SPAN_M:
        state["message"] = (
            "The tool stayed in one place, so every object would get the same X and Y. "
            "Jog the tool to each corner and calibrate again."
        )
        print(state["message"])
        return False
    rotation, translation, error_m, hover = fit_camera_to_robot(corners_xz, corners_robot)
    orientations = [c["orientation"] for c in corners]
    spread = max(
        orientation_error_rad(a, b) for a in orientations for b in orientations
    )
    data = {
        # Pixel corners: screen.py draws the calibrated area from these.
        "corners_px": [[c["u"], c["v"]] for c in corners],
        "corners_xz": corners_xz,
        "cam_xz_to_robot_R": rotation.tolist(),
        "cam_xz_to_robot_t": translation.tolist(),
        "hover_z": hover,
        # Tool orientation for tracking: corner 1's, read from the robot.
        "tcp_orientation": orientations[0],
        "hue_center": circular_mean_hue(hues),
        "hue_tol": 12,
        "s_low": int(max(0, np.median(sats) - 50)),
        "v_low": int(max(0, np.median(vals) - 60)),
    }
    save_calibration(data)
    print("Saved", Path(__file__).resolve().parent / "calibration.json")
    if TABLE_Z_ROBOT is None:
        print(f"  hover Z = {hover * 1000:.1f} mm; TABLE_Z_ROBOT not measured yet (config.py)")
        above = None
    else:
        above = hover - TABLE_Z_ROBOT
        print(f"  hover Z = {hover * 1000:.1f} mm, {above * 1000:.1f} mm above the table")
    if above is not None and above < HOVER_ABOVE_TABLE_M:
        print(
            f"  WARNING: below the {HOVER_ABOVE_TABLE_M * 1000:.0f} mm minimum; "
            "follow_object.py will refuse to move. Check the Z values or TABLE_Z_ROBOT."
        )
    z_spread = (max(c[2] for c in corners_robot) - min(c[2] for c in corners_robot)) * 1000.0
    if z_spread > 20.0:
        print(f"  the four Z values differ by {z_spread:.1f} mm; hover uses their average")
    print(f"  tool orientation {[round(v, 4) for v in orientations[0]]} rad (corner 1)")
    if spread > ORIENTATION_SPREAD_WARN_RAD:
        print(
            f"  WARNING: the corner orientations differ by up to {np.degrees(spread):.1f} deg; "
            "tracking uses corner 1's"
        )
    for i, err in enumerate(error_m, start=1):
        robot = corners_robot[i - 1]
        print(
            f"  corner {i}: robot {robot[0] * 1000:.1f} {robot[1] * 1000:.1f} "
            f"{robot[2] * 1000:.1f} mm, fit error {err * 1000:.1f} mm"
        )
    if max(error_m) > 0.015:
        print(
            f"  WARNING: fit error up to {max(error_m) * 1000:.0f} mm. Check that each click "
            "matched the corner the TCP was over, in the same order. Re-calibrate before tracking."
        )
    print(f"  hue {data['hue_center']:.1f} ± {data['hue_tol']}, S>={data['s_low']}, V>={data['v_low']}")
    state["message"] = "saved calibration.json"
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate the pad, the robot map and the object color")
    parser.add_argument("ip", nargs="?", default=ROBOT_IP, help="robot IP (read-only connection)")
    args = parser.parse_args()

    from rtde_receive import RTDEReceiveInterface

    state["robot_ip"] = args.ip
    print(f"Connecting to the robot at {args.ip} (read-only, rtde_receive) ...")
    try:
        state["receive"] = RTDEReceiveInterface(args.ip)
        pose = state["receive"].getActualTCPPose()
    except Exception as exc:
        print("Robot connection failed:", exc)
        return 1
    print("Connected. TCP (Base)", [round(v, 4) for v in pose])
    print("Hold the TCP over a corner, then click that corner on the left image.")

    cam = RealSense()
    try:
        cam.start()
    except RsError as exc:
        print(f"Camera failed: {exc}")
        print("On macOS: sudo ./.venv/bin/python project/calibrate.py")
        state["receive"].disconnect()
        return 1

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, on_mouse)
    try:
        while True:
            color, depth, _center = cam.wait_aligned()
            state["color"] = color
            state["depth"] = depth
            state["scale"] = cam.depth_scale_m if cam.depth_scale_m else 0.001
            state["intrinsics"] = cam.intrinsics
            cv2.imshow(WIN, draw_calibrate_view(
                color, depth, state["scale"], state["corners"], state["phase"], state["message"],
            ))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quit without saving.")
                break
            if key == ord("u"):
                undo()
            if key == ord("n"):
                advance_calibration()
            if key == ord("s") and save_from_clicks():
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        try:
            state["receive"].disconnect()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
