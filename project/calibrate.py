"""Click each pad corner, type its robot X Y Z, then sample the object.

Nothing is sent to the robot.

    sudo ./venv/bin/python project/calibrate.py

1. Click corner 1 on the left image.
2. In this terminal, type that corner's pendant X Y Z in millimeters.
3. Repeat for the other 3 corners, in order around the edge.
4. Press n.
5. Click the object a few times (the color is sampled around each click).
6. Press s to save.

u undoes the last click. q quits without saving.
Clicks count on the left image only.
"""

from __future__ import annotations

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
from detect_object import deproject
from realsense_overlay import RealSense, RsError, blend_overlay

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
}


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


def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    color = state["color"]
    depth = state["depth"]
    if color is None or depth is None or state["intrinsics"] is None or not state["scale"]:
        state["message"] = "camera not ready"
        return
    if x >= color.shape[1]:
        state["message"] = "click the left image"
        return
    intr = state["intrinsics"]
    if state["phase"] == "corners":
        if len(state["corners"]) >= 4:
            state["message"] = "4 corners already — press n"
            return
        raw = patch_median(depth, x, y, 4, valid=depth > 0)
        if raw is None:
            state["message"] = "no depth there — click the table again"
            return
        depth_m = raw * state["scale"]
        cam_x, _cam_y, cam_z = deproject(x, y, depth_m, intr["fx"], intr["fy"], intr["ppx"], intr["ppy"])
        state["corners"].append({"u": x, "v": y, "x": cam_x, "z": cam_z})
        state["message"] = f"corner {len(state['corners'])}/4 — type X Y Z in the terminal"
        return

    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    sample = patch_median(hsv, x, y, 6)
    if sample is None:
        return
    state["colors"].append([float(sample[0]), float(sample[1]), float(sample[2])])
    state["message"] = f"object samples: {len(state['colors'])} — press s to save"


def draw(color, heat):
    view_left = color.copy()
    corners = state["corners"]
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
    if state["phase"] == "corners":
        done = sum(1 for c in corners if "robot" in c)
        lines = [
            f"Click corner {min(done + 1, 4)}/4, then type its robot X Y Z",
            "n next   u undo   q quit",
        ]
    else:
        lines = [
            "Click the object a few times",
            "s save   u undo   q quit",
        ]
    if state["message"]:
        lines.append(state["message"])
    y = 24
    for line in lines:
        cv2.putText(view_left, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        y += 24
    return np.hstack([view_left, heat])


def undo():
    if state["phase"] == "object" and state["colors"]:
        state["colors"].pop()
        state["message"] = f"object samples: {len(state['colors'])}"
        return
    if state["corners"]:
        state["corners"].pop()
        state["phase"] = "corners"
        state["message"] = f"corner {len(state['corners'])}/4"


def parse_robot_xyz(raw: str) -> list[float] | None:
    """Three pendant numbers to meters. None when the text is not a pose."""
    text = raw.strip()
    if not text:
        return None
    parts = text.split()
    if len(parts) == 1:
        parts = [part for part in parts[0].split(",") if part.strip()]
    cleaned = []
    for part in parts:
        part = part.strip().strip(",")
        if not part:
            continue
        if "," in part and "." not in part:
            part = part.replace(",", ".")
        cleaned.append(part)
    if len(cleaned) != 3:
        return None
    try:
        vals = [float(part) for part in cleaned]
    except ValueError:
        return None
    if max(abs(v) for v in vals) < 5.0:
        return vals
    return [v / 1000.0 for v in vals]


def ask_robot_xyz(index: int) -> list[float] | None:
    """Read one corner from the terminal. None undoes that click."""
    print()
    print(f"Corner {index} of 4")
    print("Type the pendant X Y Z in millimeters. Example: 75.3 -358.1 -227.1")
    print("u undoes this corner.")
    while True:
        try:
            raw = input("> ").strip()
        except EOFError:
            return None
        if raw.lower() in {"u", "undo"}:
            return None
        pose = parse_robot_xyz(raw)
        if pose is None:
            print("Need three numbers: X Y Z")
            continue
        shown = [round(v * 1000.0, 1) for v in pose]
        print(f"  saved {shown[0]} {shown[1]} {shown[2]} mm")
        return pose


def pending_corner() -> dict | None:
    missing = [corner for corner in state["corners"] if "robot" not in corner]
    if not missing or state["phase"] != "corners":
        return None
    return missing[-1]


def save_from_clicks() -> bool:
    corners = state["corners"]
    if len(corners) != 4 or any("robot" not in c for c in corners) or not state["colors"]:
        state["message"] = "need 4 corners with X Y Z, and at least one object click"
        return False
    hues = [c[0] for c in state["colors"]]
    sats = [c[1] for c in state["colors"]]
    vals = [c[2] for c in state["colors"]]
    corners_xz = [[c["x"], c["z"]] for c in corners]
    corners_robot = [c["robot"] for c in corners]
    rotation, translation, error_m, hover = fit_camera_to_robot(corners_xz, corners_robot)
    data = {
        "corners_px": [[c["u"], c["v"]] for c in corners],
        "corners_xz": corners_xz,
        "corners_robot_xyz": corners_robot,
        "cam_xz_to_robot_R": rotation.tolist(),
        "cam_xz_to_robot_t": translation.tolist(),
        "hover_z": hover,
        "hue_center": circular_mean_hue(hues),
        "hue_tol": 12,
        "s_low": int(max(0, np.median(sats) - 50)),
        "v_low": int(max(0, np.median(vals) - 60)),
    }
    save_calibration(data)
    print("Saved", Path(__file__).resolve().parent / "calibration.json")
    print(f"  hover Z = {hover * 1000:.1f} mm")
    spread = (max(c[2] for c in corners_robot) - min(c[2] for c in corners_robot)) * 1000.0
    if spread > 20.0:
        print(f"  the four Z values differ by {spread:.1f} mm; hover uses their average")
    for i, err in enumerate(error_m, start=1):
        robot = corners_robot[i - 1]
        print(
            f"  corner {i}: robot {robot[0] * 1000:.1f} {robot[1] * 1000:.1f} "
            f"{robot[2] * 1000:.1f} mm, fit error {err * 1000:.1f} mm"
        )
    print(f"  hue {data['hue_center']:.1f} ± {data['hue_tol']}, S>={data['s_low']}, V>={data['v_low']}")
    return True


def main() -> int:
    print("Calibration. No robot connection.")
    print("Click a corner on the left image, then type its pendant X Y Z here.")
    cam = RealSense()
    try:
        cam.start()
    except RsError as exc:
        print(f"Camera failed: {exc}")
        print("On macOS: sudo ./venv/bin/python project/calibrate.py")
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
            _overlay, heat = blend_overlay(color, depth, 0.45, state["scale"])
            cv2.imshow(WIN, draw(color, heat))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quit without saving.")
                break
            if key == ord("u"):
                undo()
            if key == ord("n") and state["phase"] == "corners":
                if len(state["corners"]) == 4 and pending_corner() is None:
                    state["phase"] = "object"
                    state["message"] = "click the object"
                else:
                    state["message"] = "need 4 corners, each with X Y Z"
            if key == ord("s") and save_from_clicks():
                break
            corner = pending_corner()
            if corner is not None:
                state["message"] = f"corner {len(state['corners'])}/4 — type X Y Z in the terminal"
                cv2.imshow(WIN, draw(color, heat))
                cv2.waitKey(1)
                pose = ask_robot_xyz(len(state["corners"]))
                if pose is None:
                    state["corners"].pop()
                    state["message"] = f"corner {len(state['corners'])}/4"
                else:
                    corner["robot"] = pose
                    state["message"] = f"corner {len(state['corners'])}/4 recorded"
    finally:
        cam.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
