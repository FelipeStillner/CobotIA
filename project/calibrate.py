"""Click the pad corners and the object, then save project/calibration.json.

Nothing is sent to the robot.

    sudo ./venv/bin/python project/calibrate.py

1. Click the 4 corners of the pad, in order around the edge.
2. Press n.
3. Click the object a few times (the color is sampled around each click).
4. Press s to save.

u undoes the last click. q quits without saving.
Clicks count on the left image only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration import circular_mean_hue, save_calibration
from detect_object import deproject
from realsense_overlay import RealSense, RsError, blend_overlay

WIN = "Calibrate — click the left image"
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
        state["message"] = f"corner {len(state['corners'])}/4"
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
        lines = [
            "Click 4 pad corners in order around the edge",
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


def save_from_clicks() -> bool:
    if len(state["corners"]) != 4 or not state["colors"]:
        state["message"] = "need 4 corners and at least one object click"
        return False
    hues = [c[0] for c in state["colors"]]
    sats = [c[1] for c in state["colors"]]
    vals = [c[2] for c in state["colors"]]
    data = {
        "corners_px": [[c["u"], c["v"]] for c in state["corners"]],
        "corners_xz": [[c["x"], c["z"]] for c in state["corners"]],
        "hue_center": circular_mean_hue(hues),
        "hue_tol": 12,
        "s_low": int(max(0, np.median(sats) - 50)),
        "v_low": int(max(0, np.median(vals) - 60)),
    }
    save_calibration(data)
    print("Saved", Path(__file__).resolve().parent / "calibration.json")
    print(f"  hue {data['hue_center']:.1f} ± {data['hue_tol']}, S>={data['s_low']}, V>={data['v_low']}")
    for i, (x, z) in enumerate(data["corners_xz"], start=1):
        print(f"  corner {i}: X={x:.3f} m  Z={z:.3f} m")
    return True


def main() -> int:
    print("Calibration. No robot connection.")
    print("Click 4 corners on the left image, press n, click the object, press s.")
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
                if len(state["corners"]) == 4:
                    state["phase"] = "object"
                    state["message"] = "click the object"
                else:
                    state["message"] = "need 4 corners first"
            if key == ord("s") and save_from_clicks():
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
