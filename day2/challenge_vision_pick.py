"""Day 2 Final Challenge — Vision-guided pick simulation.

1. Detect an object (MobileNet SSD)
2. Compute its pixel center
3. Convert to robot coordinates
4. Move URSim TCP above the object
"""

from __future__ import annotations

import signal
import sys
from pathlib import Path

import cv2
import numpy as np
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

HERE = Path(__file__).resolve().parent
PROTOTXT = HERE / "MobileNetSSD_deploy.prototxt"
MODEL = HERE / "MobileNetSSD_deploy.caffemodel"
SNAPSHOT = HERE / "snapshot.jpg"
OUT_DIR = HERE.parent / "deliver" / "day2"
ANNOTATED_PATH = OUT_DIR / "challenge_detection.png"

ROBOT_IP = "127.0.0.1"
CONF_THRESH = 0.4
SPEED = 0.15
ACCEL = 0.15
MAX_STEP_XY = 0.08  # max XY move from home TCP (avoids path sanity stop)
M_PER_PIXEL = 0.00025  # pixel offset → meters relative to image center
# Known-good UR5 joint pose (radians) — reset here before vision move
HOME_Q = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
# Prefer these classes for a "pick" demo when present
PREFERRED = {"bottle", "cup", "person", "pottedplant", "chair", "tvmonitor"}

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

rtde_c = None
rtde_r = None


def disconnect():
    global rtde_c, rtde_r
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
    print("\nCtrl+C — disconnecting...")
    if rtde_c is not None:
        try:
            rtde_c.stopJ(2.0)
        except Exception:
            pass
    disconnect()
    sys.exit(0)


def pixel_to_robot(px, py, scale=0.5, offset=(0.3, -0.2)):
    """Map image pixels to robot base XY (course Exercise 4)."""
    x = offset[0] + px * scale / 1000.0
    y = offset[1] + py * scale / 1000.0
    return x, y


def load_image() -> np.ndarray:
    img = cv2.imread(str(SNAPSHOT))
    if img is None:
        raise FileNotFoundError(
            f"Could not read {SNAPSHOT}. Run day2/ex1.py first and press 'q'."
        )
    return img


def detect_best(img: np.ndarray):
    """Return best detection as (label, confidence, cx, cy, box) or None."""
    net = cv2.dnn.readNetFromCaffe(str(PROTOTXT), str(MODEL))
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5
    )
    net.setInput(blob)
    detections = net.forward()

    candidates = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < CONF_THRESH:
            continue
        idx = int(detections[0, 0, i, 1])
        label = CLASSES[idx]
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        start_x, start_y, end_x, end_y = box.astype(int)
        cx = int((start_x + end_x) / 2)
        cy = int((start_y + end_y) / 2)
        preferred = 1 if label in PREFERRED else 0
        candidates.append((preferred, confidence, label, cx, cy,
                           (start_x, start_y, end_x, end_y)))

    if not candidates:
        return None

    # Prefer workshop-relevant classes, then highest confidence
    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
    _, confidence, label, cx, cy, box = candidates[0]
    return label, confidence, cx, cy, box


def annotate(img, label, confidence, cx, cy, box):
    start_x, start_y, end_x, end_y = box
    cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    cv2.circle(img, (cx, cy), 6, (0, 0, 255), -1)
    text = f"{label}: {confidence * 100:.1f}%  center=({cx},{cy})"
    cv2.putText(img, text, (start_x, max(20, start_y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return img


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def build_reachable_target(img_shape, cx, cy, base_pose):
    """Small XY offset from a known-good base TCP — keep Z and orientation."""
    h, w = img_shape[:2]

    dx = clamp((cx - w / 2.0) * M_PER_PIXEL, -MAX_STEP_XY, MAX_STEP_XY)
    dy = clamp((cy - h / 2.0) * M_PER_PIXEL, -MAX_STEP_XY, MAX_STEP_XY)

    target = list(base_pose)
    target[0] += dx
    target[1] += dy

    if not rtde_c.isPoseWithinSafetyLimits(target):
        raise RuntimeError(f"Target outside safety limits: {target}")
    if not rtde_c.getInverseKinematicsHasSolution(target):
        raise RuntimeError(f"No IK solution for target: {target}")

    q_target = rtde_c.getInverseKinematics(target)
    return target, q_target, dx, dy


def ensure_robot_ready():
    if rtde_r.isProtectiveStopped() or rtde_r.isEmergencyStopped():
        raise RuntimeError(
            "Robot is in protective/emergency stop. "
            "In PolyScope click Enable Robot, set Remote Control, then rerun."
        )


def safe_moveJ(q, speed=SPEED, acceleration=ACCEL):
    ensure_robot_ready()
    speed = min(speed, 0.3)
    acceleration = min(acceleration, 0.3)
    ok = rtde_c.moveJ(q, speed, acceleration)
    if not ok:
        raise RuntimeError("moveJ failed (path rejected by controller)")
    return ok


def go_home():
    print(f"Moving to home joints: {HOME_Q}")
    safe_moveJ(HOME_Q)
    pose = list(rtde_r.getActualTCPPose())
    print(f"Home TCP: {[round(v, 4) for v in pose]}")
    return pose


def main():
    global rtde_c, rtde_r
    signal.signal(signal.SIGINT, on_sigint)

    print("Loading image and detecting object...")
    img = load_image()
    result = detect_best(img)
    if result is None:
        raise RuntimeError(
            f"No object detected above {CONF_THRESH:.0%} in {SNAPSHOT.name}. "
            "Capture a clearer snapshot with a bottle/person/etc. in view."
        )

    label, confidence, cx, cy, box = result
    print(f"Detected {label} ({confidence * 100:.1f}%) at pixel ({cx}, {cy})")

    # Course mapping (Exercise 4) — for the report / understanding
    course_x, course_y = pixel_to_robot(cx, cy)
    print(f"Course pixel_to_robot XY: ({course_x:.4f}, {course_y:.4f}) m")

    annotated = annotate(img.copy(), label, confidence, cx, cy, box)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ANNOTATED_PATH), annotated)
    print(f"Saved detection overlay → {ANNOTATED_PATH}")

    print(f"Connecting to URSim at {ROBOT_IP}...")
    rtde_c = RTDEControlInterface(ROBOT_IP)
    rtde_r = RTDEReceiveInterface(ROBOT_IP)
    print("Connected. Press Ctrl+C anytime to disconnect.")
    ensure_robot_ready()

    # Reset from whatever awkward pose previous runs left behind
    home_pose = go_home()

    target, q_target, dx, dy = build_reachable_target(img.shape, cx, cy, home_pose)
    print(f"Delta XY from image center: ({dx:.4f}, {dy:.4f}) m")
    print(f"Target  TCP: {[round(v, 4) for v in target]}")
    print(f"Target  joints: {[round(v, 4) for v in q_target]}")

    safe_moveJ(q_target)
    print("Reached approach pose above object.")
    disconnect()
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("Error:", exc)
        disconnect()
        sys.exit(1)
