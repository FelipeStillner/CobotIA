"""Test RealSense camera on this machine.

macOS reality check
-------------------
- Color RGB works via OpenCV (AVFoundation) without sudo.
- Depth via pyrealsense2-macosx often **segfaults** on recent macOS + D455
  (HID/IMU init). Use Homebrew librealsense tools with sudo instead.

Usage
-----
Color only (safe, no sudo):
    python project/test_realsense.py
    python project/test_realsense.py --index 0

Color + depth overlay in a window (needs password on macOS):
    ./project/test_realsense_depth.sh
    # or:
    sudo ./venv/bin/python project/realsense_overlay.py

Keys: q=quit, s=save, [ ]=overlay strength
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "captures"


def open_camera(index: int) -> cv2.VideoCapture:
    for backend in (getattr(cv2, "CAP_AVFOUNDATION", None), cv2.CAP_ANY):
        if backend is None:
            continue
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(index)


def probe_indices(max_index: int = 4) -> None:
    print("Probing camera indices…")
    for i in range(max_index + 1):
        cap = open_camera(i)
        if not cap.isOpened():
            print(f"  [{i}] unavailable")
            continue
        ret, frame = cap.read()
        shape = None if (not ret or frame is None) else frame.shape
        print(f"  [{i}] ok  shape={shape}")
        cap.release()
    print("Tip: RealSense RGB is often 640x480; MacBook FaceTime is often 1920x1080.")


def print_depth_help() -> None:
    print()
    print("=== Color + depth overlay (window) ===")
    print("  ./project/test_realsense_depth.sh")
    print("  sudo ./venv/bin/python project/realsense_overlay.py")
    print()
    print("Shows blended overlay | depth heatmap. Avoid `rs-depth` (ASCII).")
    if not shutil.which("rs-enumerate-devices"):
        print("If needed: brew install librealsense")


def run_color(index: int) -> int:
    cap = open_camera(index)
    if not cap.isOpened():
        print(f"Could not open camera index {index}.")
        print("Try: python project/test_realsense.py --list")
        return 1

    print(f"Opened camera index {index} (OpenCV color). q=quit, s=save")
    if platform.system() == "Darwin":
        print("Depth needs: sudo rs-depth   (Homebrew librealsense)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break
        label = "RealSense RGB (OpenCV)" if platform.system() == "Darwin" else "Camera"
        cv2.putText(
            frame,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("RealSense color test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            n = len(list(OUT_DIR.glob("color_*.png")))
            path = OUT_DIR / f"color_{n:03d}.png"
            cv2.imwrite(str(path), frame)
            print(f"Saved {path}")

    cap.release()
    cv2.destroyAllWindows()
    return 0


def try_pyrealsense() -> int:
    """Optional Linux/Windows path — avoided by default on macOS."""
    try:
        import numpy as np
        import pyrealsense2 as rs
    except ImportError:
        print("pyrealsense2 not installed.")
        return 1

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    print("Starting pyrealsense2 pipeline…")
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    colorizer = rs.colorizer()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        while True:
            frames = align.process(pipeline.wait_for_frames(5000))
            depth = frames.get_depth_frame()
            color = frames.get_color_frame()
            if not depth or not color:
                continue
            color_img = np.asanyarray(color.get_data())
            depth_vis = np.asanyarray(colorizer.colorize(depth).get_data())
            h, w = color_img.shape[:2]
            z = depth.get_distance(w // 2, h // 2)
            cv2.drawMarker(color_img, (w // 2, h // 2), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(
                color_img,
                f"center depth: {z:.3f} m",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("RealSense color | depth", np.hstack([color_img, depth_vis]))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                n = len(list(OUT_DIR.glob("color_*.png")))
                cv2.imwrite(str(OUT_DIR / f"color_{n:03d}.png"), color_img)
                cv2.imwrite(str(OUT_DIR / f"depth_{n:03d}.png"), depth_vis)
                print(f"Saved color_{n:03d}.png / depth_{n:03d}.png")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=int, default=0, help="OpenCV camera index for color")
    parser.add_argument("--list", action="store_true", help="list OpenCV camera indices")
    parser.add_argument("--depth-help", action="store_true", help="print depth/SDK instructions")
    parser.add_argument(
        "--pyrealsense",
        action="store_true",
        help="force pyrealsense2 pipeline (segfaults on this macOS+D455 setup)",
    )
    args = parser.parse_args()

    if args.list:
        probe_indices()
        return 0
    if args.depth_help:
        print_depth_help()
        return 0
    if args.pyrealsense:
        if platform.system() == "Darwin":
            print("WARNING: pyrealsense2 often segfaults on macOS with D455.")
            print("Prefer: sudo rs-depth")
        return try_pyrealsense()

    code = run_color(args.index)
    if platform.system() == "Darwin":
        print_depth_help()
    return code


if __name__ == "__main__":
    sys.exit(main())
