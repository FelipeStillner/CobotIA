"""Fallback color-only test via OpenCV (AVFoundation on macOS).

Useful when pyrealsense2 needs sudo and you only want to confirm the
RGB stream. Depth is NOT available this way.

Usage:
    python project/test_realsense_color_opencv.py
    python project/test_realsense_color_opencv.py --index 0

Press 'q' to quit, 's' to save a frame under project/captures/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "captures"


def open_camera(index: int) -> cv2.VideoCapture:
    # Prefer AVFoundation on macOS; fall back to default backend
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
    print("Tip: RealSense RGB is often 640x480; the MacBook camera is often 1920x1080.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="camera index")
    parser.add_argument("--list", action="store_true", help="list camera indices and exit")
    args = parser.parse_args()

    if args.list:
        probe_indices()
        return 0

    cap = open_camera(args.index)
    if not cap.isOpened():
        print(f"Could not open camera index {args.index}.")
        print("Try: python project/test_realsense_color_opencv.py --list")
        return 1

    print(f"Opened camera index {args.index}. Press q=quit, s=save.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break
        cv2.putText(
            frame,
            "OpenCV color fallback (no depth)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("RealSense color (OpenCV)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            n = len(list(OUT_DIR.glob("opencv_color_*.png")))
            path = OUT_DIR / f"opencv_color_{n:03d}.png"
            cv2.imwrite(str(path), frame)
            print(f"Saved {path}")

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
