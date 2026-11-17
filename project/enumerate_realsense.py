"""List connected RealSense devices.

On macOS the pip wheel (pyrealsense2-macosx) often segfaults when opening a
D455. Prefer Homebrew's CLI (librealsense 2.59+):

    brew install librealsense
    sudo rs-enumerate-devices

This script tries the brew CLI first, then falls back to pyrealsense2.
"""

from __future__ import annotations

import shutil
import subprocess
import sys


def via_brew() -> int | None:
    exe = shutil.which("rs-enumerate-devices")
    if not exe:
        return None
    print(f"Using Homebrew tool: {exe}", flush=True)
    print("If this fails with 'cannot access depth sensor', re-run with sudo:", flush=True)
    print(f"  sudo {exe}", flush=True)
    print("-" * 60, flush=True)
    result = subprocess.run([exe], check=False)
    return result.returncode


def via_python() -> int:
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("pyrealsense2 not installed and rs-enumerate-devices not found.")
        print("  brew install librealsense")
        print("  pip install -r project/requirements.txt")
        return 1

    print("Using pyrealsense2 (may segfault on macOS with D455)…")
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        n = devices.size()
        if n == 0:
            print("No RealSense devices detected.")
            return 1
        for i in range(n):
            # Accessing the device can SIGSEGV on macOS — keep prints minimal
            print(f"Device index {i} present (opening info may crash on macOS).")
            dev = devices[i]
            print(f"  name:     {dev.get_info(rs.camera_info.name)}")
            print(f"  serial:   {dev.get_info(rs.camera_info.serial_number)}")
            print(f"  firmware: {dev.get_info(rs.camera_info.firmware_version)}")
        return 0
    except RuntimeError as exc:
        print(f"RealSense query failed: {exc}")
        print("On macOS try: sudo rs-enumerate-devices")
        return 1


def main() -> int:
    code = via_brew()
    if code is not None:
        return code
    return via_python()


if __name__ == "__main__":
    sys.exit(main())
