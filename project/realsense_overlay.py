"""Live RealSense window: color image with depth heatmap overlay.

Uses Homebrew librealsense (C API via ctypes) — not the crashing
pyrealsense2-macosx wheel.

Usage (macOS needs sudo for USB):
    sudo ./venv/bin/python project/realsense_overlay.py
    ./project/test_realsense_depth.sh

Keys:
    q     quit
    s     save snapshot to project/captures/
    [ ]   decrease / increase overlay strength
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "captures"

RS2_API_VERSION = 2 * 10000 + 59 * 100 + 2  # 2.59.2
RS2_STREAM_DEPTH = 1
RS2_STREAM_COLOR = 2
RS2_FORMAT_Z16 = 1
RS2_FORMAT_BGR8 = 6
RS2_EXTENSION_DEPTH_FRAME = 12

LIB_CANDIDATES = [
    "/opt/homebrew/lib/librealsense2.dylib",
    "/usr/local/lib/librealsense2.dylib",
]


class RsError(RuntimeError):
    pass


def load_lib() -> ctypes.CDLL:
    last = None
    for path in LIB_CANDIDATES:
        try:
            return ctypes.CDLL(path)
        except OSError as exc:
            last = exc
    raise SystemExit(
        "Could not load librealsense2.dylib.\n"
        "Install with: brew install librealsense\n"
        f"Last error: {last}"
    )


def bind(lib: ctypes.CDLL) -> None:
    def f(name, restype, *argtypes):
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = list(argtypes)
        return fn

    P = ctypes.c_void_p
    E = ctypes.POINTER(ctypes.c_void_p)

    lib.rs2_create_context = f("rs2_create_context", P, ctypes.c_int, E)
    lib.rs2_delete_context = f("rs2_delete_context", None, P)
    lib.rs2_create_pipeline = f("rs2_create_pipeline", P, P, E)
    lib.rs2_delete_pipeline = f("rs2_delete_pipeline", None, P)
    lib.rs2_create_config = f("rs2_create_config", P, E)
    lib.rs2_delete_config = f("rs2_delete_config", None, P)
    lib.rs2_config_enable_stream = f(
        "rs2_config_enable_stream",
        None,
        P,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        E,
    )
    lib.rs2_pipeline_start_with_config = f(
        "rs2_pipeline_start_with_config", P, P, P, E
    )
    lib.rs2_pipeline_stop = f("rs2_pipeline_stop", None, P, E)
    lib.rs2_pipeline_wait_for_frames = f(
        "rs2_pipeline_wait_for_frames", P, P, ctypes.c_uint, E
    )
    lib.rs2_create_align = f("rs2_create_align", P, ctypes.c_int, E)
    lib.rs2_create_frame_queue = f("rs2_create_frame_queue", P, ctypes.c_int, E)
    lib.rs2_delete_frame_queue = f("rs2_delete_frame_queue", None, P)
    lib.rs2_start_processing_queue = f("rs2_start_processing_queue", None, P, P, E)
    lib.rs2_process_frame = f("rs2_process_frame", None, P, P, E)
    lib.rs2_wait_for_frame = f("rs2_wait_for_frame", P, P, ctypes.c_uint, E)
    lib.rs2_delete_processing_block = f("rs2_delete_processing_block", None, P)
    lib.rs2_embedded_frames_count = f("rs2_embedded_frames_count", ctypes.c_int, P, E)
    lib.rs2_extract_frame = f("rs2_extract_frame", P, P, ctypes.c_int, E)
    lib.rs2_release_frame = f("rs2_release_frame", None, P)
    lib.rs2_is_frame_extendable_to = f(
        "rs2_is_frame_extendable_to", ctypes.c_int, P, ctypes.c_int, E
    )
    lib.rs2_get_frame_width = f("rs2_get_frame_width", ctypes.c_int, P, E)
    lib.rs2_get_frame_height = f("rs2_get_frame_height", ctypes.c_int, P, E)
    lib.rs2_get_frame_stride_in_bytes = f(
        "rs2_get_frame_stride_in_bytes", ctypes.c_int, P, E
    )
    lib.rs2_get_frame_data = f("rs2_get_frame_data", ctypes.c_void_p, P, E)
    lib.rs2_depth_frame_get_distance = f(
        "rs2_depth_frame_get_distance", ctypes.c_float, P, ctypes.c_int, ctypes.c_int, E
    )
    lib.rs2_get_error_message = f("rs2_get_error_message", ctypes.c_char_p, P)
    lib.rs2_free_error = f("rs2_free_error", None, P)


class RealSense:
    def __init__(self, width=640, height=480, fps=30):
        self.lib = load_lib()
        bind(self.lib)
        self.width = width
        self.height = height
        self.fps = fps
        self._err = ctypes.c_void_p()
        self.ctx = self.pipe = self.cfg = self.align = self.queue = None

    def _check(self) -> None:
        if self._err:
            msg = self.lib.rs2_get_error_message(self._err)
            text = msg.decode() if msg else "unknown RealSense error"
            self.lib.rs2_free_error(self._err)
            self._err = ctypes.c_void_p()
            raise RsError(text)

    def start(self) -> None:
        e = ctypes.byref(self._err)
        self.ctx = self.lib.rs2_create_context(RS2_API_VERSION, e)
        self._check()
        self.pipe = self.lib.rs2_create_pipeline(self.ctx, e)
        self._check()
        self.cfg = self.lib.rs2_create_config(e)
        self._check()
        for stream, fmt in (
            (RS2_STREAM_DEPTH, RS2_FORMAT_Z16),
            (RS2_STREAM_COLOR, RS2_FORMAT_BGR8),
        ):
            self.lib.rs2_config_enable_stream(
                self.cfg, stream, -1, self.width, self.height, fmt, self.fps, e
            )
            self._check()
        self.lib.rs2_pipeline_start_with_config(self.pipe, self.cfg, e)
        self._check()
        self.align = self.lib.rs2_create_align(RS2_STREAM_COLOR, e)
        self._check()
        self.queue = self.lib.rs2_create_frame_queue(1, e)
        self._check()
        self.lib.rs2_start_processing_queue(self.align, self.queue, e)
        self._check()

    def _frame_array(self, frame, channels: int, dtype) -> np.ndarray:
        e = ctypes.byref(self._err)
        w = self.lib.rs2_get_frame_width(frame, e)
        self._check()
        h = self.lib.rs2_get_frame_height(frame, e)
        self._check()
        stride = self.lib.rs2_get_frame_stride_in_bytes(frame, e)
        self._check()
        data = self.lib.rs2_get_frame_data(frame, e)
        self._check()
        itemsize = np.dtype(dtype).itemsize
        buf = (ctypes.c_ubyte * (stride * h)).from_address(data)
        raw = np.frombuffer(buf, dtype=np.uint8).reshape(h, stride)
        useful = raw[:, : w * channels * itemsize]
        if channels == 1:
            return np.ascontiguousarray(useful.view(dtype).reshape(h, w)).copy()
        return np.ascontiguousarray(
            useful.view(dtype).reshape(h, w, channels)
        ).copy()

    def wait_aligned(self, timeout_ms=5000):
        e = ctypes.byref(self._err)
        frames = self.lib.rs2_pipeline_wait_for_frames(self.pipe, timeout_ms, e)
        self._check()
        try:
            self.lib.rs2_process_frame(self.align, frames, e)
            self._check()
        finally:
            self.lib.rs2_release_frame(frames)

        aligned = self.lib.rs2_wait_for_frame(self.queue, timeout_ms, e)
        self._check()
        try:
            return self._split_color_depth(aligned)
        finally:
            self.lib.rs2_release_frame(aligned)

    def _split_color_depth(self, composite):
        e = ctypes.byref(self._err)
        n = self.lib.rs2_embedded_frames_count(composite, e)
        self._check()
        color = depth = None
        depth_handle = None
        extracted = []
        try:
            for i in range(n):
                f = self.lib.rs2_extract_frame(composite, i, e)
                self._check()
                extracted.append(f)
                is_depth = self.lib.rs2_is_frame_extendable_to(
                    f, RS2_EXTENSION_DEPTH_FRAME, e
                )
                self._check()
                if is_depth:
                    depth = self._frame_array(f, 1, np.uint16)
                    depth_handle = f
                else:
                    color = self._frame_array(f, 3, np.uint8)
            if color is None or depth is None or depth_handle is None:
                raise RsError("Aligned frames missing color or depth")
            cx, cy = color.shape[1] // 2, color.shape[0] // 2
            z = float(
                self.lib.rs2_depth_frame_get_distance(depth_handle, cx, cy, e)
            )
            self._check()
            return color, depth, z
        finally:
            for f in extracted:
                self.lib.rs2_release_frame(f)

    def stop(self) -> None:
        e = ctypes.byref(self._err)
        if self.pipe:
            try:
                self.lib.rs2_pipeline_stop(self.pipe, e)
                self._check()
            except RsError:
                pass
        if self.align:
            self.lib.rs2_delete_processing_block(self.align)
        if self.queue:
            self.lib.rs2_delete_frame_queue(self.queue)
        if self.cfg:
            self.lib.rs2_delete_config(self.cfg)
        if self.pipe:
            self.lib.rs2_delete_pipeline(self.pipe)
        if self.ctx:
            self.lib.rs2_delete_context(self.ctx)


def depth_to_colormap(depth_u16: np.ndarray) -> np.ndarray:
    valid = depth_u16 > 0
    vis = np.zeros(depth_u16.shape, dtype=np.uint8)
    if np.any(valid):
        vals = depth_u16[valid].astype(np.float32)
        lo, hi = np.percentile(vals, [5, 95])
        if hi <= lo:
            hi = lo + 1.0
        norm = np.zeros_like(depth_u16, dtype=np.float32)
        norm[valid] = np.clip((depth_u16[valid] - lo) / (hi - lo), 0, 1)
        vis = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)


def blend_overlay(color: np.ndarray, depth_u16: np.ndarray, alpha: float) -> np.ndarray:
    heat = depth_to_colormap(depth_u16)
    valid = depth_u16 > 0
    out = color.copy()
    if np.any(valid):
        mixed = cv2.addWeighted(color, 1.0 - alpha, heat, alpha, 0)
        out[valid] = mixed[valid]
    return out, heat


def main() -> int:
    alpha = 0.45
    cam = RealSense()
    print("Starting RealSense (color + depth overlay)…")
    print("If this fails with USB/depth access, re-run with sudo.")
    try:
        cam.start()
    except RsError as exc:
        print(f"Failed to start camera: {exc}")
        print("On macOS run:")
        print("  sudo ./venv/bin/python project/realsense_overlay.py")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    win = "RealSense — overlay | depth"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            color, depth, center_z = cam.wait_aligned()
            overlay, heat = blend_overlay(color, depth, alpha)

            cv2.drawMarker(
                overlay,
                (overlay.shape[1] // 2, overlay.shape[0] // 2),
                (0, 255, 0),
                cv2.MARKER_CROSS,
                18,
                2,
            )
            cv2.putText(
                overlay,
                f"center: {center_z:.3f} m   overlay: {alpha:.2f}  ([ ] adjust, s save, q quit)",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # overlay | depth side-by-side
            view = np.hstack([overlay, heat])
            cv2.imshow(win, view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("["):
                alpha = max(0.0, alpha - 0.05)
            if key == ord("]"):
                alpha = min(1.0, alpha + 0.05)
            if key == ord("s"):
                n = len(list(OUT_DIR.glob("overlay_*.png")))
                path = OUT_DIR / f"overlay_{n:03d}.png"
                cv2.imwrite(str(path), view)
                print(f"Saved {path}")
    except RsError as exc:
        print(f"Stream error: {exc}")
        return 1
    finally:
        cam.stop()
        cv2.destroyAllWindows()

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
