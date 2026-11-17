"""RealSense color and depth capture used by the camera loop.

Uses Homebrew librealsense (C API via ctypes).
"""

from __future__ import annotations

import ctypes
import os
import sys

if sys.platform.startswith("linux"):
    # opencv-python only ships the Qt xcb plugin; "wayland" in QT_QPA_PLATFORM
    # makes window creation fail. Run through XWayland instead.
    os.environ["QT_QPA_PLATFORM"] = "xcb"
from pathlib import Path

import cv2
import numpy as np

RS2_API_VERSION = 2 * 10000 + 59 * 100 + 2  # 2.59.2
RS2_STREAM_DEPTH = 1
RS2_STREAM_COLOR = 2
RS2_FORMAT_Z16 = 1
RS2_FORMAT_BGR8 = 6
RS2_EXTENSION_DEPTH_FRAME = 12

LIB_CANDIDATES = [
    "/opt/homebrew/lib/librealsense2.dylib",
    "/usr/local/lib/librealsense2.dylib",
    "librealsense2.so",  # Linux system install (e.g. AUR librealsense)
]


def _pyrealsense2_lib() -> str | None:
    """Linux pyrealsense2 wheels statically embed librealsense and export the C API."""
    try:
        import importlib.util

        spec = importlib.util.find_spec("pyrealsense2")
    except Exception:
        return None
    if not spec or not spec.submodule_search_locations:
        return None
    pkg = Path(next(iter(spec.submodule_search_locations)))
    hits = sorted(pkg.glob("pyrealsense2.cpython-*.so"))
    return str(hits[0]) if hits else None


class RsError(RuntimeError):
    pass


class RsIntrinsics(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("ppx", ctypes.c_float),
        ("ppy", ctypes.c_float),
        ("fx", ctypes.c_float),
        ("fy", ctypes.c_float),
        ("model", ctypes.c_int),
        ("coeffs", ctypes.c_float * 5),
    ]


def load_lib() -> ctypes.CDLL:
    last = None
    candidates = list(LIB_CANDIDATES)
    if (wheel := _pyrealsense2_lib()) is not None:
        candidates.append(wheel)
    for path in candidates:
        try:
            return ctypes.CDLL(path)
        except OSError as exc:
            last = exc
    raise SystemExit(
        "Could not load librealsense2.\n"
        "macOS: brew install librealsense\n"
        "Linux: uv pip install pyrealsense2 (into the venv)\n"
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
    lib.rs2_pipeline_get_active_profile = f("rs2_pipeline_get_active_profile", P, P, E)
    lib.rs2_pipeline_profile_get_streams = f("rs2_pipeline_profile_get_streams", P, P, E)
    lib.rs2_delete_pipeline_profile = f("rs2_delete_pipeline_profile", None, P)
    lib.rs2_get_stream_profiles_count = f(
        "rs2_get_stream_profiles_count", ctypes.c_int, P, E
    )
    lib.rs2_get_stream_profile = f("rs2_get_stream_profile", P, P, ctypes.c_int, E)
    lib.rs2_delete_stream_profiles_list = f("rs2_delete_stream_profiles_list", None, P)
    lib.rs2_get_stream_profile_data = f(
        "rs2_get_stream_profile_data",
        None,
        P,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        E,
    )
    lib.rs2_get_video_stream_intrinsics = f(
        "rs2_get_video_stream_intrinsics", None, P, ctypes.POINTER(RsIntrinsics), E
    )


class RealSense:
    def __init__(self, width=640, height=480, fps=30):
        self.lib = load_lib()
        bind(self.lib)
        self.width = width
        self.height = height
        self.fps = fps
        self._err = ctypes.c_void_p()
        self.ctx = self.pipe = self.cfg = self.align = self.queue = None
        self.intrinsics: dict | None = None
        self.depth_scale_m: float | None = None

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
        self._load_color_intrinsics()
        self.align = self.lib.rs2_create_align(RS2_STREAM_COLOR, e)
        self._check()
        self.queue = self.lib.rs2_create_frame_queue(1, e)
        self._check()
        self.lib.rs2_start_processing_queue(self.align, self.queue, e)
        self._check()

    def _load_color_intrinsics(self) -> None:
        e = ctypes.byref(self._err)
        profile = self.lib.rs2_pipeline_get_active_profile(self.pipe, e)
        self._check()
        streams = self.lib.rs2_pipeline_profile_get_streams(profile, e)
        self._check()
        try:
            count = self.lib.rs2_get_stream_profiles_count(streams, e)
            self._check()
            for i in range(count):
                mode = self.lib.rs2_get_stream_profile(streams, i, e)
                self._check()
                stream = ctypes.c_int()
                fmt = ctypes.c_int()
                index = ctypes.c_int()
                unique = ctypes.c_int()
                fps = ctypes.c_int()
                self.lib.rs2_get_stream_profile_data(
                    mode,
                    ctypes.byref(stream),
                    ctypes.byref(fmt),
                    ctypes.byref(index),
                    ctypes.byref(unique),
                    ctypes.byref(fps),
                    e,
                )
                self._check()
                if stream.value != RS2_STREAM_COLOR:
                    continue
                intr = RsIntrinsics()
                self.lib.rs2_get_video_stream_intrinsics(mode, ctypes.byref(intr), e)
                self._check()
                self.intrinsics = {
                    "fx": float(intr.fx),
                    "fy": float(intr.fy),
                    "ppx": float(intr.ppx),
                    "ppy": float(intr.ppy),
                    "width": int(intr.width),
                    "height": int(intr.height),
                }
                print(
                    f"Color intrinsics {intr.width}x{intr.height} "
                    f"fx={intr.fx:.1f} fy={intr.fy:.1f}"
                )
                return
        finally:
            self.lib.rs2_delete_stream_profiles_list(streams)
            self.lib.rs2_delete_pipeline_profile(profile)
        print("Warning: color intrinsics not found")

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
            if depth[cy, cx] > 0 and z > 0:
                self.depth_scale_m = z / float(depth[cy, cx])
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


# Depth colors: 0 cm is one end of the scale, 100 cm is the other.
DEPTH_COLOR_MAX_M = 1.0


def depth_to_colormap(depth_u16: np.ndarray, depth_scale_m: float) -> np.ndarray:
    valid = depth_u16 > 0
    depth_m = depth_u16.astype(np.float32) * float(depth_scale_m)
    norm = np.clip(depth_m / DEPTH_COLOR_MAX_M, 0, 1)
    vis = (norm * 255).astype(np.uint8)
    heat = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    heat[~valid] = 0
    return heat


def blend_overlay(
    color: np.ndarray, depth_u16: np.ndarray, alpha: float, depth_scale_m: float = 0.001
) -> tuple[np.ndarray, np.ndarray]:
    heat = depth_to_colormap(depth_u16, depth_scale_m)
    valid = depth_u16 > 0
    out = color.copy()
    if np.any(valid):
        mixed = cv2.addWeighted(color, 1.0 - alpha, heat, alpha, 0)
        out[valid] = mixed[valid]
    return out, heat
