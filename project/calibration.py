"""Saved pad corners and object color.

Written by calibrate.py. Missing file means the fixed 40×70 cm box in config.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import WORK_DEPTH_Z, WORK_WIDTH_X, WORK_X0, WORK_Z0

PATH = Path(__file__).resolve().parent / "calibration.json"

_cache: dict = {"mtime": None, "data": None}


def load_calibration() -> dict | None:
    if not PATH.exists():
        _cache["mtime"] = None
        _cache["data"] = None
        return None
    mtime = PATH.stat().st_mtime
    if _cache["mtime"] != mtime:
        _cache["data"] = json.loads(PATH.read_text())
        _cache["mtime"] = mtime
    return _cache["data"]


def save_calibration(data: dict) -> None:
    PATH.write_text(json.dumps(data, indent=2) + "\n")
    _cache["mtime"] = None


def points_in_polygon(x, z, corners) -> np.ndarray:
    """Ray-cast test. x and z are broadcastable arrays, corners is Nx2 (x, z)."""
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    poly = np.asarray(corners, dtype=float)
    inside = np.zeros(np.broadcast(x, z).shape, dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, zi = poly[i]
        xj, zj = poly[j]
        denom = zj - zi
        intersect = ((zi > z) != (zj > z)) & (
            x < (xj - xi) * (z - zi) / (denom if denom != 0 else 1e-12) + xi
        )
        inside ^= intersect
        j = i
    return inside


def xz_inside(x, z) -> np.ndarray:
    """True where (x, z) is on the pad. Uses the calibrated corners when present."""
    calib = load_calibration()
    if calib and calib.get("corners_xz"):
        return points_in_polygon(x, z, calib["corners_xz"])
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    return (
        (x >= WORK_X0)
        & (x <= WORK_X0 + WORK_WIDTH_X)
        & (z >= WORK_Z0)
        & (z <= WORK_Z0 + WORK_DEPTH_Z)
    )


def circular_mean_hue(hues) -> float:
    """Mean of OpenCV hues (0–179)."""
    ang = np.deg2rad(np.asarray(hues, dtype=float) * 2.0)
    degrees = float(np.rad2deg(np.angle(np.mean(np.exp(1j * ang)))))
    if degrees < 0.0:
        degrees += 360.0
    hue = (degrees / 2.0) % 180.0
    if hue >= 179.999:
        hue = 0.0
    return float(hue)
