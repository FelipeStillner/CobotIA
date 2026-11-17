"""Day 5 — Part 4: Camera → robot calibration (affine transform).

Collect pixel <-> robot XY correspondences, then estimate a 2x3 affine map:
    [X]       [px]
    [Y] = A * [py]
              [ 1]
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "deliver" / "day5"
OUT_PATH = OUT_DIR / "part4_calibration.png"
NPY_PATH = OUT_DIR / "part4_affine_A.npy"


def compute_affine_transform(pixels: np.ndarray, robots: np.ndarray) -> np.ndarray:
    """Estimate a 2x3 affine transform with least squares.

    pixels: Nx2 (px, py)
    robots: Nx2 (X, Y) in meters
    returns: A shape (2, 3) such that [X, Y]^T = A @ [px, py, 1]^T
    """
    pixels = np.asarray(pixels, dtype=float)
    robots = np.asarray(robots, dtype=float)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("pixels must be Nx2")
    if robots.shape != pixels.shape:
        raise ValueError("robots must match pixels shape (Nx2)")
    if pixels.shape[0] < 3:
        raise ValueError("Need at least 3 correspondences for affine estimation")

    # Build design matrix for affine: X = a00*px + a01*py + a02
    #                                  Y = a10*px + a11*py + a12
    n = pixels.shape[0]
    M = np.zeros((2 * n, 6))
    b = np.zeros(2 * n)
    for i, ((px, py), (X, Y)) in enumerate(zip(pixels, robots)):
        M[2 * i] = [px, py, 1, 0, 0, 0]
        M[2 * i + 1] = [0, 0, 0, px, py, 1]
        b[2 * i] = X
        b[2 * i + 1] = Y

    params, *_ = np.linalg.lstsq(M, b, rcond=None)
    A = params.reshape(2, 3)
    return A


def pixel_to_robot(A: np.ndarray, px: float, py: float) -> tuple[float, float]:
    """Apply affine map to one pixel."""
    vec = np.array([px, py, 1.0])
    xy = A @ vec
    return float(xy[0]), float(xy[1])


def demo_correspondences():
    """Synthetic calibration set (replace with real ArUco measurements in lab)."""
    # Pretend camera pixels of 4 known board corners
    pixels = np.array([
        [100.0, 80.0],
        [540.0, 90.0],
        [120.0, 400.0],
        [520.0, 410.0],
    ])
    # Corresponding robot XY (meters) measured by jogging TCP onto each mark
    robots = np.array([
        [0.30, -0.20],
        [0.50, -0.18],
        [0.32, 0.05],
        [0.48, 0.04],
    ])
    return pixels, robots


def main():
    print("Defined compute_affine_transform(pixels, robots)")

    pixels, robots = demo_correspondences()
    A = compute_affine_transform(pixels, robots)
    print("Affine A (2x3):\n", A)

    # Reprojection error
    preds = np.array([pixel_to_robot(A, px, py) for px, py in pixels])
    err = np.linalg.norm(preds - robots, axis=1)
    print("Per-point error [m]:", np.round(err, 5))
    print("Mean error [m]:", float(err.mean()))

    # Demo query
    qx, qy = 320.0, 240.0
    rx, ry = pixel_to_robot(A, qx, qy)
    print(f"Example pixel ({qx}, {qy}) -> robot XY ({rx:.4f}, {ry:.4f}) m")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(NPY_PATH, A)
    print(f"Saved {NPY_PATH}")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(robots[:, 0], robots[:, 1], c="red", label="Measured robot XY", zorder=3)
    ax.scatter(preds[:, 0], preds[:, 1], c="dodgerblue", marker="x", s=80, label="Affine prediction")
    for (X, Y), (px, py) in zip(robots, pixels):
        ax.annotate(f"({px:.0f},{py:.0f})", (X, Y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Robot X [m]")
    ax.set_ylabel("Robot Y [m]")
    ax.set_title("Part 4 — Pixel→Robot affine calibration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved {OUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
