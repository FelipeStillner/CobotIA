"""Day 4 — Exercise 5: Map discrete grid actions to robot-like TCP deltas."""

from pathlib import Path
import random
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ex1 import ACTION_NAMES, GridWorld
from ex2 import q_learning
from ex3 import greedy_policy, simulate_episode

OUT_DIR = Path(__file__).resolve().parent.parent / "deliver" / "day4"
OUT_PATH = OUT_DIR / "ex5_tcp_mapping.png"

# Grid action → small TCP translation in the robot base frame (meters).
# Convention used here (top-down workspace view):
#   0=up    → +Y
#   1=right → +X
#   2=down  → -Y
#   3=left  → -X
ACTION_TO_DELTA = {
    0: (0.0, 1.0, 0.0),   # up
    1: (1.0, 0.0, 0.0),   # right
    2: (0.0, -1.0, 0.0),  # down
    3: (-1.0, 0.0, 0.0),  # left
}


def grid_action_to_tcp_delta(action, step_size=0.05):
    """Map a GridWorld action to a TCP (dx, dy, dz) in meters."""
    if action not in ACTION_TO_DELTA:
        raise ValueError(f"Invalid action {action}; expected 0..3")
    ux, uy, uz = ACTION_TO_DELTA[action]
    return (ux * step_size, uy * step_size, uz * step_size)


def path_to_tcp_commands(actions, step_size=0.05, start_tcp=(0.3, -0.2, 0.25)):
    """Convert a sequence of grid actions into TCP waypoints."""
    x, y, z = start_tcp
    waypoints = [(x, y, z)]
    deltas = []
    for a in actions:
        dx, dy, dz = grid_action_to_tcp_delta(a, step_size=step_size)
        deltas.append((dx, dy, dz))
        x, y, z = x + dx, y + dy, z + dz
        waypoints.append((x, y, z))
    return deltas, waypoints


def visualize(actions, waypoints, save_path=OUT_PATH):
    xs = [p[0] for p in waypoints]
    ys = [p[1] for p in waypoints]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    for a in range(4):
        dx, dy, dz = grid_action_to_tcp_delta(a, step_size=0.05)
        ax.arrow(0, 0, dx, dy, head_width=0.008, length_includes_head=True,
                 color="tab:blue")
        ax.text(dx * 1.15, dy * 1.15, f"{a}:{ACTION_NAMES[a]}\n({dx:.2f},{dy:.2f})",
                ha="center", va="center", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-0.08, 0.08)
    ax.set_ylim(-0.08, 0.08)
    ax.axhline(0, color="lightgray")
    ax.axvline(0, color="lightgray")
    ax.set_xlabel("TCP X [m]")
    ax.set_ylabel("TCP Y [m]")
    ax.set_title("Action → TCP delta (step=0.05 m)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(xs, ys, "-o", color="dodgerblue", label="TCP path")
    ax.scatter(xs[0], ys[0], c="lime", s=120, edgecolors="black", zorder=3, label="Start TCP")
    ax.scatter(xs[-1], ys[-1], c="red", s=140, marker="*", edgecolors="black", zorder=3, label="End TCP")
    ax.set_aspect("equal")
    ax.set_xlabel("TCP X [m]")
    ax.set_ylabel("TCP Y [m]")
    ax.set_title("Greedy policy as TCP waypoints")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Exercise 5 — Grid actions → robot-like TCP commands")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")
    plt.show()


def main():
    random.seed(0)

    # Notebook-style mapping examples
    print("Mapping examples (step_size=0.05 m):")
    for a in range(4):
        delta = grid_action_to_tcp_delta(a, step_size=0.05)
        print(f"  action {a} ({ACTION_NAMES[a]:5s}) -> TCP delta {delta}")

    env = GridWorld(
        n_rows=5,
        n_cols=5,
        start=(0, 0),
        goal=(4, 4),
        obstacles={(1, 1), (2, 2), (3, 1)},
    )
    Q, _ = q_learning(env, episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1)
    path, actions, _ = simulate_episode(env, Q)

    deltas, waypoints = path_to_tcp_commands(actions, step_size=0.05)
    print("\nGreedy grid path:", path)
    print("TCP command sequence:")
    for i, (a, d, wp) in enumerate(zip(actions, deltas, waypoints[1:]), start=1):
        print(f"  {i:2d}. {ACTION_NAMES[a]:5s}  delta={d}  -> TCP={tuple(round(v, 3) for v in wp)}")

    visualize(actions, waypoints)


if __name__ == "__main__":
    main()
