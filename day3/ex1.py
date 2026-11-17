"""Day 3 — Exercise 1: Represent the workspace as a grid graph."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "deliver" / "day3"
OUT_PATH = OUT_DIR / "ex1_grid.png"


def create_grid(size=10, obstacle_prob=0.2, seed=42):
    """Build a size×size occupancy grid (0 = free, 1 = obstacle)."""
    rng = np.random.default_rng(seed)
    grid = np.zeros((size, size), dtype=int)
    obstacles = rng.random((size, size)) < obstacle_prob
    grid[obstacles] = 1
    return grid


def pick_free_cell(grid, rng, avoid=()):
    """Pick a random free cell, not in `avoid`."""
    free = [
        (r, c)
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if grid[r, c] == 0 and (r, c) not in avoid
    ]
    if not free:
        raise RuntimeError("No free cells left in the grid")
    idx = int(rng.integers(0, len(free)))
    return free[idx]


def show_grid(grid, start, goal, save_path=OUT_PATH):
    """Display the grid graph with start (S) and goal (G)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap="Greys", origin="upper")

    # Draw cell borders so the graph structure is visible
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="lightgray", linestyle="-", linewidth=0.5)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    ax.scatter(start[1], start[0], s=180, c="lime", edgecolors="black", zorder=3, label="Start")
    ax.scatter(goal[1], goal[0], s=180, c="red", edgecolors="black", marker="*", zorder=3, label="Goal")
    ax.text(start[1], start[0], " S", color="black", ha="left", va="center", fontsize=10, fontweight="bold")
    ax.text(goal[1], goal[0], " G", color="black", ha="left", va="center", fontsize=10, fontweight="bold")

    ax.set_title("Grid Workspace (0=free, 1=obstacle)")
    ax.legend(loc="upper right")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")
    print(f"Grid size: {grid.shape[0]}×{grid.shape[1]}")
    print(f"Obstacles: {int(grid.sum())}")
    print(f"Start: {start}")
    print(f"Goal:  {goal}")
    plt.show()


def main():
    rng = np.random.default_rng(42)
    grid = create_grid(size=10, obstacle_prob=0.2, seed=42)

    # Ensure start/goal are free cells (graph nodes the planner can use)
    start = pick_free_cell(grid, rng)
    goal = pick_free_cell(grid, rng, avoid={start})

    # Each free cell is a node; 4-connected neighbors are edges (used in Ex2+)
    show_grid(grid, start, goal)


if __name__ == "__main__":
    main()
