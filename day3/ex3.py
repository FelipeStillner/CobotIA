"""Day 3 — Exercise 3: Implement A* search on the grid graph."""

from heapq import heappop, heappush
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ex1 import create_grid, pick_free_cell
from ex2 import bfs, dijkstra, neighbors, reconstruct_path

OUT_DIR = Path(__file__).resolve().parent.parent / "deliver" / "day3"
OUT_PATH = OUT_DIR / "ex3_astar.png"


def heuristic(a, b):
    """Manhattan distance (admissible on a 4-connected grid)."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(start, goal, grid):
    """A* — optimal path using g(n) + h(n)."""
    if start == goal:
        return [start], {start}, 0.0

    open_heap = []  # (f_score, counter, node)
    counter = 0
    heappush(open_heap, (heuristic(start, goal), counter, start))

    came_from = {}
    g_score = {start: 0.0}
    visited = set()

    while open_heap:
        _, _, current = heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            break

        for nxt in neighbors(current, grid):
            tentative = g_score[current] + 1.0
            if tentative < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative
                f = tentative + heuristic(nxt, goal)
                counter += 1
                heappush(open_heap, (f, counter, nxt))

    path = reconstruct_path(came_from, start, goal)
    total = g_score.get(goal, float("inf")) if path else float("inf")
    return path, visited, total


def draw_result(ax, grid, start, goal, path, visited, title):
    ax.imshow(grid, cmap="Greys", origin="upper")
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="lightgray", linestyle="-", linewidth=0.5)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    if visited:
        vr, vc = zip(*visited)
        ax.scatter(vc, vr, s=40, c="skyblue", alpha=0.6, label="Visited")

    if path:
        pr, pc = zip(*path)
        ax.plot(pc, pr, color="dodgerblue", linewidth=2, marker="o", markersize=4, label="Path")

    ax.scatter(start[1], start[0], s=160, c="lime", edgecolors="black", zorder=3, label="Start")
    ax.scatter(goal[1], goal[0], s=180, c="red", edgecolors="black", marker="*", zorder=3, label="Goal")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)


def main():
    rng = np.random.default_rng(42)
    grid = create_grid(size=10, obstacle_prob=0.2, seed=42)
    start = pick_free_cell(grid, rng)
    goal = pick_free_cell(grid, rng, avoid={start})

    bfs_path, bfs_visited = bfs(start, goal, grid)
    dij_path, dij_visited, dij_cost = dijkstra(start, goal, grid)
    astar_path, astar_visited, astar_cost = astar(start, goal, grid)

    print(f"Start: {start} | Goal: {goal}")
    print(f"BFS:      path_len={len(bfs_path) if bfs_path else None}, visited={len(bfs_visited)}")
    print(f"Dijkstra: path_len={len(dij_path) if dij_path else None}, "
          f"cost={dij_cost}, visited={len(dij_visited)}")
    print(f"A*:       path_len={len(astar_path) if astar_path else None}, "
          f"cost={astar_cost}, visited={len(astar_visited)}")
    print(f"A* path: {astar_path}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    draw_result(axes[0], grid, start, goal, bfs_path, bfs_visited, "BFS")
    draw_result(axes[1], grid, start, goal, dij_path, dij_visited, "Dijkstra")
    draw_result(axes[2], grid, start, goal, astar_path, astar_visited, "A*")
    fig.suptitle("Exercise 3 — A* (vs BFS / Dijkstra)")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved {OUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
