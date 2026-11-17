"""Day 3 — Exercise 2: Implement BFS and Dijkstra on the grid graph."""

from collections import deque
from heapq import heappop, heappush
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ex1 import create_grid, pick_free_cell

OUT_DIR = Path(__file__).resolve().parent.parent / "deliver" / "day3"
OUT_PATH = OUT_DIR / "ex2_bfs_dijkstra.png"


def neighbors(pos, grid):
    x, y = pos
    for dx, dy in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and grid[nx, ny] == 0:
            yield (nx, ny)


def reconstruct_path(came_from, start, goal):
    if goal not in came_from and goal != start:
        return None
    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


def bfs(start, goal, grid):
    """Breadth-first search — shortest path in number of steps (unweighted)."""
    if start == goal:
        return [start], {start}

    queue = deque([start])
    came_from = {start: None}
    visited = {start}

    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for nxt in neighbors(current, grid):
            if nxt not in came_from:
                came_from[nxt] = current
                visited.add(nxt)
                queue.append(nxt)

    path = reconstruct_path(
        {k: v for k, v in came_from.items() if v is not None}, start, goal
    )
    return path, visited


def dijkstra(start, goal, grid):
    """Dijkstra — shortest path with uniform edge cost 1 (same length as BFS here)."""
    if start == goal:
        return [start], {start}, 0.0

    pq = [(0.0, start)]  # (cost, node)
    came_from = {}
    cost_so_far = {start: 0.0}
    visited = set()

    while pq:
        cost, current = heappop(pq)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            break
        for nxt in neighbors(current, grid):
            new_cost = cost + 1.0
            if new_cost < cost_so_far.get(nxt, float("inf")):
                cost_so_far[nxt] = new_cost
                came_from[nxt] = current
                heappush(pq, (new_cost, nxt))

    path = reconstruct_path(came_from, start, goal)
    total = cost_so_far.get(goal, float("inf")) if path else float("inf")
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

    print(f"Start: {start} | Goal: {goal}")
    print(f"BFS:      path_len={len(bfs_path) if bfs_path else None}, visited={len(bfs_visited)}")
    print(f"Dijkstra: path_len={len(dij_path) if dij_path else None}, "
          f"cost={dij_cost}, visited={len(dij_visited)}")
    print(f"BFS path:      {bfs_path}")
    print(f"Dijkstra path: {dij_path}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    draw_result(axes[0], grid, start, goal, bfs_path, bfs_visited, "BFS")
    draw_result(axes[1], grid, start, goal, dij_path, dij_visited, "Dijkstra")
    fig.suptitle("Exercise 2 — BFS vs Dijkstra")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved {OUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
