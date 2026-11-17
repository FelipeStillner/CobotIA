"""Day 3 Final Challenge — Combine A* planning and FSM for pick-and-place.

1. A* from start → object
2. FSM transitions to simulate grasp
3. A* from object → place
4. FSM transitions to complete the task
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ex1 import create_grid, pick_free_cell
from ex3 import astar
from ex4 import RobotFSM

OUT_DIR = Path(__file__).resolve().parent.parent / "deliver" / "day3"
FIG_PATH = OUT_DIR / "challenge_pick_place.png"
LOG_PATH = OUT_DIR / "challenge_pick_place.txt"


def simulate_pick_and_place(grid, start, object_pos, place_pos):
    """Run planning + FSM for a full pick-and-place sequence."""
    fsm = RobotFSM()
    log = []

    def step(msg):
        log.append(msg)
        print(msg)

    step(f"Start={start} | Object={object_pos} | Place={place_pos}")
    step(f"FSM state: {fsm.state}")

    # --- Phase 1: detect object and plan approach ---
    fsm.next("object_detected")
    step("Planning A* path: start → object")
    path_to_object, visited_to_object, cost1 = astar(start, object_pos, grid)
    if not path_to_object:
        raise RuntimeError("No path from start to object")
    step(f"  path length={len(path_to_object)}, cost={cost1}, visited={len(visited_to_object)}")
    fsm.next("reached_object")

    # --- Phase 2: grasp ---
    fsm.next("grasp_success")
    step("Grasp simulated: object acquired")

    # --- Phase 3: plan transport to place ---
    step("Planning A* path: object → place")
    path_to_place, visited_to_place, cost2 = astar(object_pos, place_pos, grid)
    if not path_to_place:
        raise RuntimeError("No path from object to place")
    step(f"  path length={len(path_to_place)}, cost={cost2}, visited={len(visited_to_place)}")
    fsm.next("reached_place")

    # --- Phase 4: place and finish ---
    fsm.next("place_success")
    step(f"Place simulated. Final FSM state: {fsm.state}")

    return {
        "fsm": fsm,
        "path_to_object": path_to_object,
        "path_to_place": path_to_place,
        "visited_to_object": visited_to_object,
        "visited_to_place": visited_to_place,
        "log": log,
    }


def draw(grid, start, object_pos, place_pos, result, save_path=FIG_PATH):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: approach
    ax = axes[0]
    ax.imshow(grid, cmap="Greys", origin="upper")
    if result["visited_to_object"]:
        vr, vc = zip(*result["visited_to_object"])
        ax.scatter(vc, vr, s=30, c="skyblue", alpha=0.5, label="Visited")
    pr, pc = zip(*result["path_to_object"])
    ax.plot(pc, pr, color="dodgerblue", linewidth=2, marker="o", markersize=4, label="A* path")
    ax.scatter(start[1], start[0], s=140, c="lime", edgecolors="black", zorder=3, label="Start")
    ax.scatter(object_pos[1], object_pos[0], s=160, c="orange", edgecolors="black",
               marker="s", zorder=3, label="Object")
    ax.set_title("1) Approach: start → object")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

    # Right: transport
    ax = axes[1]
    ax.imshow(grid, cmap="Greys", origin="upper")
    if result["visited_to_place"]:
        vr, vc = zip(*result["visited_to_place"])
        ax.scatter(vc, vr, s=30, c="skyblue", alpha=0.5, label="Visited")
    pr, pc = zip(*result["path_to_place"])
    ax.plot(pc, pr, color="purple", linewidth=2, marker="o", markersize=4, label="A* path")
    ax.scatter(object_pos[1], object_pos[0], s=160, c="orange", edgecolors="black",
               marker="s", zorder=3, label="Object")
    ax.scatter(place_pos[1], place_pos[0], s=180, c="red", edgecolors="black",
               marker="*", zorder=3, label="Place")
    ax.set_title("2) Transport: object → place")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

    history = " → ".join(state for _, state in result["fsm"].history)
    fig.suptitle(f"Pick-and-place challenge\nFSM: {history}")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")
    return fig


def main():
    rng = np.random.default_rng(42)
    grid = create_grid(size=10, obstacle_prob=0.2, seed=42)

    start = pick_free_cell(grid, rng)
    object_pos = pick_free_cell(grid, rng, avoid={start})
    place_pos = pick_free_cell(grid, rng, avoid={start, object_pos})

    result = simulate_pick_and_place(grid, start, object_pos, place_pos)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("\n".join(result["log"]) + "\n")
    print(f"Saved {LOG_PATH}")

    draw(grid, start, object_pos, place_pos, result)
    plt.show()


if __name__ == "__main__":
    main()
