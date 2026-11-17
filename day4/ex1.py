"""Day 4 — Exercise 1: Implement a GridWorld environment (MDP)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "deliver" / "day4"
OUT_PATH = OUT_DIR / "ex1_gridworld.png"

# Actions: 0=up, 1=right, 2=down, 3=left
ACTIONS = {
    0: (-1, 0),  # up
    1: (0, 1),   # right
    2: (1, 0),   # down
    3: (0, -1),  # left
}
ACTION_NAMES = {0: "up", 1: "right", 2: "down", 3: "left"}


class GridWorld:
    def __init__(self, n_rows=5, n_cols=5, start=(0, 0), goal=(4, 4), obstacles=None):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.start = start
        self.goal = goal
        self.obstacles = set(obstacles) if obstacles else set()
        self.reset()

    def reset(self):
        self.state = self.start
        return self.state

    def in_bounds(self, s):
        r, c = s
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def is_free(self, s):
        return self.in_bounds(s) and s not in self.obstacles

    def step(self, action):
        """Take action → (next_state, reward, done).

        Deterministic transitions:
        - Move into free cell → go there, reward -1 (step cost)
        - Hit wall/obstacle → stay put, reward -1
        - Reach goal → reward +10, done=True
        """
        if action not in ACTIONS:
            raise ValueError(f"Invalid action {action}; expected 0..3")

        dr, dc = ACTIONS[action]
        r, c = self.state
        nxt = (r + dr, c + dc)

        if not self.is_free(nxt):
            nxt = self.state  # bump into wall/obstacle: no move

        self.state = nxt

        if self.state == self.goal:
            return self.state, 10.0, True

        return self.state, -1.0, False


def show_env(env, path=None, save_path=OUT_PATH):
    grid = np.zeros((env.n_rows, env.n_cols))
    for r, c in env.obstacles:
        grid[r, c] = 1

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(grid, cmap="Greys", origin="upper")
    ax.set_xticks(np.arange(-0.5, env.n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.n_rows, 1), minor=True)
    ax.grid(which="minor", color="lightgray", linewidth=0.5)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    ax.scatter(env.start[1], env.start[0], s=160, c="lime", edgecolors="black", label="Start", zorder=3)
    ax.scatter(env.goal[1], env.goal[0], s=180, c="red", edgecolors="black", marker="*", label="Goal", zorder=3)

    if path:
        pr, pc = zip(*path)
        ax.plot(pc, pr, color="dodgerblue", marker="o", linewidth=2, label="Demo path")

    ax.set_title("GridWorld (Exercise 1)")
    ax.legend(loc="upper right")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")
    plt.show()


def main():
    env = GridWorld(
        n_rows=5,
        n_cols=5,
        start=(0, 0),
        goal=(4, 4),
        obstacles={(1, 1), (2, 2), (3, 1)},
    )
    print("Initial state:", env.reset())

    # Demo: walk right/down toward the goal (may bump obstacles)
    demo_actions = [1, 1, 2, 1, 2, 1, 2, 2, 1, 2]
    path = [env.state]
    total_reward = 0.0
    for a in demo_actions:
        nxt, reward, done = env.step(a)
        total_reward += reward
        path.append(nxt)
        print(f"action={a}({ACTION_NAMES[a]}) -> state={nxt}, reward={reward}, done={done}")
        if done:
            break

    print("Path:", path)
    print("Total reward:", total_reward)
    show_env(env, path=path)


if __name__ == "__main__":
    main()
