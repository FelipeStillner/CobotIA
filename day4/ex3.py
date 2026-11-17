"""Day 4 — Exercise 3: Derive greedy policy from Q and visualize it."""

from pathlib import Path
import random
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ex1 import ACTION_NAMES, ACTIONS, GridWorld
from ex2 import q_learning

OUT_DIR = Path(__file__).resolve().parent.parent / "deliver" / "day4"
OUT_PATH = OUT_DIR / "ex3_policy.png"


def greedy_policy(Q, state):
    """Pick the action with highest Q-value in this state."""
    qvals = Q.get(state, [0.0, 0.0, 0.0, 0.0])
    return max(range(4), key=lambda a: qvals[a])


def simulate_episode(env, Q, max_steps=50):
    """Follow the greedy policy for one episode and record the path."""
    state = env.reset()
    path = [state]
    rewards = []
    actions = []

    for _ in range(max_steps):
        a = greedy_policy(Q, state)
        state, r, done = env.step(a)
        path.append(state)
        rewards.append(r)
        actions.append(a)
        if done:
            break

    return path, actions, rewards


def policy_arrows(env, Q):
    """For every free non-goal cell, return (row, col, drow, dcol) for quiver."""
    arrows = []
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            s = (r, c)
            if s in env.obstacles or s == env.goal:
                continue
            a = greedy_policy(Q, s)
            dr, dc = ACTIONS[a]
            arrows.append((c, r, dc * 0.35, dr * 0.35))  # x,y,u,v in plot coords
    return arrows


def visualize(env, Q, path, save_path=OUT_PATH):
    grid = np.zeros((env.n_rows, env.n_cols))
    for r, c in env.obstacles:
        grid[r, c] = 1

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: greedy policy arrows
    ax = axes[0]
    ax.imshow(grid, cmap="Greys", origin="upper")
    arrows = policy_arrows(env, Q)
    if arrows:
        xs, ys, us, vs = zip(*arrows)
        ax.quiver(xs, ys, us, vs, angles="xy", scale_units="xy", scale=1,
                  color="tab:blue", width=0.012)
    ax.scatter(env.start[1], env.start[0], s=140, c="lime", edgecolors="black",
               zorder=3, label="Start")
    ax.scatter(env.goal[1], env.goal[0], s=160, c="red", edgecolors="black",
               marker="*", zorder=3, label="Goal")
    ax.set_title("Greedy policy (from Q)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

    # Right: one simulated episode
    ax = axes[1]
    ax.imshow(grid, cmap="Greys", origin="upper")
    if path:
        pr, pc = zip(*path)
        ax.plot(pc, pr, color="dodgerblue", marker="o", linewidth=2, label="Episode path")
    ax.scatter(env.start[1], env.start[0], s=140, c="lime", edgecolors="black",
               zorder=3, label="Start")
    ax.scatter(env.goal[1], env.goal[0], s=160, c="red", edgecolors="black",
               marker="*", zorder=3, label="Goal")
    ax.set_title("Simulated greedy episode")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.suptitle("Exercise 3 — Policy evaluation & visualization")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")
    plt.show()


def main():
    random.seed(0)

    env = GridWorld(
        n_rows=5,
        n_cols=5,
        start=(0, 0),
        goal=(4, 4),
        obstacles={(1, 1), (2, 2), (3, 1)},
    )

    print("Training Q-learning...")
    Q, _ = q_learning(env, episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1)

    path, actions, rewards = simulate_episode(env, Q)
    print("Path:", path)
    print("Actions:", [ACTION_NAMES[a] for a in actions])
    print("Total reward:", sum(rewards))
    print("Reached goal:", path[-1] == env.goal)

    # Notebook-style print
    print("\nPer-state greedy action near the path:")
    for s in path[:-1]:
        a = greedy_policy(Q, s)
        print(f"  {s} -> {a} ({ACTION_NAMES[a]})  Q={Q[s]}")

    visualize(env, Q, path)


if __name__ == "__main__":
    main()
