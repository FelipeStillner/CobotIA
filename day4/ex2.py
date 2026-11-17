"""Day 4 — Exercise 2: Implement a Q-learning agent on GridWorld."""

from collections import defaultdict
from pathlib import Path
import random
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ex1 import ACTION_NAMES, GridWorld, show_env

OUT_DIR = Path(__file__).resolve().parent.parent / "deliver" / "day4"
OUT_PATH = OUT_DIR / "ex2_qlearning.png"


def choose_action(Q, state, epsilon):
    """Epsilon-greedy action selection."""
    if random.random() < epsilon:
        return random.randint(0, 3)
    qvals = Q[state]
    max_q = max(qvals)
    # Break ties randomly among best actions
    best = [a for a, q in enumerate(qvals) if q == max_q]
    return random.choice(best)


def q_learning(env, episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1, max_steps=100):
    """Train Q(s,a) with tabular Q-learning.

    Update:
        Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
    """
    Q = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    returns = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        total = 0.0
        steps = 0

        while not done and steps < max_steps:
            action = choose_action(Q, state, epsilon)
            next_state, reward, done = env.step(action)
            total += reward

            old = Q[state][action]
            best_next = 0.0 if done else max(Q[next_state])
            td_target = reward + gamma * best_next
            Q[state][action] = old + alpha * (td_target - old)

            state = next_state
            steps += 1

        returns.append(total)
        if (ep + 1) % 100 == 0:
            avg = sum(returns[-100:]) / 100
            print(f"Episode {ep + 1:4d} | last return={total:6.1f} | avg100={avg:6.2f}")

    return Q, returns


def greedy_path(env, Q, max_steps=50):
    state = env.reset()
    path = [state]
    total = 0.0
    for _ in range(max_steps):
        action = choose_action(Q, state, epsilon=0.0)  # pure greedy
        state, reward, done = env.step(action)
        total += reward
        path.append(state)
        if done:
            break
    return path, total


def plot_training(returns, path, env, save_path=OUT_PATH):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Smoothed learning curve
    window = 20
    smooth = np.convolve(returns, np.ones(window) / window, mode="valid")
    axes[0].plot(returns, alpha=0.3, label="episode return")
    axes[0].plot(range(window - 1, len(returns)), smooth, label=f"avg{window}")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return")
    axes[0].set_title("Q-learning training")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Greedy path on grid
    grid = np.zeros((env.n_rows, env.n_cols))
    for r, c in env.obstacles:
        grid[r, c] = 1
    ax = axes[1]
    ax.imshow(grid, cmap="Greys", origin="upper")
    if path:
        pr, pc = zip(*path)
        ax.plot(pc, pr, color="dodgerblue", marker="o", linewidth=2, label="Greedy path")
    ax.scatter(env.start[1], env.start[0], s=140, c="lime", edgecolors="black", zorder=3, label="Start")
    ax.scatter(env.goal[1], env.goal[0], s=160, c="red", edgecolors="black", marker="*", zorder=3, label="Goal")
    ax.set_title("Greedy policy path")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")
    plt.show()


def main():
    random.seed(0)
    np.random.seed(0)

    env = GridWorld(
        n_rows=5,
        n_cols=5,
        start=(0, 0),
        goal=(4, 4),
        obstacles={(1, 1), (2, 2), (3, 1)},
    )

    Q, returns = q_learning(env, episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1)
    print("Trained Q sample for start state:", Q[env.start])
    for a, q in enumerate(Q[env.start]):
        print(f"  action {a} ({ACTION_NAMES[a]}): {q:.3f}")

    path, total = greedy_path(env, Q)
    print("Greedy path:", path)
    print("Greedy return:", total)

    plot_training(returns, path, env)


if __name__ == "__main__":
    main()
