"""Day 4 — Exercise 4: Value Iteration (Dynamic Programming) on GridWorld."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ex1 import ACTION_NAMES, ACTIONS, GridWorld

OUT_DIR = Path(__file__).resolve().parent.parent / "deliver" / "day4"
OUT_PATH = OUT_DIR / "ex4_value_iteration.png"


def all_states(env):
    return [
        (r, c)
        for r in range(env.n_rows)
        for c in range(env.n_cols)
        if (r, c) not in env.obstacles
    ]


def transition(env, state, action):
    """Deterministic next_state, reward, done (does not mutate env)."""
    if state == env.goal:
        return state, 0.0, True

    dr, dc = ACTIONS[action]
    nxt = (state[0] + dr, state[1] + dc)
    if not env.is_free(nxt):
        nxt = state

    if nxt == env.goal:
        return nxt, 10.0, True
    return nxt, -1.0, False


def value_iteration(env, gamma=0.99, theta=1e-4):
    """Compute optimal state values V(s) by synchronous value iteration.

    Bellman optimality update (deterministic):
        V(s) = max_a [ r(s,a) + gamma * V(s') ]
    """
    states = all_states(env)
    V = {s: 0.0 for s in states}

    iteration = 0
    while True:
        iteration += 1
        delta = 0.0
        V_new = dict(V)

        for s in states:
            if s == env.goal:
                V_new[s] = 0.0
                continue

            q_sa = []
            for a in range(4):
                nxt, reward, done = transition(env, s, a)
                bootstrap = 0.0 if done else V[nxt]
                q_sa.append(reward + gamma * bootstrap)
            V_new[s] = max(q_sa)
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new
        if delta < theta:
            break

    print(f"Value iteration converged in {iteration} sweeps (delta < {theta})")
    return V


def policy_from_V(env, V, gamma=0.99):
    """Greedy policy w.r.t. V: π(s) = argmax_a [r + γ V(s')]."""
    policy = {}
    Q = {}
    for s in all_states(env):
        if s == env.goal:
            continue
        qvals = []
        for a in range(4):
            nxt, reward, done = transition(env, s, a)
            bootstrap = 0.0 if done else V[nxt]
            qvals.append(reward + gamma * bootstrap)
        Q[s] = qvals
        policy[s] = max(range(4), key=lambda a: qvals[a])
    return policy, Q


def simulate_with_policy(env, policy, max_steps=50):
    state = env.reset()
    path = [state]
    total = 0.0
    for _ in range(max_steps):
        a = policy[state]
        state, r, done = env.step(a)
        path.append(state)
        total += r
        if done:
            break
    return path, total


def visualize(env, V, policy, path, save_path=OUT_PATH):
    # Heatmap of V
    heat = np.full((env.n_rows, env.n_cols), np.nan)
    for (r, c), v in V.items():
        heat[r, c] = v
    for r, c in env.obstacles:
        heat[r, c] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    im = ax.imshow(heat, cmap="viridis", origin="upper")
    for (r, c), v in V.items():
        if (r, c) in env.obstacles:
            continue
        ax.text(c, r, f"{v:.1f}", ha="center", va="center", color="white", fontsize=8)
    ax.scatter(env.start[1], env.start[0], s=120, c="lime", edgecolors="black", zorder=3, label="Start")
    ax.scatter(env.goal[1], env.goal[0], s=140, c="red", edgecolors="black", marker="*", zorder=3, label="Goal")
    ax.set_title("Value function V(s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1]
    grid = np.zeros((env.n_rows, env.n_cols))
    for r, c in env.obstacles:
        grid[r, c] = 1
    ax.imshow(grid, cmap="Greys", origin="upper")
    for s, a in policy.items():
        dr, dc = ACTIONS[a]
        ax.quiver(s[1], s[0], dc * 0.35, dr * 0.35, angles="xy",
                  scale_units="xy", scale=1, color="tab:blue", width=0.012)
    if path:
        pr, pc = zip(*path)
        ax.plot(pc, pr, color="orange", marker="o", linewidth=2, label="VI path")
    ax.scatter(env.start[1], env.start[0], s=120, c="lime", edgecolors="black", zorder=3, label="Start")
    ax.scatter(env.goal[1], env.goal[0], s=140, c="red", edgecolors="black", marker="*", zorder=3, label="Goal")
    ax.set_title("Greedy policy from V")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.suptitle("Exercise 4 — Value Iteration")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
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

    V = value_iteration(env, gamma=0.99, theta=1e-4)
    print("Sample V at start:", V.get(env.start))
    print("Sample V near goal (4,3):", V.get((4, 3)))

    policy, Q = policy_from_V(env, V)
    path, total = simulate_with_policy(env, policy)
    print("VI greedy path:", path)
    print("VI return:", total)
    print("Start action:", policy[env.start], ACTION_NAMES[policy[env.start]])

    # Notebook-style check
    print("\nNotebook check: V at start =", V.get(env.start))

    visualize(env, V, policy, path)


if __name__ == "__main__":
    main()
