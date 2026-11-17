"""Day 4 Final Challenge — Drive URSim with a trained GridWorld Q-policy.

1. Train Q-learning on GridWorld
2. Extract greedy action sequence
3. Map actions → small TCP deltas (Ex5)
4. Execute those deltas on URSim ( cautiously )
"""

from __future__ import annotations

import signal
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import random
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ex1 import ACTION_NAMES, GridWorld
from ex2 import q_learning
from ex3 import simulate_episode
from ex5 import grid_action_to_tcp_delta

OUT_DIR = Path(__file__).resolve().parent.parent / "deliver" / "day4"
FIG_PATH = OUT_DIR / "challenge_ursim_qpolicy.png"
LOG_PATH = OUT_DIR / "challenge_ursim_qpolicy.txt"

ROBOT_IP = "127.0.0.1"
HOME_Q = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
STEP_SIZE = 0.03   # meters per grid action (keep small!)
SPEED = 0.12
ACCEL = 0.12

rtde_c = None
rtde_r = None


def disconnect():
    global rtde_c, rtde_r
    for name, client in (("control", rtde_c), ("receive", rtde_r)):
        if client is None:
            continue
        try:
            client.disconnect()
            print(f"Disconnected {name}")
        except Exception as exc:
            print(f"Error disconnecting {name}: {exc}")
    rtde_c = None
    rtde_r = None


def on_sigint(signum, frame):
    print("\nCtrl+C — stopping and disconnecting...")
    if rtde_c is not None:
        try:
            rtde_c.stopJ(2.0)
        except Exception:
            pass
    disconnect()
    sys.exit(0)


def ensure_robot_ready():
    if rtde_r.isProtectiveStopped() or rtde_r.isEmergencyStopped():
        raise RuntimeError(
            "Robot in protective/emergency stop. "
            "In PolyScope: Enable Robot + Remote Control, then rerun."
        )


def safe_moveJ(q, speed=SPEED, acceleration=ACCEL):
    ensure_robot_ready()
    ok = rtde_c.moveJ(q, min(speed, 0.3), min(acceleration, 0.3))
    if not ok:
        raise RuntimeError("moveJ failed (path rejected)")
    return ok


def go_home():
    print(f"Moving to home joints: {HOME_Q}")
    safe_moveJ(HOME_Q)
    pose = list(rtde_r.getActualTCPPose())
    print(f"Home TCP: {[round(v, 4) for v in pose]}")
    return pose


def apply_tcp_delta(dx, dy, dz):
    """Apply a small TCP translation, keeping current orientation (via IK + moveJ)."""
    current = list(rtde_r.getActualTCPPose())
    target = list(current)
    target[0] += dx
    target[1] += dy
    target[2] += dz

    if not rtde_c.isPoseWithinSafetyLimits(target):
        raise RuntimeError(f"Target outside safety limits: {target}")
    if not rtde_c.getInverseKinematicsHasSolution(target):
        raise RuntimeError(f"No IK solution for target: {target}")

    q = rtde_c.getInverseKinematics(target)
    safe_moveJ(q)
    return list(rtde_r.getActualTCPPose())


def train_and_get_actions():
    random.seed(0)
    env = GridWorld(
        n_rows=5,
        n_cols=5,
        start=(0, 0),
        goal=(4, 4),
        obstacles={(1, 1), (2, 2), (3, 1)},
    )
    print("Training Q-learning on GridWorld...")
    Q, _ = q_learning(env, episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1)
    path, actions, rewards = simulate_episode(env, Q)
    print("Greedy grid path:", path)
    print("Actions:", [ACTION_NAMES[a] for a in actions])
    print("Grid return:", sum(rewards))
    return path, actions


def execute_on_ursim(actions, step_size=STEP_SIZE):
    global rtde_c, rtde_r
    signal.signal(signal.SIGINT, on_sigint)

    print(f"\nConnecting to URSim at {ROBOT_IP}...")
    rtde_c = RTDEControlInterface(ROBOT_IP)
    rtde_r = RTDEReceiveInterface(ROBOT_IP)
    print("Connected. Press Ctrl+C anytime to stop.")
    ensure_robot_ready()

    go_home()
    tcp_path = [list(rtde_r.getActualTCPPose())]
    log = []

    for i, action in enumerate(actions, start=1):
        dx, dy, dz = grid_action_to_tcp_delta(action, step_size=step_size)
        msg = (
            f"Step {i}/{len(actions)}: {ACTION_NAMES[action]} "
            f"delta=({dx:.3f}, {dy:.3f}, {dz:.3f})"
        )
        print(msg)
        log.append(msg)
        pose = apply_tcp_delta(dx, dy, dz)
        tcp_path.append(pose)
        print(f"  TCP -> {[round(v, 4) for v in pose[:3]]}")

    print("Finished Q-policy execution on URSim.")
    disconnect()
    return tcp_path, log


def save_outputs(grid_path, actions, tcp_path, log):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "Day 4 challenge — Q-policy → URSim",
        f"Grid path: {grid_path}",
        f"Actions: {[ACTION_NAMES[a] for a in actions]}",
        f"Step size: {STEP_SIZE} m",
        "",
        *log,
        "",
        "TCP waypoints (x,y,z):",
    ]
    for p in tcp_path:
        lines.append(f"  ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})")
    LOG_PATH.write_text("\n".join(lines) + "\n")
    print(f"Saved {LOG_PATH}")

    xs = [p[0] for p in tcp_path]
    ys = [p[1] for p in tcp_path]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(xs, ys, "-o", color="dodgerblue", label="Executed TCP path")
    ax.scatter(xs[0], ys[0], c="lime", s=120, edgecolors="black", zorder=3, label="Start")
    ax.scatter(xs[-1], ys[-1], c="red", s=140, marker="*", edgecolors="black", zorder=3, label="End")
    ax.set_aspect("equal")
    ax.set_xlabel("TCP X [m]")
    ax.set_ylabel("TCP Y [m]")
    ax.set_title("Day 4 challenge — Q-policy on URSim")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    print(f"Saved {FIG_PATH}")
    plt.close(fig)


def main():
    grid_path, actions = train_and_get_actions()

    print("\nPlanned TCP deltas:")
    for a in actions:
        print(f"  {ACTION_NAMES[a]:5s} -> {grid_action_to_tcp_delta(a, STEP_SIZE)}")

    tcp_path, log = execute_on_ursim(actions, step_size=STEP_SIZE)
    save_outputs(grid_path, actions, tcp_path, log)
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("Error:", exc)
        disconnect()
        sys.exit(1)
