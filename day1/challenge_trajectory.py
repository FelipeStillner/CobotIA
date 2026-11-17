"""Day 1 Challenge — log TCP pose during motion, save with Polars, and plot."""

import signal
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

ROBOT_IP = "127.0.0.1"
OUT_DIR = Path(__file__).resolve().parent / "deliver" / "day1"
PARQUET_PATH = OUT_DIR / "trajectory.parquet"
PLOT_PATH = OUT_DIR / "trajectory.png"

# Two waypoints so we get a visible path in TCP space
HOME_Q = [0.0, -1.57, 1.57, 0.0, 1.57, 0.0]
TARGET_Q = [-0.8, -1.2, 1.4, -0.5, 1.57, 0.3]

SPEED = 0.3
ACCEL = 0.3
SAMPLE_DT = 0.05  # seconds
COLUMNS = ["t", "x", "y", "z", "rx", "ry", "rz"]

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


def safe_move_async(target_q, speed=SPEED, acceleration=ACCEL):
    if len(target_q) != 6:
        raise ValueError("target_q must have 6 joints")
    speed = min(speed, 0.5)
    acceleration = min(acceleration, 0.5)
    if rtde_r.isProtectiveStopped() or rtde_r.isEmergencyStopped():
        raise RuntimeError("Robot is in protective/emergency stop")
    return rtde_c.moveJ(target_q, speed, acceleration, True)  # asynchronous


def log_until_steady(rows, t0):
    """Sample TCP pose until the robot finishes the async move.

    t0 is a shared start time so all motion segments share one timeline.
    """
    while True:
        pose = rtde_r.getActualTCPPose()  # [x, y, z, rx, ry, rz]
        rows.append({"t": time.time() - t0, "x": pose[0], "y": pose[1], "z": pose[2],
                     "rx": pose[3], "ry": pose[4], "rz": pose[5]})
        # isSteady lives on the control interface, not receive
        if rtde_c.isSteady():
            pose = rtde_r.getActualTCPPose()
            rows.append({"t": time.time() - t0, "x": pose[0], "y": pose[1], "z": pose[2],
                         "rx": pose[3], "ry": pose[4], "rz": pose[5]})
            break
        time.sleep(SAMPLE_DT)


def to_dataframe(rows) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=COLUMNS)


def save_parquet(df: pl.DataFrame):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(PARQUET_PATH)
    print(f"Saved {df.height} samples → {PARQUET_PATH}")


def plot_trajectory(df: pl.DataFrame):
    ts = df["t"].to_list()
    xs = df["x"].to_list()
    ys = df["y"].to_list()
    zs = df["z"].to_list()

    fig = plt.figure(figsize=(10, 4))

    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot(xs, ys, zs, color="C0")
    ax3d.scatter(xs[0], ys[0], zs[0], color="green", label="start")
    ax3d.scatter(xs[-1], ys[-1], zs[-1], color="red", label="end")
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.set_title("TCP trajectory")
    ax3d.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(ts, xs, label="x")
    ax2.plot(ts, ys, label="y")
    ax2.plot(ts, zs, label="z")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Position [m]")
    ax2.set_title("TCP vs time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"Saved plot → {PLOT_PATH}")
    plt.show()


def main():
    global rtde_c, rtde_r
    signal.signal(signal.SIGINT, on_sigint)

    print(f"Connecting to {ROBOT_IP}...")
    rtde_c = RTDEControlInterface(ROBOT_IP)
    rtde_r = RTDEReceiveInterface(ROBOT_IP)
    print("Connected. Press Ctrl+C anytime to disconnect.")

    rows = []
    t0 = time.time()

    print("Moving to home...")
    safe_move_async(HOME_Q)
    log_until_steady(rows, t0)

    print("Moving to target...")
    safe_move_async(TARGET_Q)
    log_until_steady(rows, t0)

    print("Returning home...")
    safe_move_async(HOME_Q)
    log_until_steady(rows, t0)

    df = to_dataframe(rows)
    save_parquet(df)
    plot_trajectory(df)
    disconnect()
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("Error:", exc)
        disconnect()
        sys.exit(1)
