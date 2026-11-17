"""Day 5 — Part 2: Connect to UR5 via RTDE and read state (no moves).

Default IP is URSim localhost. Pass the real robot IP when ready, e.g.:
  python3 day5/part2_rtde_connect.py 192.168.0.10
"""

from __future__ import annotations

import sys
from pathlib import Path

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

OUT_DIR = Path(__file__).resolve().parent.parent / "deliver" / "day5"
OUT_PATH = OUT_DIR / "part2_rtde_connect.txt"

# Notebook default placeholder — override via CLI arg
DEFAULT_IP = "127.0.0.1"  # safe default: URSim
NOTEBOOK_EXAMPLE_IP = "192.168.0.10"


def connect(robot_ip: str):
    """Open RTDE control + receive. Caller must disconnect."""
    print(f"Connecting to RTDE at {robot_ip} ...")
    rtde_c = RTDEControlInterface(robot_ip)
    rtde_r = RTDEReceiveInterface(robot_ip)
    print("Connected.")
    return rtde_c, rtde_r


def read_state(rtde_r: RTDEReceiveInterface) -> dict:
    """Read-only robot state (no motion commands)."""
    q = list(rtde_r.getActualQ())
    tcp = list(rtde_r.getActualTCPPose())
    mode = rtde_r.getRobotMode()
    protective = rtde_r.isProtectiveStopped()
    emergency = rtde_r.isEmergencyStopped()
    return {
        "joints_rad": q,
        "tcp_pose": tcp,
        "robot_mode": mode,
        "protective_stop": protective,
        "emergency_stop": emergency,
    }


def print_state(state: dict):
    print("\n--- Receive test (read-only) ---")
    print("Joints q [rad]:", [round(v, 4) for v in state["joints_rad"]])
    print("TCP pose [x,y,z,rx,ry,rz]:", [round(v, 4) for v in state["tcp_pose"]])
    print("Robot mode:", state["robot_mode"])
    print("Protective stop:", state["protective_stop"])
    print("Emergency stop:", state["emergency_stop"])


def disconnect(rtde_c, rtde_r):
    for name, client in (("control", rtde_c), ("receive", rtde_r)):
        if client is None:
            continue
        try:
            client.disconnect()
            print(f"Disconnected {name}")
        except Exception as exc:
            print(f"Error disconnecting {name}: {exc}")


def main():
    robot_ip = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IP
    if robot_ip == NOTEBOOK_EXAMPLE_IP:
        print(
            f"Note: using notebook example IP {NOTEBOOK_EXAMPLE_IP}. "
            "Confirm this is your real robot before proceeding."
        )

    rtde_c = rtde_r = None
    try:
        rtde_c, rtde_r = connect(robot_ip)
        # Part 2 requirement: test reading joint positions — do NOT move
        state = read_state(rtde_r)
        print_state(state)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        lines = [
            "Day 5 Part 2 — RTDE connect (read-only)",
            f"IP: {robot_ip}",
            f"Joints: {state['joints_rad']}",
            f"TCP: {state['tcp_pose']}",
            f"Robot mode: {state['robot_mode']}",
            f"Protective stop: {state['protective_stop']}",
            f"Emergency stop: {state['emergency_stop']}",
            "Moves executed: NO",
        ]
        OUT_PATH.write_text("\n".join(lines) + "\n")
        print(f"\nSaved {OUT_PATH}")
        print("Part 2 OK — connection + receive test only (no moves).")
    except Exception as exc:
        print("Connection failed or RTDE libs not available:", exc)
        raise SystemExit(1)
    finally:
        disconnect(rtde_c, rtde_r)


if __name__ == "__main__":
    main()
