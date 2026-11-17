"""Day 5 — Part 6: Pick-and-place pipeline template.

Integrates: perception → affine mapping → simple approach → FSM → execution.
Default is simulation/dry-run. Use --execute for URSim after Parts 1–5 checks.
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

import numpy as np
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "day3"))

from part4_calibration import compute_affine_transform, demo_correspondences, pixel_to_robot
from part5_move_tcp_delta import move_tcp_delta

# Reuse Day 3 FSM
from ex4 import RobotFSM  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "deliver" / "day5"
OUT_PATH = OUT_DIR / "part6_pipeline.txt"

HOME_Q = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
APPROACH_Z = 0.25
PLACE_OFFSET = (0.05, 0.0)  # place 5 cm in +X from pick XY (demo)

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
    print("\nCtrl+C — disconnecting...")
    if rtde_c is not None:
        try:
            rtde_c.stopJ(2.0)
        except Exception:
            pass
    disconnect()
    sys.exit(0)


def camera_capture_stub():
    """Replace with RealSense / webcam detection. Returns (px, py, label)."""
    # Demo: center-ish pixel of a detected object
    return 320.0, 240.0, "demo_object"


def detect_object(camera_capture) -> tuple[float, float, str]:
    if callable(camera_capture):
        return camera_capture()
    return camera_capture


def plan_straight_approach(current_xy, target_xy, n_steps=4):
    """Simple straight-line waypoints in XY (no grid obstacles)."""
    x0, y0 = current_xy
    x1, y1 = target_xy
    return [
        (x0 + (x1 - x0) * t / n_steps, y0 + (y1 - y0) * t / n_steps)
        for t in range(1, n_steps + 1)
    ]


def run_pick_and_place_pipeline(rtde_control, rtde_receive, camera_capture, affine_A,
                                execute_motions: bool = True, speed: float = 0.05):
    """Skeleton pipeline: detect → map → plan → FSM → execute."""
    fsm = RobotFSM()
    log = [f"FSM start: {fsm.state}"]

    # 1) detect object in image -> px, py
    px, py, label = detect_object(camera_capture)
    log.append(f"Detected {label} at pixel ({px}, {py})")
    fsm.next("object_detected")

    # 2) convert to robot XY using affine_A
    obj_x, obj_y = pixel_to_robot(affine_A, px, py)
    place_x = obj_x + PLACE_OFFSET[0]
    place_y = obj_y + PLACE_OFFSET[1]
    log.append(f"Mapped pick XY=({obj_x:.4f}, {obj_y:.4f}), place XY=({place_x:.4f}, {place_y:.4f})")

    # Current TCP XY as planning start
    tcp = list(rtde_receive.getActualTCPPose())
    start_xy = (tcp[0], tcp[1])

    # 3) plan path (straight-line approach)
    approach = plan_straight_approach(start_xy, (obj_x, obj_y), n_steps=4)
    transport = plan_straight_approach((obj_x, obj_y), (place_x, place_y), n_steps=4)
    log.append(f"Approach waypoints: {[(round(x,3), round(y,3)) for x,y in approach]}")
    log.append(f"Transport waypoints: {[(round(x,3), round(y,3)) for x,y in transport]}")

    def go_xy(x, y, z=APPROACH_Z):
        cur = list(rtde_receive.getActualTCPPose())
        dx, dy, dz = x - cur[0], y - cur[1], z - cur[2]
        if execute_motions:
            move_tcp_delta(rtde_control, rtde_receive, dx, dy, dz, speed=speed)
        else:
            log.append(f"  [dry] would move_tcp_delta ({dx:.4f}, {dy:.4f}, {dz:.4f})")

    # 4) FSM sequence: move -> grasp -> move -> release
    for wp in approach:
        go_xy(*wp)
    fsm.next("reached_object")
    log.append("Simulated grasp")
    fsm.next("grasp_success")

    for wp in transport:
        go_xy(*wp)
    fsm.next("reached_place")
    log.append("Simulated release")
    fsm.next("place_success")

    log.append(f"FSM final: {fsm.state}")
    for line in log:
        print(line)
    return {"fsm": fsm, "log": log, "pick": (obj_x, obj_y), "place": (place_x, place_y)}


def main():
    parser = argparse.ArgumentParser(description="Day 5 Part 6 — pick-and-place pipeline")
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--execute", action="store_true", help="Run motions on robot/URSim")
    parser.add_argument("--i-understand", action="store_true")
    parser.add_argument("--speed", type=float, default=0.05)
    args = parser.parse_args()

    print("Defined run_pick_and_place_pipeline(...)")

    # Calibration from Part 4 (demo correspondences — replace with lab data)
    pixels, robots = demo_correspondences()
    affine_A = compute_affine_transform(pixels, robots)
    print("Using affine_A from Part 4 demo correspondences")

    is_local = args.ip in ("127.0.0.1", "localhost", "::1")
    if args.execute and not is_local and not args.i_understand:
        print("Refusing non-localhost execution without --i-understand")
        raise SystemExit(2)

    global rtde_c, rtde_r
    signal.signal(signal.SIGINT, on_sigint)

    # Always need a receive connection for current pose in the template.
    # If not --execute, still connect read-only when possible; else dry fully.
    try:
        if args.execute:
            rtde_c = RTDEControlInterface(args.ip)
            rtde_r = RTDEReceiveInterface(args.ip)
            print(f"Connected to {args.ip}")
            print("Moving home...")
            rtde_c.moveJ(HOME_Q, 0.15, 0.15)
            result = run_pick_and_place_pipeline(
                rtde_c, rtde_r, camera_capture_stub, affine_A,
                execute_motions=True, speed=args.speed,
            )
        else:
            # Offline dry-run with a fake receive pose
            class FakeReceive:
                def getActualTCPPose(self):
                    return [0.35, -0.15, APPROACH_Z, 0.0, 0.0, 0.0]

                def isProtectiveStopped(self):
                    return False

                def isEmergencyStopped(self):
                    return False

            class FakeControl:
                pass

            print("Dry-run (no robot connection). Pass --execute to run on URSim.")
            result = run_pick_and_place_pipeline(
                FakeControl(), FakeReceive(), camera_capture_stub, affine_A,
                execute_motions=False, speed=args.speed,
            )
    except Exception as exc:
        print("Error:", exc)
        disconnect()
        raise SystemExit(1) from exc
    finally:
        disconnect()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        "\n".join(
            [
                "Day 5 Part 6 — pick-and-place pipeline",
                f"Execute motions: {args.execute}",
                f"Pick: {result['pick']}",
                f"Place: {result['place']}",
                f"FSM: {result['fsm'].state}",
                "",
                *result["log"],
            ]
        )
        + "\n"
    )
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
