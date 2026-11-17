"""Day 5 — Part 5: Small TCP delta moves (URSim → real transfer checklist helper).

Implements move_tcp_delta for cautious relative translations.
Default: dry-run / URSim. Real robot only with explicit flags.
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

OUT_DIR = Path(__file__).resolve().parent.parent / "deliver" / "day5"
OUT_PATH = OUT_DIR / "part5_move_tcp_delta.txt"

DEFAULT_IP = "127.0.0.1"
HOME_Q = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]

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


def move_tcp_delta(rtde_control, rtde_receive, dx, dy, dz, speed=0.05, acceleration=0.05):
    """Translate TCP by (dx, dy, dz) relative to the current pose.

    Keeps orientation. Uses IK + moveJ for safer small steps in URSim/real tests.
    Notebook idea: small relative translation only — keep speed low.
    """
    speed = min(abs(speed), 0.1)
    acceleration = min(abs(acceleration), 0.1)

    if rtde_receive.isProtectiveStopped() or rtde_receive.isEmergencyStopped():
        raise RuntimeError("Robot is in protective/emergency stop — abort.")

    current = list(rtde_receive.getActualTCPPose())
    target = list(current)
    target[0] += dx
    target[1] += dy
    target[2] += dz

    print(
        f"move_tcp_delta: {[round(v, 4) for v in current[:3]]} "
        f"+ ({dx:.4f},{dy:.4f},{dz:.4f}) -> {[round(v, 4) for v in target[:3]]}"
    )

    if not rtde_control.isPoseWithinSafetyLimits(target):
        raise RuntimeError(f"Target outside safety limits: {target}")
    if not rtde_control.getInverseKinematicsHasSolution(target):
        raise RuntimeError(f"No IK solution for target: {target}")

    q = rtde_control.getInverseKinematics(target)
    ok = rtde_control.moveJ(q, speed, acceleration)
    if not ok:
        raise RuntimeError("moveJ failed for TCP delta")
    return list(rtde_receive.getActualTCPPose())


def main():
    parser = argparse.ArgumentParser(description="Day 5 Part 5 — move_tcp_delta")
    parser.add_argument("--ip", default=DEFAULT_IP)
    parser.add_argument("--dx", type=float, default=0.02)
    parser.add_argument("--dy", type=float, default=0.0)
    parser.add_argument("--dz", type=float, default=0.0)
    parser.add_argument("--speed", type=float, default=0.05)
    parser.add_argument("--execute", action="store_true", help="Connect and move")
    parser.add_argument(
        "--i-understand",
        action="store_true",
        help="Required for non-localhost execution",
    )
    parser.add_argument("--home", action="store_true", help="Go to HOME_Q before the delta")
    args = parser.parse_args()

    print("Implemented move_tcp_delta(rtde_control, rtde_receive, dx, dy, dz, speed=0.05)")
    print("Checklist before real robot:")
    print("  - TCP verified")
    print("  - Speeds/accels reduced")
    print("  - Use small incremental motions")

    if not args.execute:
        print("\nDry-run only. Example:")
        print("  python3 day5/part5_move_tcp_delta.py --execute --dx 0.02 --home")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(
            "Day 5 Part 5 — move_tcp_delta defined (dry-run, no motion)\n"
            f"Requested delta: ({args.dx}, {args.dy}, {args.dz})\n"
        )
        print(f"Saved {OUT_PATH}")
        return

    is_local = args.ip in ("127.0.0.1", "localhost", "::1")
    if not is_local and not args.i_understand:
        print("Refusing non-localhost move without --i-understand")
        raise SystemExit(2)

    global rtde_c, rtde_r
    signal.signal(signal.SIGINT, on_sigint)
    try:
        rtde_c = RTDEControlInterface(args.ip)
        rtde_r = RTDEReceiveInterface(args.ip)
        print(f"Connected to {args.ip}")

        if args.home:
            print("Moving home first...")
            rtde_c.moveJ(HOME_Q, 0.15, 0.15)

        before = list(rtde_r.getActualTCPPose())
        after = move_tcp_delta(rtde_c, rtde_r, args.dx, args.dy, args.dz, speed=args.speed)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(
            "\n".join(
                [
                    "Day 5 Part 5 — move_tcp_delta executed",
                    f"IP: {args.ip}",
                    f"Delta: ({args.dx}, {args.dy}, {args.dz})",
                    f"Before: {before}",
                    f"After:  {after}",
                ]
            )
            + "\n"
        )
        print(f"Saved {OUT_PATH}")
    except Exception as exc:
        print("Error:", exc)
        raise SystemExit(1) from exc
    finally:
        disconnect()


if __name__ == "__main__":
    main()
