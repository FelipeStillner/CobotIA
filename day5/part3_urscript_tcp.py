"""Day 5 — Part 3: Send URScript over TCP (port 30002).

Defines send_urscript() as in the notebook. By default this script does NOT
send anything to a robot — use --execute only with instructor permission.

Examples:
  # Dry-run (default): show what would be sent
  python3 day5/part3_urscript_tcp.py

  # URSim — harmless popup only
  python3 day5/part3_urscript_tcp.py --ip 127.0.0.1 --execute

  # Real robot — ONLY with instructor OK
  python3 day5/part3_urscript_tcp.py --ip 192.168.0.10 --execute --i-understand
"""

from __future__ import annotations

import argparse
import socket
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "deliver" / "day5"
OUT_PATH = OUT_DIR / "part3_urscript_tcp.txt"

# Secondary client interface — URScript commands
URSCRIPT_PORT = 30002

# Harmless default: show a popup on PolyScope (no motion)
SAFE_DEMO_SCRIPT = 'popup("Day5 Part3 URScript OK", title="CobotIA", blocking=False)'


def send_urscript(robot_ip: str, port: int, script: str, timeout: float = 2.0) -> None:
    """Open a TCP socket and send one URScript line to the robot/controller."""
    if not script.endswith("\n"):
        script = script + "\n"

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect((robot_ip, port))
        s.sendall(script.encode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Day 5 Part 3 — URScript via TCP")
    parser.add_argument("--ip", default="127.0.0.1", help="Robot / URSim IP")
    parser.add_argument("--port", type=int, default=URSCRIPT_PORT, help="URScript port (30002)")
    parser.add_argument(
        "--script",
        default=SAFE_DEMO_SCRIPT,
        help="URScript command (default: harmless popup, no motion)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually send the script (otherwise dry-run only)",
    )
    parser.add_argument(
        "--i-understand",
        action="store_true",
        help="Required together with --execute when IP is not localhost",
    )
    args = parser.parse_args()

    print("Defined send_urscript(robot_ip, port, script)")
    print(f"Target: {args.ip}:{args.port}")
    print(f"Script: {args.script}")

    is_localhost = args.ip in ("127.0.0.1", "localhost", "::1")
    executed = False
    error = None

    if not args.execute:
        print("\nDry-run only — nothing sent.")
        print("To send (URSim):  python3 day5/part3_urscript_tcp.py --execute")
        print(
            "To send (real):   python3 day5/part3_urscript_tcp.py "
            "--ip <ROBOT_IP> --execute --i-understand"
        )
    else:
        if not is_localhost and not args.i_understand:
            print(
                "\nRefusing to send to a non-localhost IP without --i-understand.\n"
                "Get instructor permission first, then rerun with that flag."
            )
            raise SystemExit(2)

        try:
            send_urscript(args.ip, args.port, args.script)
            executed = True
            print("\nURScript sent successfully.")
            if "move" in args.script.lower():
                print("Warning: script looks motion-related — watch the robot closely.")
        except Exception as exc:
            error = str(exc)
            print("Send failed:", exc)
            raise SystemExit(1) from exc

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "Day 5 Part 3 — URScript via TCP",
        f"IP: {args.ip}",
        f"Port: {args.port}",
        f"Script: {args.script}",
        f"Executed: {executed}",
        f"Error: {error}",
        "Note: Do NOT run motion scripts on the real robot without instructor OK.",
    ]
    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
