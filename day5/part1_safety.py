"""Day 5 — Part 1: Safety checklist & pre-flight checks.

Run this before connecting to the real UR5.
"""

from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "deliver" / "day5"
OUT_PATH = OUT_DIR / "part1_safety_checklist.txt"

CHECKS = [
    ("E-stop reachable", "Emergency stop is within immediate reach of the operator."),
    ("Workspace clear", "No people, tools, or loose objects inside the robot reach."),
    ("TCP / tool set", "Tool Center Point is defined; payload and tool settings are correct."),
    ("Speed limits", "Speed and acceleration are reduced for first real-robot tests."),
    ("URSim validated", "Same motion/script was tested successfully on URSim first."),
    ("Remote Control", "Robot is powered, enabled, and in Remote Control (or ready for it)."),
    ("Network known", "Robot IP is known; only one RTDE client will connect."),
    ("Instructor OK", "Instructor/lab supervisor has approved real-robot execution."),
]


def ask_yes(prompt: str) -> bool:
    while True:
        ans = input(f"{prompt} [y/n]: ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("  Please answer y or n.")


def safety_checks(interactive: bool = True) -> bool:
    """Run pre-flight safety checks.

    Returns True only if every check passes.
    In non-interactive mode, prints the checklist and returns False
    (forces a conscious interactive confirmation before real robot use).
    """
    print("=" * 60)
    print("Day 5 — Safety Checklist & Pre-flight Checks")
    print("=" * 60)

    results = []
    for i, (title, detail) in enumerate(CHECKS, start=1):
        print(f"\nCheck {i}: {title}")
        print(f"  → {detail}")
        if interactive:
            ok = ask_yes("  Confirm")
        else:
            print("  (non-interactive: not confirmed)")
            ok = False
        results.append((title, ok))
        print(f"  {'OK' if ok else 'FAIL'}")

    print("\n" + "-" * 60)
    all_ok = all(ok for _, ok in results)
    if all_ok:
        print("ALL CHECKS PASSED — you may proceed to connection tests.")
    else:
        failed = [t for t, ok in results if not ok]
        print("CHECKS FAILED — do NOT run the real robot yet:")
        for t in failed:
            print(f"  - {t}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["Day 5 Part 1 — Safety checklist results", ""]
    for title, ok in results:
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {title}")
    lines.append("")
    lines.append(f"Overall: {'PASS' if all_ok else 'FAIL'}")
    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"\nSaved {OUT_PATH}")
    return all_ok


def main():
    # Interactive confirmation — required before real robot work
    ok = safety_checks(interactive=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
