"""Day 3 — Exercise 4: Finite State Machine for robot decision-making."""

from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "deliver" / "day3"
OUT_PATH = OUT_DIR / "ex4_fsm.txt"


class RobotFSM:
    """Simple pick-and-place FSM.

    States: SEARCH → APPROACH → GRASP → TRANSPORT → PLACE → DONE
    Also handles failure back to SEARCH.
    """

    TRANSITIONS = {
        ("SEARCH", "object_detected"): "APPROACH",
        ("SEARCH", "timeout"): "SEARCH",
        ("APPROACH", "reached_object"): "GRASP",
        ("APPROACH", "lost_object"): "SEARCH",
        ("GRASP", "grasp_success"): "TRANSPORT",
        ("GRASP", "grasp_failed"): "SEARCH",
        ("TRANSPORT", "reached_place"): "PLACE",
        ("TRANSPORT", "collision"): "SEARCH",
        ("PLACE", "place_success"): "DONE",
        ("PLACE", "place_failed"): "TRANSPORT",
        ("DONE", "reset"): "SEARCH",
    }

    def __init__(self):
        self.state = "SEARCH"
        self.history = [("init", self.state)]

    def next(self, event):
        key = (self.state, event)
        if key not in self.TRANSITIONS:
            raise ValueError(
                f"Invalid transition: state={self.state!r}, event={event!r}. "
                f"Valid events from {self.state}: "
                f"{sorted(e for (s, e) in self.TRANSITIONS if s == self.state)}"
            )
        prev = self.state
        self.state = self.TRANSITIONS[key]
        self.history.append((event, self.state))
        print(f"{prev} --({event})--> {self.state}")
        return self.state


def demo_sequence():
    """Simulate a successful pick-and-place decision sequence."""
    fsm = RobotFSM()
    print(f"Initial state: {fsm.state}")

    events = [
        "object_detected",
        "reached_object",
        "grasp_success",
        "reached_place",
        "place_success",
    ]
    for event in events:
        fsm.next(event)

    print(f"Final state: {fsm.state}")
    return fsm


def demo_recovery():
    """Simulate losing the object, then recovering."""
    print("\n--- Recovery scenario ---")
    fsm = RobotFSM()
    fsm.next("object_detected")
    fsm.next("lost_object")  # back to SEARCH
    fsm.next("object_detected")
    fsm.next("reached_object")
    fsm.next("grasp_failed")  # back to SEARCH
    return fsm


def main():
    fsm = demo_sequence()
    demo_recovery()

    # Notebook-style call
    print("\n--- Notebook-style call ---")
    fsm2 = RobotFSM()
    fsm2.next("object_detected")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["FSM history (success run):"]
    for event, state in fsm.history:
        lines.append(f"  {event:20s} -> {state}")
    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
