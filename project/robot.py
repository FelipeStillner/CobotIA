"""RTDE link to the UR5. Sends only poses that stay on the hover plane."""

from __future__ import annotations

from config import MAX_STEP_M, ROBOT_IP, SERVO_ACCEL, SERVO_SPEED, TCP_ORIENTATION
from safety import enforce_hover, limit_step


class Robot:
    def __init__(self, ip: str = ROBOT_IP):
        self.ip = ip
        self.control = None
        self.receive = None
        self._printed_xy: tuple[float, float] | None = None
        self._last_printed: tuple[float, ...] | None = None

    def connect(self) -> None:
        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface

        print(f"Connecting to UR5 at {self.ip} ...")
        self.control = RTDEControlInterface(self.ip)
        self.receive = RTDEReceiveInterface(self.ip)
        pose = self.receive.getActualTCPPose()
        print(
            "Connected. TCP",
            [round(v, 4) for v in pose],
        )

    def disconnect(self) -> None:
        if self.control is not None:
            try:
                self.control.servoStop()
            except Exception:
                pass
        for name, client in (("control", self.control), ("receive", self.receive)):
            if client is None:
                continue
            try:
                client.disconnect()
                print(f"Disconnected {name}")
            except Exception as exc:
                print(f"Error disconnecting {name}: {exc}")
        self.control = None
        self.receive = None

    def hold(self) -> None:
        """Stop a servo stream and leave the arm where it is."""
        if self.control is None:
            return
        try:
            self.control.servoStop()
        except Exception:
            pass

    def command_pose(self, target_xyz: list[float], current_xy: tuple[float, float]) -> list[float]:
        """Step-limited hover pose. Z is forced onto the hover plane."""
        x, y = limit_step(current_xy, (target_xyz[0], target_xyz[1]), MAX_STEP_M)
        return enforce_hover([x, y, target_xyz[2], *TCP_ORIENTATION])

    def print_command(self, target_xyz: list[float]) -> str:
        """Print the servoL call. Does not connect or move the robot.

        The first pose is the full target, because the live TCP is unknown.
        Later poses step at most MAX_STEP_M from the last printed command.
        """
        if self._printed_xy is None:
            current_xy = (target_xyz[0], target_xyz[1])
        else:
            current_xy = self._printed_xy
        pose = self.command_pose(target_xyz, current_xy)
        self._printed_xy = (pose[0], pose[1])
        key = tuple(round(v, 4) for v in pose)
        if key == self._last_printed:
            return "printed — pose unchanged"
        self._last_printed = key
        shown = [round(v, 4) for v in pose]
        print(
            f"servoL({shown}, {SERVO_SPEED}, {SERVO_ACCEL}, 0.1, 0.1, 200)"
        )
        return "printed servoL"

    def track(self, target_xyz: list[float]) -> str:
        """Move toward target_xyz. Z is forced onto the hover plane."""
        if self.receive.isProtectiveStopped() or self.receive.isEmergencyStopped():
            return "robot stopped — not moving"

        current = list(self.receive.getActualTCPPose())
        pose = self.command_pose(target_xyz, (current[0], current[1]))

        if not self.control.isPoseWithinSafetyLimits(pose):
            return "outside UR safety limits — held"
        if not self.control.getInverseKinematicsHasSolution(pose):
            return "no IK solution — held"

        # time 0.1 s, lookahead 0.1 s, gain 200: smooth and slow
        self.control.servoL(pose, SERVO_SPEED, SERVO_ACCEL, 0.1, 0.1, 200)
        return "tracking"
