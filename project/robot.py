"""RTDE link to the UR5. Sends only poses that stay on the hover plane."""

from __future__ import annotations

from config import MAX_STEP_M, ROBOT_IP, SERVO_ACCEL, SERVO_SPEED, TCP_ORIENTATION
from safety import enforce_hover, limit_step


class Robot:
    def __init__(self, ip: str = ROBOT_IP):
        self.ip = ip
        self.control = None
        self.receive = None

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

    def track(self, target_xyz: list[float]) -> str:
        """Move toward target_xyz. Z is forced onto the hover plane."""
        if self.receive.isProtectiveStopped() or self.receive.isEmergencyStopped():
            return "robot stopped — not moving"

        current = list(self.receive.getActualTCPPose())
        x, y = limit_step(
            (current[0], current[1]),
            (target_xyz[0], target_xyz[1]),
            MAX_STEP_M,
        )
        pose = enforce_hover([x, y, target_xyz[2], *TCP_ORIENTATION])

        if not self.control.isPoseWithinSafetyLimits(pose):
            return "outside UR safety limits — held"
        if not self.control.getInverseKinematicsHasSolution(pose):
            return "no IK solution — held"

        # time 0.1 s, lookahead 0.1 s, gain 200: smooth and slow
        self.control.servoL(pose, SERVO_SPEED, SERVO_ACCEL, 0.1, 0.1, 200)
        return "tracking"
