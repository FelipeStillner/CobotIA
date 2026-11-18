"""RTDE link to the UR5. Sends only poses that stay on the hover plane.

Motion happens in two stages:

1. approach(): one slow, straight-line moveL from wherever the TCP is onto
   the hover plane, with the fixed tool orientation. The caller confirms it.
2. track(): servoL toward the object. Tracking refuses to run unless the TCP
   is already on the hover plane with the fixed orientation, so servoL only
   ever makes small XY moves. It never makes large Z or rotation moves, which
   the controller would carry out as fast joint-space jumps.
"""

from __future__ import annotations

import time

from config import (
    APPROACH_ACCEL,
    APPROACH_SPEED,
    HOVER_Z_TOL_M,
    MAX_SERVO_TIME,
    MAX_SPEED_M_S,
    MIN_SERVO_TIME,
    ORIENTATION_TOL_RAD,
    ROBOT_IP,
    SERVO_GAIN,
    SERVO_LOOKAHEAD,
    SERVO_TIME,
    TABLE_Z_ROBOT,
)
from safety import (
    check_hover_plane,
    enforce_hover,
    limit_step,
    orientation_error_rad,
    tcp_orientation,
)

# servoL ignores speed and acceleration; step / servo time sets the speed.


class Robot:
    def __init__(self, ip: str = ROBOT_IP):
        self.ip = ip
        self.control = None
        self.receive = None
        self.ready = False   # True once the TCP is on the hover plane
        self._last_servo: float | None = None   # time.monotonic() of the last servoL
        self.last_actual: list[float] | None = None   # TCP pose read by the last track()
        self.last_sent: list[float] | None = None     # pose sent by the last servoL

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
            for stop in (lambda: self.control.servoStop(), lambda: self.control.stopL(0.5)):
                try:
                    stop()
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
        self.ready = False

    def hold(self) -> None:
        """Stop a servo stream and leave the arm where it is."""
        self._last_servo = None
        if self.control is None:
            return
        try:
            self.control.servoStop()
        except Exception:
            pass

    def _stopped(self) -> bool:
        return self.receive.isProtectiveStopped() or self.receive.isEmergencyStopped()

    def approach_pose(self) -> list[float]:
        """Current XY, hover Z, fixed orientation: where approach() goes."""
        current = list(self.receive.getActualTCPPose())
        return enforce_hover([current[0], current[1], check_hover_plane(), *tcp_orientation()])

    def on_hover_plane(self) -> tuple[bool, str]:
        """True when the TCP is on the hover plane with the fixed orientation."""
        current = list(self.receive.getActualTCPPose())
        dz = current[2] - check_hover_plane()
        rot = orientation_error_rad(current[3:6], tcp_orientation())
        if abs(dz) > HOVER_Z_TOL_M:
            return False, f"TCP is {dz * 1000:+.1f} mm off the hover plane"
        if rot > ORIENTATION_TOL_RAD:
            return False, f"tool is {rot * 57.2958:.1f} deg off the calibrated orientation"
        return True, "on hover plane"

    def describe_approach(self) -> list[str]:
        """The approach move in words, for the operator to confirm."""
        current = list(self.receive.getActualTCPPose())
        pose = self.approach_pose()
        dz = pose[2] - current[2]
        rot = orientation_error_rad(current[3:6], pose[3:6])
        seconds = abs(dz) / APPROACH_SPEED
        lines = [
            f"  from {[round(v, 4) for v in current]}",
            f"  to   {[round(v, 4) for v in pose]}",
        ]
        if abs(dz) > 0.0005:
            direction = "DOWN" if dz < 0 else "UP"
            lines.append(
                f"  moves straight {direction} {abs(dz) * 1000:.0f} mm "
                f"at {APPROACH_SPEED * 100:.0f} cm/s (about {seconds:.0f} s), XY unchanged"
            )
        lines.append(
            f"  ends at Z {pose[2]:.3f} m, {pose[2] - TABLE_Z_ROBOT:.3f} m above the table"
        )
        if rot > ORIENTATION_TOL_RAD:
            lines.append(f"  turns the tool {rot * 57.2958:.0f} deg to the calibrated orientation")
        return lines

    def approach(self) -> None:
        """Slow straight-line moveL onto the hover plane. Raises if refused."""
        if self._stopped():
            raise RuntimeError("robot is protective- or emergency-stopped")
        pose = self.approach_pose()
        if not self.control.isPoseWithinSafetyLimits(pose):
            raise RuntimeError("approach pose is outside the UR safety limits")
        if not self.control.getInverseKinematicsHasSolution(pose):
            raise RuntimeError("approach pose has no IK solution")
        if not self.control.moveL(pose, APPROACH_SPEED, APPROACH_ACCEL):
            where = [round(v, 4) for v in self.receive.getActualTCPPose()]
            if self.receive.isEmergencyStopped():
                cause = "emergency stop"
            elif self.receive.isProtectiveStopped():
                cause = "protective stop"
            else:
                cause = "moveL did not finish"
            raise RuntimeError(f"approach interrupted ({cause}); TCP stopped at {where}")
        ok, why = self.on_hover_plane()
        if not ok:
            raise RuntimeError(f"after approach: {why}")
        self.ready = True

    def servo_dt(self) -> float:
        """Real time since the last servoL, clamped to [MIN, MAX]_SERVO_TIME."""
        if self._last_servo is None:
            return SERVO_TIME
        dt = time.monotonic() - self._last_servo
        return min(max(dt, MIN_SERVO_TIME), MAX_SERVO_TIME)

    def command_pose(self, target_xyz: list[float], current_xyz: list[float], dt: float) -> list[float]:
        """Next servo pose: on the hover plane, at most MAX_SPEED_M_S * dt from current."""
        step = MAX_SPEED_M_S * dt
        x, y, z = limit_step(current_xyz, (target_xyz[0], target_xyz[1], check_hover_plane()), step)
        return enforce_hover([x, y, z, *tcp_orientation()])

    def track(self, target_xyz: list[float]) -> str:
        """Step toward target_xyz at most MAX_SPEED_M_S. Holds if anything is off."""
        if not self.ready:
            self.hold()
            return "not on hover plane — run approach first"
        if self._stopped():
            self.ready = False
            return "robot stopped — not moving"

        ok, why = self.on_hover_plane()
        if not ok:
            # Never let servoL correct a Z or orientation error: hold instead.
            self.ready = False
            self.hold()
            return f"{why} — held"

        current = list(self.receive.getActualTCPPose())
        self.last_actual = current
        self.last_sent = None
        dt = self.servo_dt()
        pose = self.command_pose(target_xyz, current[:3], dt)

        if not self.control.isPoseWithinSafetyLimits(pose):
            self.hold()
            return "outside UR safety limits — held"
        if not self.control.getInverseKinematicsHasSolution(pose):
            self.hold()
            return "no IK solution — held"

        # speed and acceleration (0, 0) are ignored by servoL; servo time = dt
        self.control.servoL(pose, 0.0, 0.0, dt, SERVO_LOOKAHEAD, SERVO_GAIN)
        self._last_servo = time.monotonic()
        self.last_sent = pose
        return f"tracking ({1.0 / dt:.0f} Hz)"
