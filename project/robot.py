"""RTDE link to the UR5. Sends only poses at or above the floor.

Motion happens in two stages:

1. approach(): one slow, straight-line moveL from wherever the TCP is onto
   the hover height, with the fixed tool orientation. The caller confirms it.
2. track(): servoL toward the object. Tracking refuses to run unless the tool
   orientation matches and Z is at least the floor, so servoL only ever makes
   small steps. It never makes a large rotation, which the controller would
   carry out as a fast joint-space jump.
"""

from __future__ import annotations

import time

from config import (
    APPROACH_ACCEL,
    HOVER_Z_TOL_M,
    MAX_SERVO_TIME,
    MIN_SERVO_TIME,
    ORIENTATION_TOL_RAD,
    ROBOT_IP,
    SERVO_GAIN,
    SERVO_LOOKAHEAD,
    SERVO_TIME,
    TABLE_Z_ROBOT,
)
from safety import (
    approach_speed_m_s,
    check_hover_plane,
    enforce_hover,
    limit_step,
    min_tcp_z,
    orientation_error_rad,
    tcp_orientation,
    tracking_speed,
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
        self.ensure_control()

    def disconnect(self) -> None:
        if self.control is not None and self.control.isProgramRunning():
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

    def _in_remote_control(self) -> bool | None:
        """True, False, or None when the dashboard cannot answer."""
        try:
            from dashboard_client import DashboardClient

            dash = DashboardClient(self.ip)
            dash.connect(2000)
            try:
                return bool(dash.isInRemoteControl())
            finally:
                dash.disconnect()
        except Exception:
            return None

    def ensure_control(self) -> None:
        """Start the UR control script. It only runs in Remote Control.

        Without that script, the controller answers the safety query with
        false, which is not the same thing as a pose outside the safety planes.
        """
        if self.control is None or not self.control.isConnected():
            raise RuntimeError("Not connected to the robot")
        if self.control.isProgramRunning():
            return
        if self._in_remote_control() is False:
            raise RuntimeError(
                "The robot is in Local control, so it will not run the approach. "
                "In PolyScope, open the mode menu at the top and choose Remote Control."
            )
        if not self.control.reuploadScript():
            raise RuntimeError(
                "The UR control script did not start. Set the robot to Remote Control and try again."
            )
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if self.control.isProgramRunning():
                return
            time.sleep(0.05)
        raise RuntimeError(
            "The UR control script did not start. Set the robot to Remote Control and try again."
        )

    def hold(self) -> None:
        """Stop a servo stream and leave the arm where it is."""
        self._last_servo = None
        if self.control is None:
            return
        try:
            if not self.control.isProgramRunning():
                return
            self.control.servoStop()
        except Exception:
            pass

    def _stopped(self) -> bool:
        return self.receive.isProtectiveStopped() or self.receive.isEmergencyStopped()

    def approach_z(self) -> float:
        """Hover height, raised to the floor when the floor is higher."""
        return max(check_hover_plane(), min_tcp_z())

    def approach_pose(self) -> list[float]:
        """Current XY, approach Z, fixed orientation: where approach() goes."""
        current = list(self.receive.getActualTCPPose())
        return enforce_hover([current[0], current[1], self.approach_z(), *tcp_orientation()])

    def on_hover_plane(self) -> tuple[bool, str]:
        """True when the TCP is on the approach height with the fixed orientation."""
        current = list(self.receive.getActualTCPPose())
        dz = current[2] - self.approach_z()
        rot = orientation_error_rad(current[3:6], tcp_orientation())
        if abs(dz) > HOVER_Z_TOL_M:
            return False, f"TCP is {dz * 1000:+.1f} mm off the hover plane"
        if rot > ORIENTATION_TOL_RAD:
            return False, f"tool is {rot * 57.2958:.1f} deg off the calibrated orientation"
        return True, "on hover plane"

    def on_trackable(self) -> tuple[bool, str]:
        """True when tracking may servo: tool angle matches and Z is at least 200 mm."""
        current = list(self.receive.getActualTCPPose())
        floor = min_tcp_z()
        if current[2] < floor - HOVER_Z_TOL_M:
            return False, f"TCP Z={current[2]:.3f} m is below {floor:.3f} m"
        rot = orientation_error_rad(current[3:6], tcp_orientation())
        if rot > ORIENTATION_TOL_RAD:
            return False, f"tool is {rot * 57.2958:.1f} deg off the calibrated orientation"
        return True, "tracking allowed"

    def describe_approach(self) -> list[str]:
        """The approach move in words, for the operator to confirm."""
        current = list(self.receive.getActualTCPPose())
        pose = self.approach_pose()
        dz = pose[2] - current[2]
        rot = orientation_error_rad(current[3:6], pose[3:6])
        seconds = abs(dz) / approach_speed_m_s()
        lines = [
            f"  from {[round(v, 4) for v in current]}",
            f"  to   {[round(v, 4) for v in pose]}",
        ]
        if abs(dz) > 0.0005:
            direction = "DOWN" if dz < 0 else "UP"
            lines.append(
                f"  moves straight {direction} {abs(dz) * 1000:.0f} mm "
                f"at {approach_speed_m_s() * 100:.0f} cm/s (about {seconds:.0f} s), XY unchanged"
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
        self.ensure_control()
        pose = self.approach_pose()
        if not self.control.isPoseWithinSafetyLimits(pose):
            where = [round(v, 4) for v in pose]
            raise RuntimeError(
                f"approach pose {where} is outside the UR safety limits"
            )
        if not self.control.getInverseKinematicsHasSolution(pose):
            raise RuntimeError("approach pose has no IK solution")
        if not self.control.moveL(pose, approach_speed_m_s(), APPROACH_ACCEL):
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
        """Next servo pose: toward the target, at the clamped speed, Z at least the floor."""
        step = tracking_speed() * dt
        z_goal = max(float(target_xyz[2]), min_tcp_z())
        x, y, z = limit_step(current_xyz, (target_xyz[0], target_xyz[1], z_goal), step)
        return enforce_hover([x, y, z, *tcp_orientation()])

    def track(self, target_xyz: list[float]) -> str:
        """Step toward target_xyz at most MAX_SPEED_M_S. Holds if anything is off."""
        if not self.ready:
            self.hold()
            return "not on hover plane — run approach first"
        if self._stopped():
            self.ready = False
            return "robot stopped — not moving"
        try:
            self.ensure_control()
        except RuntimeError as exc:
            self.ready = False
            return str(exc)

        ok, why = self.on_trackable()
        if not ok:
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
