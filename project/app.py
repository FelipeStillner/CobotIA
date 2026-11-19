"""One window for calibration and following.

The picture is on the left. The controls are on the right.
Nothing moves until Confirm approach is pressed.

    sudo ./venv/bin/python project/app.py

URSim is selected when the window opens. Choose Real arm to use config.ROBOT_IP.
"""

from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

if sys.platform.startswith("linux"):
    os.environ["QT_QPA_PLATFORM"] = "xcb"
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibrate import (
    advance_calibration,
    click_at,
    reset_calibration,
    save_from_clicks,
    state as cal_state,
    undo,
)
from config import ROBOT_IP
from detect_object import ClosestObjectPicker, detect_objects
from realsense_overlay import RealSense, RsError
from robot import Robot
from safety import (
    check_hover_plane,
    has_robot_map,
    in_camera_workspace,
    set_clearance,
    set_min_tcp_z,
    set_tracking_speed,
    target_over_object,
    tcp_orientation,
)
from screen import draw_calibrate_view, draw_follow_view

robot = Robot()

STAGE = "#141414"
BG = "#f6f4f1"
INK = "#1c1c1c"
MUTED = "#6f6a64"
LINE = "#e4e0da"
ACCENT = "#1f6b4a"


def _family(root: tk.Tk, names: tuple[str, ...], fallback: str) -> str:
    have = set(tkfont.families(root))
    for name in names:
        if name in have:
            return name
    return fallback


class App:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Cobot")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.alive = True
        self.lock = threading.Lock()
        self.latest = None
        self.cam = None
        self.cam_error = ""
        self.photo = None
        self.phase = "idle"  # idle, calibrate, confirm, approaching, tracking
        self.holding = False
        self.picker = ClosestObjectPicker()
        self.jobs: list[tuple] = []
        self.job_lock = threading.Lock()
        self._build_ui()
        self.root.bind("<Key>", self.on_key)
        threading.Thread(target=self.camera_loop, daemon=True).start()
        self.root.after(33, self.tick)

    def _build_ui(self) -> None:
        self.font = _family(self.root, ("Helvetica Neue", "Segoe UI", "Ubuntu", "DejaVu Sans"), "Helvetica")
        self.native = sys.platform == "darwin"
        self.panel = self.root.cget("bg") if self.native else BG
        self.quiet = "systemSecondaryLabelColor" if self.native else MUTED
        self.ink = "systemLabelColor" if self.native else INK
        self.root.configure(bg=self.panel)
        self.root.minsize(1080, 560)
        style = ttk.Style(self.root)
        if not self.native:
            style.theme_use("clam")
            style.configure("TButton", padding=(10, 7), background="#ffffff", foreground=INK,
                            bordercolor=LINE, lightcolor="#ffffff", darkcolor="#ffffff",
                            borderwidth=1, focusthickness=0, focuscolor=BG, font=(self.font, 13))
            style.map("TButton",
                      background=[("pressed", "#efece7"), ("active", "#f7f5f2"), ("disabled", "#f3f1ee")],
                      foreground=[("disabled", "#b0aaa4")])
            style.configure("Primary.TButton", background=ACCENT, foreground="#ffffff",
                            bordercolor=ACCENT, lightcolor=ACCENT, darkcolor=ACCENT,
                            font=(self.font, 13))
            style.map("Primary.TButton",
                      background=[("pressed", "#174e36"), ("active", "#1a5c40"), ("disabled", "#e7e4df")],
                      foreground=[("disabled", "#b0aaa4")],
                      bordercolor=[("disabled", "#e7e4df"), ("active", "#1a5c40")])
            style.configure("TRadiobutton", background=BG, foreground=INK, focuscolor=BG, font=(self.font, 13))
            style.map("TRadiobutton", background=[("active", BG)])
            style.configure("Horizontal.TScale", background=INK, troughcolor="#e5e1db",
                            borderwidth=0, sliderthickness=14)
            style.map("Horizontal.TScale", background=[("active", ACCENT)])
        else:
            style.configure("TButton", font=(self.font, 13), padding=(8, 4))
            style.configure("TRadiobutton", font=(self.font, 13))

        body = tk.Frame(self.root, bg=self.panel)
        body.pack(fill=tk.BOTH, expand=True)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, minsize=300)
        body.grid_rowconfigure(0, weight=1)

        stage = tk.Frame(body, bg=STAGE, width=1100, height=640)
        stage.grid(row=0, column=0, sticky="nsew")
        stage.grid_propagate(False)
        self.stage = stage

        side = tk.Frame(body, bg=self.panel)
        side.grid(row=0, column=1, sticky="nsew")
        content = tk.Frame(side, bg=self.panel)
        content.pack(fill=tk.X, padx=24, pady=24)
        self.image_label = tk.Label(
            stage, bg=STAGE, fg="#9a9690", text="Waiting for the camera",
            font=(self.font, 14), highlightthickness=0, bd=0,
        )
        self.image_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.image_label.bind("<Button-1>", self.on_click)

        title = tk.Frame(content, bg=self.panel)
        title.pack(fill=tk.X)
        tk.Label(title, text="Cobot", bg=self.panel, fg=self.ink, font=(self.font, 22)).pack(side=tk.LEFT)
        self.phase_text = tk.StringVar(value="Idle")
        self.phase_label = tk.Label(
            title, textvariable=self.phase_text, bg=self.panel, fg=self.quiet, font=(self.font, 12),
        )
        self.phase_label.pack(side=tk.RIGHT, pady=(8, 0))

        self.robot_choice = tk.StringVar(value="sim")
        ttk.Radiobutton(
            content, text=f"URSim  ·  127.0.0.1", variable=self.robot_choice,
            value="sim", command=self.on_robot_choice,
        ).pack(anchor=tk.W, pady=(16, 0))
        ttk.Radiobutton(
            content, text=f"Real arm  ·  {ROBOT_IP}", variable=self.robot_choice,
            value="real", command=self.on_robot_choice,
        ).pack(anchor=tk.W, pady=(4, 0))

        self.speed_value = self._caption(content, "Speed", "0.10 m/s", top=18)
        self.speed = ttk.Scale(content, from_=0.01, to=0.10, command=self.on_speed)
        self.speed.pack(fill=tk.X, pady=(2, 0))
        self.clearance_value = self._caption(content, "Above the object", "20 cm")
        self.clearance = ttk.Scale(content, from_=10, to=50, command=self.on_clearance)
        self.clearance.pack(fill=tk.X, pady=(2, 0))
        self.floor_value = self._caption(content, "Lowest Z", "20 cm")
        self.floor = ttk.Scale(content, from_=10, to=50, command=self.on_floor)
        self.floor.pack(fill=tk.X, pady=(2, 0))
        self.speed.set(0.10)
        self.clearance.set(20)
        self.floor.set(20)

        self._wide(content, "Calibrate", self.start_calibrate, top=18)
        row = tk.Frame(content, bg=self.panel)
        row.pack(fill=tk.X, pady=(6, 0))
        for col in (0, 2, 4):
            row.columnconfigure(col, weight=1, uniform="steps")
        row.columnconfigure(1, minsize=6)
        row.columnconfigure(3, minsize=6)
        for col, (text, command) in enumerate((
            ("Next", advance_calibration),
            ("Undo", undo),
            ("Save", self.save_calibration),
        )):
            ttk.Button(row, text=text, command=command).grid(row=0, column=col * 2, sticky="ew")
        self._wide(content, "Start follow", self.start_follow)
        confirm_style = "TButton" if self.native else "Primary.TButton"
        self.confirm_button = self._wide(content, "Confirm approach", self.confirm_approach, confirm_style)
        self.confirm_button.configure(state=tk.DISABLED)
        self._wide(content, "Stop", self.stop_motion)

        ttk.Separator(content, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(18, 12))
        self.approach_box = tk.Label(
            content,
            text="The arm stays still until you confirm the approach.",
            bg=self.panel, fg=self.ink, justify=tk.LEFT, anchor=tk.W, wraplength=240,
            font=(self.font, 12),
        )
        self.approach_box.pack(fill=tk.X)

        self.status = tk.StringVar(value="Opening the camera")
        tk.Label(
            content, textvariable=self.status, bg=self.panel, fg=self.quiet, justify=tk.LEFT, anchor=tk.W,
            wraplength=240, font=(self.font, 12),
        ).pack(anchor=tk.W, pady=(10, 0))

    def _caption(self, parent, text: str, value: str, top: int = 10) -> tk.Label:
        row = tk.Frame(parent, bg=self.panel)
        row.pack(fill=tk.X, pady=(top, 0))
        tk.Label(row, text=text, bg=self.panel, fg=self.quiet, font=(self.font, 12)).pack(side=tk.LEFT)
        label = tk.Label(row, text=value, bg=self.panel, fg=self.ink, font=(self.font, 12))
        label.pack(side=tk.RIGHT)
        return label

    def _wide(self, parent, text: str, command, style_name: str = "TButton", top: int = 6) -> ttk.Button:
        button = ttk.Button(parent, text=text, command=command, style=style_name)
        button.pack(fill=tk.X, pady=(top, 0))
        return button

    def ip(self) -> str:
        return ROBOT_IP if self.robot_choice.get() == "real" else "127.0.0.1"

    def post(self, kind: str, payload) -> None:
        with self.job_lock:
            self.jobs.append((kind, payload))

    def take_jobs(self) -> list[tuple]:
        with self.job_lock:
            jobs = self.jobs
            self.jobs = []
        return jobs

    def set_approach_text(self, text: str) -> None:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        words = [line for line in lines if not line.startswith(("from ", "to "))]
        shown = "\n".join(words or lines)
        self.approach_box.configure(
            text=shown or "The arm stays still until you confirm the approach."
        )

    def _snap(self, scale: ttk.Scale, value: str, step: float) -> float:
        snapped = round(round(float(value) / step) * step, 4)
        snapped = min(float(scale.cget("to")), max(float(scale.cget("from")), snapped))
        if abs(float(scale.get()) - snapped) > step / 4:
            scale.set(snapped)
        return snapped

    def on_speed(self, value: str) -> None:
        snapped = self._snap(self.speed, value, 0.01)
        set_tracking_speed(snapped)
        self.speed_value.configure(text=f"{snapped:.2f} m/s")

    def on_clearance(self, value: str) -> None:
        snapped = self._snap(self.clearance, value, 1)
        set_clearance(snapped / 100.0)
        self.clearance_value.configure(text=f"{snapped:.0f} cm")

    def on_floor(self, value: str) -> None:
        snapped = self._snap(self.floor, value, 1)
        set_min_tcp_z(snapped / 100.0)
        self.floor_value.configure(text=f"{snapped:.0f} cm")

    def on_robot_choice(self) -> None:
        self.stop_motion()
        self.close_calibration_link()
        robot.disconnect()
        where = "the real arm" if self.robot_choice.get() == "real" else "URSim"
        self.status.set(f"Selected {where}. Nothing is connected yet.")

    def close_calibration_link(self) -> None:
        receive = cal_state.get("receive")
        if receive is None:
            return
        try:
            receive.disconnect()
        except Exception:
            pass
        cal_state["receive"] = None

    def camera_loop(self) -> None:
        try:
            self.cam = RealSense()
            self.cam.start()
        except RsError as exc:
            self.post("camera", str(exc))
            return
        self.post("camera", "")
        while self.alive:
            try:
                color, depth, _center = self.cam.wait_aligned()
            except Exception as exc:
                self.post("camera", str(exc))
                continue
            with self.lock:
                self.latest = (color, depth, self.cam.depth_scale_m, self.cam.intrinsics)

    def latest_frame(self):
        with self.lock:
            return self.latest

    def start_calibrate(self) -> None:
        if self.phase == "approaching":
            self.status.set("Wait for the approach to finish, or press Stop.")
            return
        self.stop_motion()
        robot.disconnect()
        self.close_calibration_link()
        from rtde_receive import RTDEReceiveInterface

        try:
            receive = RTDEReceiveInterface(self.ip())
            receive.getActualTCPPose()
        except Exception as exc:
            self.status.set(f"Robot connection failed: {exc}")
            return
        cal_state["receive"] = receive
        reset_calibration()
        self.phase = "calibrate"
        self.confirm_button.configure(state=tk.DISABLED)
        self.set_approach_text("")
        self.status.set("Calibration. Hold the tool still over a corner, then click it.")

    def save_calibration(self) -> None:
        if save_from_clicks():
            self.status.set("Saved calibration.json")
        else:
            self.status.set(cal_state.get("message") or "Not saved")

    def start_follow(self) -> None:
        if self.phase == "approaching":
            return
        self.stop_motion()
        self.close_calibration_link()
        if self.phase == "calibrate":
            self.phase = "idle"
        if not has_robot_map():
            self.status.set("No calibration yet. Calibrate the four corners and the object first.")
            return
        try:
            check_hover_plane()
            tcp_orientation()
        except RuntimeError as exc:
            self.status.set(str(exc))
            return
        robot.ip = self.ip()
        try:
            if robot.control is None:
                robot.connect()
        except Exception as exc:
            self.status.set(f"Robot connection failed: {exc}")
            return
        try:
            lines = robot.describe_approach()
        except Exception as exc:
            self.status.set(str(exc))
            return
        self.set_approach_text("\n".join(lines))
        self.phase = "confirm"
        self.confirm_button.configure(state=tk.NORMAL)
        self.picker.reset()
        self.status.set("Nothing is moving. Read the approach, then confirm it.")

    def confirm_approach(self) -> None:
        if self.phase != "confirm":
            return
        self.phase = "approaching"
        self.confirm_button.configure(state=tk.DISABLED)
        self.status.set("Approaching the hover height.")

        def work() -> None:
            try:
                robot.approach()
            except Exception as exc:
                self.post("approach", exc)
                return
            self.post("approach", None)

        threading.Thread(target=work, daemon=True).start()

    def stop_motion(self) -> None:
        if self.phase == "approaching" and robot.control is not None:
            try:
                robot.control.stopL(0.5)
            except Exception:
                pass
        if robot.control is not None:
            robot.hold()
            robot.ready = False
        if self.phase != "calibrate":
            self.phase = "idle"
        self.confirm_button.configure(state=tk.DISABLED)
        self.status.set("Stopped. The camera stays open.")

    def on_click(self, event) -> None:
        if self.phase != "calibrate" or self.photo is None:
            return
        frame = self.latest_frame()
        if frame is None:
            return
        color = frame[0]
        view = self.photo
        # draw_*_view stacks color | depth. Map the click back onto that image.
        full_w = color.shape[1] * 2
        full_h = color.shape[0]
        src_x = int(event.x * full_w / max(view.width(), 1))
        src_y = int(event.y * full_h / max(view.height(), 1))
        click_at(src_x, src_y)
        self.status.set(cal_state.get("message") or "")

    def on_key(self, event) -> None:
        key = event.keysym.lower()
        if self.phase == "calibrate":
            if key == "n":
                advance_calibration()
            elif key == "u":
                undo()
            elif key == "s":
                self.save_calibration()
        if key == "q":
            self.close()

    def show(self, bgr) -> None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        box_w = self.stage.winfo_width()
        box_h = self.stage.winfo_height()
        if box_w <= 1:
            box_w = self.stage.winfo_reqwidth()
        if box_h <= 1:
            box_h = self.stage.winfo_reqheight()
        box_w = max(box_w - 24, 1)
        box_h = max(box_h - 24, 1)
        scale = min(box_w / width, box_h / height)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        if (new_w, new_h) != (width, height):
            rgb = cv2.resize(
                rgb, (new_w, new_h),
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
            )
            height, width = new_h, new_w
        header = f"P6 {width} {height} 255\n".encode()
        photo = tk.PhotoImage(data=header + rgb.tobytes(), format="PPM")
        self.photo = photo
        self.image_label.configure(image=photo, text="")

    def draw(self) -> None:
        frame = self.latest_frame()
        if frame is None:
            return
        color, depth, scale, intrinsics = frame
        if self.phase == "calibrate":
            cal_state["color"] = color
            cal_state["depth"] = depth
            cal_state["scale"] = scale if scale else 0.001
            cal_state["intrinsics"] = intrinsics
            image = draw_calibrate_view(
                color, depth, cal_state["scale"], cal_state["corners"],
                cal_state["phase"], cal_state["message"],
            )
            self.show(image)
            self.status.set(cal_state.get("message") or "")
            return

        det = None
        detections = []
        target = None
        status = self.status.get()
        if intrinsics is not None and scale:
            detections = detect_objects(depth, scale, intrinsics, color)
            det = self.picker.pick(detections)
            if det is not None and in_camera_workspace(det.cam_x, det.cam_z):
                target = target_over_object(det.cam_x, det.cam_z, det.height_m)
            if self.phase == "tracking":
                if target is not None:
                    self.holding = False
                    try:
                        status = robot.track(target)
                    except Exception as exc:
                        status = f"move rejected: {exc}"
                        robot.hold()
                        self.holding = True
                else:
                    if not self.holding:
                        robot.hold()
                        self.holding = True
                    status = "no object - not moving" if det is None else "outside pad — holding"
        image = draw_follow_view(
            color, depth, scale, intrinsics, det, target, status, detections,
        )
        self.show(image)
        if self.phase == "tracking":
            self.status.set(status)

    def tick(self) -> None:
        try:
            for kind, payload in self.take_jobs():
                if kind == "camera" and payload:
                    self.cam_error = payload
                    self.status.set(
                        f"Camera failed: {payload}. On macOS run with sudo ./venv/bin/python project/app.py"
                    )
                elif kind == "approach":
                    if self.phase != "approaching":
                        continue
                    if payload is None:
                        self.phase = "tracking"
                        self.holding = False
                        self.picker.reset()
                        self.status.set("Tracking.")
                    else:
                        self.phase = "confirm"
                        self.confirm_button.configure(state=tk.NORMAL)
                        robot.ready = False
                        self.status.set(f"Approach stopped: {payload}")
            self.draw()
            names = {
                "idle": "Idle",
                "calibrate": "Calibrating",
                "confirm": "Confirm",
                "approaching": "Moving",
                "tracking": "Following",
            }
            self.phase_text.set(names.get(self.phase, self.phase))
            active = self.phase in ("approaching", "tracking", "confirm")
            self.phase_label.configure(fg=self.ink if active else self.quiet)
            self.image_label.configure(cursor="crosshair" if self.phase == "calibrate" else "")
        except Exception as exc:
            self.status.set(str(exc))
        if self.alive:
            self.root.after(33, self.tick)

    def close(self) -> None:
        self.alive = False
        self.stop_motion()
        self.close_calibration_link()
        robot.disconnect()
        if self.cam is not None:
            try:
                self.cam.stop()
            except Exception:
                pass
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    set_tracking_speed(0.10)
    set_clearance(0.20)
    set_min_tcp_z(0.20)
    App().run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
