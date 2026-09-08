from __future__ import annotations

import json
import math
from pathlib import Path
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog

from ..attention import Attention
from ..mapping import map_sample
from ..model import SonicScene
from ..session import Transport
from ..sources.synthetic import SCENARIOS, make_sample

BG = "#101823"
PANEL = "#182434"
TEXT = "#e6edf5"
MUTED = "#99acbe"
POS = "#63d9be"
NEG = "#eaa2d7"


def audio_status(diagnostic):
    """Keep operational status legible; full implementation diagnostics export separately."""
    state = diagnostic.get("state", "unknown")
    if diagnostic.get("last_error"):
        return f"音频 {state} · {diagnostic['last_error']} · 可点击重连"
    hrtf = diagnostic.get("hrtf_status", "unknown")
    spatial = "HRTF 已启用" if hrtf in {"enabled", "required", "headphones_detected"} else f"HRTF 未启用 ({hrtf}) · 当前不作空间验收"
    quality = diagnostic.get("data_state", "")
    return f"{diagnostic.get('device', '音频设备')} · {spatial} · {state} / {quality}"


class SonicApp:
    def __init__(self, window: tk.Tk, *, no_audio=False, dll_path=None, engine=None):
        self.window = window
        self.no_audio = no_audio
        self.dll_path = dll_path
        self.engine = engine
        self.audio_ready = False
        self.opening = False
        self._mute_requested = False
        self.closing = False
        self._open_result = None
        self._tick_id = None
        self._last_tick = time.monotonic()
        self._last_diag = 0.0
        self.transport = Transport()
        self.sequence = 0
        self.center = (0.0, 0.0)
        self.sample = None
        self.scene = SonicScene()
        self._labels = {v: k for k, v in SCENARIOS.items()}
        self.scenario = tk.StringVar(value=next(iter(SCENARIOS.values())))
        self.mode = tk.StringVar(value="both")
        self.radius = tk.DoubleVar(value=0.55)
        self.background = tk.DoubleVar(value=0.15)
        self.master = tk.DoubleVar(value=0.15)
        self.static = tk.BooleanVar(value=True)
        self.hrtf = tk.BooleanVar(value=True)
        self.device = tk.StringVar(value="系统默认设备")
        self.status = tk.StringVar(value="无声预览" if no_audio else "尚未开启音频 · 点击“开始试听”")
        self.time_label = tk.StringVar()
        self.summary = tk.StringVar()
        self._build()
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.bind("<space>", lambda event: self.toggle_play())
        self._tick()

    def _build(self):
        w = self.window
        # Keep the full control panel visible even with Windows desktop scaling.
        w.tk.call("tk", "scaling", 96 / 72)
        w.title("MuMax-Sonic · P1 空间听觉实验室")
        w.geometry("1120x940")
        w.minsize(1050, 900)
        w.configure(bg=BG)
        style = ttk.Style(w)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=TEXT, font=("Microsoft YaHei UI", 10))
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 21, "bold"))
        style.configure("TButton", padding=(12, 7), font=("Microsoft YaHei UI", 10))
        style.configure("TCheckbutton", background=BG, foreground=TEXT, font=("Microsoft YaHei UI", 10))
        style.map("TCheckbutton", background=[("active", PANEL)], foreground=[("active", TEXT)])
        style.configure("TRadiobutton", background=BG, foreground=TEXT)
        style.map("TRadiobutton", background=[("active", PANEL)], foreground=[("active", TEXT)])
        style.configure("Horizontal.TScale", background=BG)
        style.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=TEXT, rowheight=27)
        style.configure("Treeview.Heading", font=("Microsoft YaHei UI", 9, "bold"))
        shell = ttk.Frame(w, padding=22)
        shell.pack(fill="both", expand=True)
        ttk.Label(shell, text="MuMax-Sonic", style="Title.TLabel").pack(anchor="w")
        ttk.Label(shell, text="P1 · 合成声源演示 / Synthetic demo · 尚未计算物理拓扑荷", style="Muted.TLabel").pack(anchor="w", pady=(2, 14))
        toolbar = ttk.Frame(shell)
        toolbar.pack(fill="x", pady=(0, 12))
        ttk.Label(toolbar, text="场景").pack(side="left", padx=(0, 8))
        combo = ttk.Combobox(toolbar, textvariable=self.scenario, values=list(self._labels), state="readonly", width=24)
        combo.pack(side="left")
        combo.bind("<<ComboboxSelected>>", lambda e: self.reset())
        self.play_button = ttk.Button(toolbar, text="播放", command=self.toggle_play)
        self.play_button.pack(side="left", padx=8)
        ttk.Button(toolbar, text="重置", command=self.reset).pack(side="left")
        ttk.Checkbutton(toolbar, text="暂停时探听静态", variable=self.static).pack(side="left", padx=14)
        ttk.Label(toolbar, textvariable=self.time_label, style="Muted.TLabel").pack(side="right")
        middle = ttk.Frame(shell)
        middle.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(middle, bg=PANEL, highlightthickness=0, width=690, height=370)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.move_roi)
        self.canvas.bind("<B1-Motion>", self.move_roi)
        controls = ttk.Frame(middle, padding=(20, 4, 0, 0), width=280)
        controls.pack(side="right", fill="y")
        ttk.Label(controls, text="关注区域 · 拖动画面移动").pack(anchor="w", pady=(0, 7))
        for label, var, lo, hi in (("半径", self.radius, 0.1, 1.5), ("背景比例", self.background, 0, 1), ("总音量", self.master, 0, 0.6)):
            ttk.Label(controls, text=label, style="Muted.TLabel").pack(anchor="w")
            ttk.Scale(controls, from_=lo, to=hi, variable=var, length=245).pack(fill="x", pady=(0, 9))
        ttk.Button(controls, text="关注区域回中", command=lambda: setattr(self, "center", (0.0, 0.0))).pack(fill="x", pady=4)
        ttk.Label(controls, text="正负声部 · 共用强度标尺").pack(anchor="w", pady=(16, 6))
        for value, label in (("both", "同时听正负"), ("positive", "仅正声部 +"), ("negative", "仅负声部 −")):
            ttk.Radiobutton(controls, text=label, value=value, variable=self.mode).pack(anchor="w", pady=3)
        ttk.Label(controls, text="绿色 + / 紫色 −\n位置表示方位，音色表示符号\n主声场最多 4 路 · 虚拟距离固定", style="Muted.TLabel").pack(anchor="w", pady=12)
        ttk.Label(shell, textvariable=self.summary, style="Muted.TLabel").pack(anchor="w", pady=(9, 5))
        columns = ("id", "sign", "xy", "strength", "gain", "angle")
        self.table = ttk.Treeview(shell, columns=columns, show="headings", height=4, selectmode="none")
        for key, label, width in zip(columns, ("声源", "符号", "位置 / µm", "原始强度", "播放增益", "角标签 / °"), (180, 65, 180, 130, 130, 130)):
            self.table.heading(key, text=label)
            self.table.column(key, width=width, anchor="center")
        self.table.pack(fill="x")
        audio_row = ttk.Frame(shell)
        audio_row.pack(fill="x", pady=(12, 6))
        self.device_combo = ttk.Combobox(audio_row, textvariable=self.device, values=["系统默认设备"], state="readonly", width=29)
        self.device_combo.pack(side="left")
        ttk.Checkbutton(audio_row, text="请求 HRTF", variable=self.hrtf).pack(side="left", padx=8)
        self.audio_button = ttk.Button(audio_row, text="开始试听 / 重连", command=self.open_audio)
        self.audio_button.pack(side="left")
        ttk.Button(audio_row, text="静音", command=self.mute).pack(side="left", padx=6)
        ttk.Button(audio_row, text="刷新设备", command=self.refresh_devices).pack(side="left")
        ttk.Button(audio_row, text="导出诊断", command=self.export_diagnostics).pack(side="right")
        ttk.Label(shell, textvariable=self.status, style="Muted.TLabel", wraplength=1040).pack(anchor="w")

    def refresh_devices(self):
        if self.no_audio or self.opening:
            return
        try:
            from ..audio import enumerate_devices
            self.device_combo.configure(values=["系统默认设备", *enumerate_devices(self.dll_path)])
        except Exception as exc:
            self.status.set(f"设备枚举失败：{exc}")

    def open_audio(self):
        if self.no_audio:
            self.status.set("此窗口以 --no-audio 启动，仅做界面和映射预览")
            return
        if self.opening:
            return
        self.opening = True
        self._mute_requested = False
        self.audio_ready = False
        self.status.set("正在打开音频设备…")
        device_name = None if self.device.get() == "系统默认设备" else self.device.get()
        hrtf = self.hrtf.get()

        def work():
            try:
                from ..audio import AudioConfig, AudioEngine
                if self.engine is None:
                    self.engine = AudioEngine()
                else:
                    self.engine.close()
                self.engine.open(AudioConfig(dll_path=self.dll_path, device_name=device_name, hrtf=hrtf))
                self._open_result = (True, "")
            except Exception as exc:
                self._open_result = (False, str(exc))
            finally:
                if self.closing and self.engine:
                    self.engine.close()
        threading.Thread(target=work, name="sonic-device-open", daemon=True).start()

    def mute(self):
        self._mute_requested = True
        self.audio_ready = False
        if self.engine and not self.opening:
            self.engine.stop()
        self.status.set("已静音 · 点击“开始试听 / 重连”恢复")

    def toggle_play(self):
        self.transport.playing = not self.transport.playing
        self.play_button.configure(text="暂停" if self.transport.playing else "播放")

    def reset(self):
        self.transport.sim_time_s = 0.0
        self.sequence = 0

    def bounds(self):
        return (35, 28, max(100, self.canvas.winfo_width()-35), max(100, self.canvas.winfo_height()-34))

    def move_roi(self, event):
        left, top, right, bottom = self.bounds()
        self.center = (max(-1, min(1, 2*(event.x-left)/(right-left)-1)),
                       max(-1, min(1, 1-2*(event.y-top)/(bottom-top))))

    def _draw(self, attention):
        c = self.canvas
        c.delete("all")
        left, top, right, bottom = self.bounds()
        def xy(x, y):
            return left+(x+1)*(right-left)/2, top+(1-y)*(bottom-top)/2
        for v in (-1, -0.5, 0, 0.5, 1):
            px, py = xy(v, v)
            c.create_line(px, top, px, bottom, fill="#2b3b4c")
            c.create_line(left, py, right, py, fill="#2b3b4c")
        c.create_text(left, 12, text="+y 向上 / 仰角", fill=MUTED, anchor="w")
        c.create_text(right, bottom+19, text="+x 向右 / 方位角", fill=MUTED, anchor="e")
        cx, cy = xy(*self.center)
        rx, ry = attention.radius*(right-left)/2, attention.radius*(bottom-top)/2
        c.create_oval(cx-rx, cy-ry, cx+rx, cy+ry, outline="#e7bd75", width=2, dash=(6, 4))
        c.create_line(cx-5, cy, cx+5, cy, fill="#e7bd75")
        c.create_line(cx, cy-5, cx, cy+5, fill="#e7bd75")
        gains = {s.source_id: s.gain for s in self.scene.sources}
        for i, o in enumerate(self.sample.observations):
            px, py = xy(o.position_m[0]/1e-6, o.position_m[1]/1e-6)
            color = POS if o.sign > 0 else NEG
            radius = 8 + 14*math.sqrt(min(o.strength, 1))
            # Concentric sign outlines preserve true position for colocated sources.
            if o.sign < 0:
                radius += 4
            c.create_oval(px-radius, py-radius, px+radius, py+radius, outline=color,
                          width=3 if gains.get(o.source_id, 0) > 0 else 1)
            c.create_text(px, py + (radius+13)*(1 if o.sign > 0 else -1),
                          text=("+ " if o.sign > 0 else "− ")+o.source_id, fill=color)
            a = o.orientation_rad
            c.create_line(px, py, px+radius*math.cos(a), py-radius*math.sin(a), fill=color, arrow="last")
        if self.sample.validity != "valid":
            c.create_text((left+right)/2, (top+bottom)/2, text=f"数据状态：{self.sample.validity}\n物理声源静音", fill="#ffc092", font=("Microsoft YaHei UI", 16, "bold"))

    def _tick(self):
        if self.closing:
            return
        if self._tick_id is not None:
            self.window.after_cancel(self._tick_id)
            self._tick_id = None
        now = time.monotonic()
        self.transport.static_listen = self.static.get()
        self.transport.advance(now-self._last_tick)
        self._last_tick = now
        if self._open_result is not None:
            ok, error = self._open_result
            self._open_result = None
            self.opening = False
            self.audio_ready = ok and not self._mute_requested
            self.status.set(("已静音" if self._mute_requested else "音频已开启") if ok else f"无法开启音频：{error}")
        scenario = self._labels[self.scenario.get()]
        self.sample = make_sample(scenario, self.transport.sim_time_s, self.sequence)
        self.sequence += 1
        attention = Attention(self.center, self.radius.get(), self.background.get())
        self.scene = map_sample(self.sample, attention, mode=self.mode.get(), master_gain=self.master.get(), audible=self.transport.audible)
        if self.audio_ready and not self.opening:
            try:
                self.engine.update(self.scene)
                if now-self._last_diag > 0.5:
                    self.status.set(audio_status(self.engine.diagnostics()))
                    self._last_diag = now
            except Exception as exc:
                self.audio_ready = False
                self.status.set(f"音频故障：{exc} · 请重连")
        self.time_label.set(f"t = {self.transport.sim_time_s*1e9:.2f} ns · 演示时标 1 ns/s")
        positive = sum(o.strength for o in self.sample.observations if o.sign > 0)
        negative = sum(o.strength for o in self.sample.observations if o.sign < 0)
        orientation_note = " · 角标签仅可视化，尚未映射声音" if scenario == "orientation" else ""
        self.summary.set(f"标签强度 Σ+ {positive:.2f}   Σ− {negative:.2f}   净值 {positive-negative:.2f}   绝对和 {positive+negative:.2f}  · 非物理 Q   |   {self.sample.validity} · 覆盖 {self.sample.coverage:.0%}{orientation_note}")
        self._draw(attention)
        self.table.delete(*self.table.get_children())
        gains = {s.source_id: s.gain for s in self.scene.sources}
        for o in self.sample.observations:
            self.table.insert("", "end", values=(o.source_id, "+" if o.sign > 0 else "−",
                f"{o.position_m[0]*1e6:.2f}, {o.position_m[1]*1e6:.2f}", f"{o.strength:.3f}",
                f"{gains.get(o.source_id, 0):.4f}", f"{math.degrees(o.orientation_rad):.1f}"))
        self._tick_id = self.window.after(33, self._tick)

    def export_diagnostics(self):
        filename = filedialog.asksaveasfilename(defaultextension=".json", initialfile="sonic-diagnostics.json", filetypes=[("JSON", "*.json")])
        if filename:
            payload = {"source_kind": "synthetic", "physical_topology_computed": False,
                       "scenario": self._labels[self.scenario.get()], "gain": self.master.get(),
                       "roi": {"center": self.center, "radius": self.radius.get(), "background": self.background.get()},
                       "audio": self.engine.diagnostics() if self.engine else {"state": "not_open"}}
            Path(filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def close(self):
        self.closing = True
        if self._tick_id:
            self.window.after_cancel(self._tick_id)
        if self.engine and not self.opening:
            self.engine.close()
        self.window.destroy()


def run(no_audio=False, dll_path=None):
    configure_dpi()
    root = tk.Tk()
    SonicApp(root, no_audio=no_audio, dll_path=dll_path)
    root.mainloop()


def configure_dpi():
    """Use physical window coordinates on Windows displays with scaling."""
    import ctypes
    import sys
    if sys.platform == "win32":
        ctypes.windll.user32.SetProcessDPIAware()
