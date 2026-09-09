from __future__ import annotations

import math
from pathlib import Path
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog

from ..attention import Attention
from ..mapping import map_sample_with_report
from ..model import SonicScene, MAX_SOURCE_BUDGET
from .coverage import selection_summary
from ..session import Transport
from ..reporting import report_json
from ..sources.synthetic import SCENARIOS, make_sample

FIELD_SCENARIOS = {'field:skyrmion': '拓扑 · 解析单纹理', 'field:opposite_pair': '拓扑 · 正负净零双纹理',
                   'field:uniform': '拓扑 · 均匀场', 'field:wall_inplane': '取向 · 面内域轴',
                   'field:wall_pma': '取向 · 面外域轴',
                   'field:activity_rotation': '活动 · 均匀旋转', 'field:activity_localized': '活动 · 局域旋转',
                   'field:band_in': '频带 · 带内 10 GHz', 'field:band_out': '频带 · 带外 30 GHz',
                   'field:band_opposite': '频带 · 空间反相', 'field:band_mixed': '频带 · 带内外分区'}

ACTIVITY_REASONS = {
    'no previous physical frame': '等待前一物理帧，不能把首帧当作零活动',
    'adjacent dynamic frames compared': '由相邻物理帧计算；暂停保留该测量值',
    'physical input sequence has a gap': '源数据缺帧，不跨缺口计算活动',
    'physical time interval exceeds max_dt_s': '物理采样间隔超过设置上限',
    'activity requires dynamic field frames': '静态或松弛数据不计算物理活动速率',
    'not every material site has a valid vector pair': '存在无效矢量对，覆盖不足，声音静音',
    'field segment changed': '仿真阶段变化，等待同阶段相邻帧',
    'material mask changed': '材料区域变化，等待新的相邻帧',
}

BAND_REASONS = {
    'full contiguous dynamic window analyzed': '完整历史窗已计算',
    'need a full contiguous physical window': '等待连续历史；缺帧或阶段变化后重新预热',
    'band power requires dynamic field frames': '静态或松弛数据不计算物理频谱',
    'band upper edge must be strictly below Nyquist': '上限须低于采样 Nyquist 频率',
    'band contains fewer than two positive FFT bins': '窗口过短或频带过窄，请增加窗口帧数',
    'nonuniform physical timestamps require resampling': '非均匀采样，当前不支持直接计算频谱',
    'not every material site has valid vectors over the full window': '历史窗存在无效矢量，声音静音',
}

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
    voices = f"目标 {diagnostic.get('target_source_count', 0)} / 占用 {diagnostic.get('active_source_count', 0)} 路"
    return f"{diagnostic.get('device', '音频设备')} · {spatial} · {state} / {quality} · {voices}"


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
        self.selection_report = None
        self.source_budget = tk.IntVar(value=4)
        self.aggregation_mode = tk.StringVar(value='fixed')
        self._aggregation_key = None
        self._aggregation_view = None
        from .aggregation_worker import LatestAggregation
        self._aggregation_worker = LatestAggregation()
        self._aggregation_base = None
        self.selection_note = tk.StringVar()
        self._labels = {v: k for k, v in SCENARIOS.items()}
        self._labels.update({v: k for k, v in FIELD_SCENARIOS.items()})
        self.field_view = None
        self.replay = None
        self._field_cache = None
        self._field_key = None
        self.max_dt_s = None
        from ..observers.band import BandConfig
        self.band_config = BandConfig()
        self.band_reference = 0.005
        self.band_low = tk.StringVar(value='8')
        self.band_high = tk.StringVar(value='12')
        self.band_window = tk.StringVar(value='256')
        self.band_axis = tk.StringVar(value='0,0,1')
        self.activity_reference = 1e9  # fixed rad/s reference, not frame normalization
        self.replay_recipe = tk.StringVar(value='拓扑')
        self.speed = tk.StringVar(value='1')
        self.seek_ns = tk.StringVar(value='0')
        self.max_dt_ns = tk.StringVar(value='')
        self.data_note = tk.StringVar(value='')
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
        self.subtitle = tk.StringVar(value='P1 合成声源 / P2a 三分量场观察')
        self.legend = tk.StringVar(value='绿色 + / 紫色 −\n位置表示方位，音色表示符号\n主声场最多 4 路 · 虚拟距离固定')
        self._build()
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.bind("<space>", lambda event: self.toggle_play())
        self._tick()

    def _build(self):
        w = self.window
        # Keep the full control panel visible even with Windows desktop scaling.
        w.tk.call("tk", "scaling", 96 / 72)
        w.title("MuMax-Sonic · 场观察与空间听觉")
        w.geometry("1120x1000")
        w.minsize(1050, 980)
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
        shell = ttk.Frame(w, padding=18)
        shell.pack(fill="both", expand=True)
        ttk.Label(shell, text="MuMax-Sonic", style="Title.TLabel").pack(anchor="w")
        ttk.Label(shell, textvariable=self.subtitle, style="Muted.TLabel").pack(anchor="w", pady=(2, 8))
        toolbar = ttk.Frame(shell)
        toolbar.pack(fill="x", pady=(0, 8))
        ttk.Label(toolbar, text="场景").pack(side="left", padx=(0, 8))
        combo = ttk.Combobox(toolbar, textvariable=self.scenario, values=list(self._labels), state="readonly", width=24)
        combo.pack(side="left")
        self.scenario_combo = combo
        combo.bind("<<ComboboxSelected>>", lambda e: self.reset())
        self.play_button = ttk.Button(toolbar, text="播放", command=self.toggle_play)
        self.play_button.pack(side="left", padx=8)
        ttk.Button(toolbar, text="重置", command=self.reset).pack(side="left")
        ttk.Checkbutton(toolbar, text="暂停时探听静态", variable=self.static).pack(side="left", padx=14)
        ttk.Label(toolbar, textvariable=self.time_label, style="Muted.TLabel").pack(side="right")
        timeline = ttk.Frame(shell)
        timeline.pack(fill='x', pady=(0, 4))
        ttk.Label(timeline, text='回放配方').pack(side='left')
        recipe_box = ttk.Combobox(timeline, textvariable=self.replay_recipe, values=['拓扑', '活动', '频带'], state='readonly', width=6)
        recipe_box.pack(side='left', padx=5)
        recipe_box.bind('<<ComboboxSelected>>', lambda e: self.invalidate_field())
        ttk.Label(timeline, text='倍速').pack(side='left')
        speed_box = ttk.Combobox(timeline, textvariable=self.speed, values=['0.25', '0.5', '1', '2', '4'], state='readonly', width=5)
        speed_box.pack(side='left', padx=5)
        speed_box.bind('<<ComboboxSelected>>', self.change_speed)
        ttk.Label(timeline, text='跳转/ns').pack(side='left')
        ttk.Entry(timeline, textvariable=self.seek_ns, width=9).pack(side='left', padx=5)
        ttk.Button(timeline, text='跳转', command=self.seek_to_entry).pack(side='left')
        ttk.Button(timeline, text='上一帧', command=lambda: self.step_frame(-1)).pack(side='left', padx=4)
        ttk.Button(timeline, text='下一帧', command=lambda: self.step_frame(1)).pack(side='left')
        ttk.Label(timeline, text='最大间隔/ns').pack(side='left', padx=(8, 3))
        ttk.Entry(timeline, textvariable=self.max_dt_ns, width=7).pack(side='left')
        ttk.Button(timeline, text='应用', command=self.apply_max_dt).pack(side='left', padx=4)
        band_row = ttk.Frame(shell)
        band_row.pack(fill='x', pady=(0, 4))
        for label, variable, width in [('频带下限/GHz', self.band_low, 6), ('上限/GHz', self.band_high, 6), ('窗口/帧', self.band_window, 6), ('参考轴 XYZ', self.band_axis, 12)]:
            ttk.Label(band_row, text=label).pack(side='left', padx=(0, 4))
            ttk.Entry(band_row, textvariable=variable, width=width).pack(side='left', padx=(0, 8))
        ttk.Button(band_row, text='应用频带', command=self.apply_band).pack(side='left')
        ttk.Label(band_row, text='声源上限 / 分摊音量').pack(side='left', padx=(16, 4))
        ttk.Combobox(band_row, textvariable=self.source_budget, values=list(range(1, MAX_SOURCE_BUDGET+1)), state='readonly', width=4).pack(side='left')
        ttk.Label(band_row, text='空间聚合').pack(side='left', padx=(12, 4))
        ttk.Combobox(band_row, textvariable=self.aggregation_mode, values=['fixed', 'adaptive'], state='readonly', width=9).pack(side='left')
        ttk.Label(shell, textvariable=self.data_note, style='Muted.TLabel', wraplength=1040).pack(anchor='w', pady=(0, 5))
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
        ttk.Label(controls, text="正负声部 · 共用强度标尺").pack(anchor="w", pady=(8, 6))
        self.sign_buttons = []
        for value, label in (("both", "同时听正负"), ("positive", "仅正声部 +"), ("negative", "仅负声部 −")):
            button = ttk.Radiobutton(controls, text=label, value=value, variable=self.mode)
            button.pack(anchor="w", pady=3)
            self.sign_buttons.append(button)
        ttk.Label(controls, textvariable=self.legend, style="Muted.TLabel").pack(anchor="w", pady=6)
        ttk.Label(shell, textvariable=self.summary, style="Muted.TLabel").pack(anchor="w", pady=(9, 5))
        ttk.Label(shell, textvariable=self.selection_note, style="Muted.TLabel", wraplength=1040).pack(anchor="w", pady=(0, 5))
        columns = ("id", "sign", "xy", "strength", "gain", "angle", "selection")
        self.table = ttk.Treeview(shell, columns=columns, show="headings", height=4, selectmode="none")
        for key, label, width in zip(columns, ("声源", "符号", "位置 / µm", "原始强度", "目标增益", "角标签 / °", "范围 / 输出候选"), (160, 55, 155, 100, 100, 100, 160)):
            self.table.heading(key, text=label)
            self.table.column(key, width=width, anchor="center")
        self.table.pack(fill="x")
        audio_row = ttk.Frame(shell)
        audio_row.pack(fill="x", pady=(12, 6))
        self.device_combo = ttk.Combobox(audio_row, textvariable=self.device, values=["系统默认设备"], state="readonly", width=23)
        self.device_combo.pack(side="left")
        ttk.Checkbutton(audio_row, text="请求 HRTF", variable=self.hrtf).pack(side="left", padx=8)
        self.audio_button = ttk.Button(audio_row, text="试听 / 重连", command=self.open_audio)
        self.audio_button.pack(side="left")
        ttk.Button(audio_row, text="静音", command=self.mute).pack(side="left", padx=6)
        ttk.Button(audio_row, text="刷新设备", command=self.refresh_devices).pack(side="left")
        ttk.Button(audio_row, text="加载场", command=self.load_field).pack(side="left", padx=4)
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
        self._last_tick = time.monotonic()
        self.play_button.configure(text="暂停" if self.transport.playing else "播放")

    def reset(self):
        self.transport.seek(self.replay.frames[0].sim_time_s if self.replay and self._labels[self.scenario.get()] == 'replay' else 0.0)
        self._last_tick = time.monotonic()
        self.invalidate_field()
        self.sequence = 0

    def invalidate_field(self):
        self._field_key = None

    def change_speed(self, event=None):
        self.transport.set_speed(float(self.speed.get()))
        self._last_tick = time.monotonic()

    def seek_to(self, value):
        if not math.isfinite(value) or value < 0:
            raise ValueError('跳转时间必须有限且非负')
        if self.replay and self._labels[self.scenario.get()] == 'replay':
            value = max(self.replay.frames[0].sim_time_s, min(self.replay.frames[-1].sim_time_s, value))
        self.transport.seek(value)
        self.transport.playing = False
        self.play_button.configure(text='播放')
        self._last_tick = time.monotonic()
        self.invalidate_field()

    def seek_to_entry(self):
        try:
            self.seek_to(float(self.seek_ns.get())*1e-9)
        except ValueError as exc:
            self.status.set(f'跳转失败：{exc}')

    def step_frame(self, delta):
        scenario = self._labels[self.scenario.get()]
        if scenario == 'replay' and self.replay:
            index = self.replay.index_at(self.transport.sim_time_s)
            index = max(0, min(len(self.replay.frames)-1, index+delta))
            self.seek_to(self.replay.frames[index].sim_time_s)
        else:
            from ..sources.activity_demo import STEP_S
            if scenario.startswith('field:band_'):
                from ..sources.band_demo import STEP_S
            index = int(round(self.transport.sim_time_s/STEP_S))
            self.seek_to(max(0, index+delta)*STEP_S)

    def apply_band(self):
        from ..observers.band import BandConfig
        try:
            config = BandConfig(float(self.band_low.get())*1e9, float(self.band_high.get())*1e9,
                                int(self.band_window.get()), tuple(float(v) for v in self.band_axis.get().split(',')))
            self.band_config = config
            self.invalidate_field()
        except (ValueError, TypeError) as exc:
            self.status.set(f'频带设置失败：{exc}')

    def apply_max_dt(self):
        try:
            value = float(self.max_dt_ns.get())*1e-9 if self.max_dt_ns.get().strip() else None
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError('间隔必须有限且大于零；留空表示不设阈值')
            self.max_dt_s = value
            self.invalidate_field()
        except ValueError as exc:
            self.status.set(f'间隔设置失败：{exc}')

    def load_field(self):
        path = filedialog.askopenfilename(filetypes=[('NPZ replay', '*.npz'), ('OVF sequence manifest', '*.json')])
        if not path:
            return
        try:
            from ..sources.ovf_replay import load_field_replay
            self.replay = load_field_replay(path)
            self._labels['回放 · 三分量场'] = 'replay'
            self.scenario_combo.configure(values=list(self._labels))
            self.scenario.set('回放 · 三分量场')
            self._field_key = None
            self.reset()
        except Exception as exc:
            self.status.set(f'场文件加载失败：{exc}')

    def _get_field_view(self, scenario):
        from ..fields import FieldFrame
        from ..field_pipeline import observe_field
        from ..sources.analytic import make_field
        previous = None
        history = None
        if scenario == 'replay':
            previous, frame = self.replay.pair_at(self.transport.sim_time_s)
            recipe, axes = {'活动': 'activity', '频带': 'band'}.get(self.replay_recipe.get(), 'topology'), {}
            key = (scenario, self.replay.sha256, frame.sequence, recipe, self.max_dt_s, self.band_config)
            index = self.replay.index_at(self.transport.sim_time_s)
            history = self.replay.frames[max(0, index-self.band_config.window_samples+1):index+1]
            if self.transport.sim_time_s >= self.replay.frames[-1].sim_time_s:
                self.transport.sim_time_s = self.replay.frames[-1].sim_time_s
                self.transport.playing = False
                self.play_button.configure(text='播放')
        else:
            name = scenario.split(':', 1)[1]
            if name.startswith('band_'):
                from ..sources.band_demo import make_band_frame, STEP_S
                tick = int(self.transport.sim_time_s/STEP_S + 1e-9)
                key = (scenario, tick, self.band_config)
                if key != self._field_key:
                    history = tuple(make_band_frame(name, i) for i in range(max(0, tick-self.band_config.window_samples+1), tick+1))
                    self._field_cache = observe_field(history[-1], 'band', history=history, band_config=self.band_config)
                    self._field_key = key
                return self._field_cache
            tick = int(self.transport.sim_time_s / 5e-11 + 1e-9) if name.startswith(('wall', 'activity_')) else 0
            key = (scenario, tick, self.max_dt_s)
            if key == self._field_key:
                return self._field_cache
            if name.startswith('activity_'):
                from ..sources.activity_demo import make_activity_frame
                frame = make_activity_frame(name, tick)
                previous = make_activity_frame(name, tick-1) if tick else None
                self._field_cache = observe_field(frame, 'activity', previous=previous, max_dt_s=self.max_dt_s)
                self._field_key = key
                return self._field_cache
            t = tick*5e-11
            vectors = make_field(name, t)
            spacing = 2e-6/(vectors.shape[1]-1)
            frame = FieldFrame(vectors, spacing, spacing, t, (-1e-6, -1e-6, 0),
                               sequence=tick, time_kind='dynamics' if name.startswith('wall') else 'static',
                               provenance=f'analytic:{name}')
            recipe = 'direction' if name.startswith('wall') else 'topology'
            axes = dict(domain_axis=(0, 0, 1), reference_axis=(1, 0, 0)) if name == 'wall_pma' else {}
        if key != self._field_key:
            self._field_cache = observe_field(frame, recipe, previous=previous, max_dt_s=self.max_dt_s, history=history, band_config=self.band_config, **axes)
            self._field_key = key
        return self._field_cache

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
        if self.field_view is not None:
            # Sparse XYZ view: arrow XY, colour encodes laboratory z.
            field = self.field_view.field
            vectors = field.vectors
            for j in range(0, vectors.shape[0], max(1, vectors.shape[0]//12)):
                for i in range(0, vectors.shape[1], max(1, vectors.shape[1]//12)):
                    u = vectors[j, i]
                    norm = math.sqrt(sum(float(v)**2 for v in u))
                    if not field.mask[j, i] or not math.isfinite(norm) or norm < 1e-12:
                        continue
                    u = u/norm
                    px, py = xy((field.origin_m[0]+i*field.dx_m-attention.origin_m[0])/attention.extent_m,
                                (field.origin_m[1]+j*field.dy_m-attention.origin_m[1])/attention.extent_m)
                    color = '#688cbe' if u[2] >= 0 else '#bc8063'
                    c.create_line(px-7*u[0], py+7*u[1], px+7*u[0], py-7*u[1], fill=color, arrow='last')
                    c.create_oval(px-2, py-2, px+2, py+2, fill=color, outline='')
        c.create_text(left, 12, text="+y 向上 / 仰角", fill=MUTED, anchor="w")
        c.create_text(right, bottom+19, text="+x 向右 / 方位角", fill=MUTED, anchor="e")
        c.create_text(left, bottom+19, text='粗圈：中心在范围内  ·  白点：输出候选（非设备发声状态）',
                      fill=TEXT, anchor='w', font=('Microsoft YaHei UI', 8))
        cx, cy = xy(*self.center)
        rx, ry = attention.radius*(right-left)/2, attention.radius*(bottom-top)/2
        c.create_oval(cx-rx, cy-ry, cx+rx, cy+ry, outline="#e7bd75", width=2, dash=(6, 4))
        c.create_line(cx-5, cy, cx+5, cy, fill="#e7bd75")
        c.create_line(cx, cy-5, cx, cy+5, fill="#e7bd75")
        gains = {s.source_id: s.gain for s in self.scene.sources}
        activity = self.field_view is not None and self.field_view.diagnostic['recipe'] == 'activity'
        band = self.field_view is not None and self.field_view.diagnostic['recipe'] == 'band'
        for i, o in enumerate(self.sample.observations):
            px, py = xy((o.position_m[0]-attention.origin_m[0])/attention.extent_m, (o.position_m[1]-attention.origin_m[1])/attention.extent_m)
            color = POS if o.sign > 0 else NEG
            radius = 8 + 14*math.sqrt(min(o.strength/(self.activity_reference if activity else (self.band_reference if band else 1)), 1))
            # Concentric sign outlines preserve true position for colocated sources.
            if o.sign < 0:
                radius += 4
            c.create_oval(px-radius, py-radius, px+radius, py+radius, outline=color,
                          width=3 if attention.contains(o) else 1,
                          dash=(3, 3) if o.source_id.endswith(':background') else (), tags=(f'source:{o.source_id}',))
            if gains.get(o.source_id, 0) > 0:
                c.create_oval(px-3, py-3, px+3, py+3, fill=TEXT, outline='', tags=(f'selected:{o.source_id}',))
            if self.field_view is None or gains.get(o.source_id, 0) > 0:
                label = o.source_id
                if label.startswith('adaptive:'):
                    parts = label.split(':')
                    label = {'background': '背景', 'focus': '前景', 'overview': '总览'}[parts[2]]
                    if len(parts) > 3:
                        label += parts[3].replace('root', '').replace('.', '')
                c.create_text(px, py + (radius+13)*(1 if o.sign > 0 else -1),
                              text=('b ' if band else 'a ' if activity else ('φ ' if o.orientation_enabled else ("+ " if o.sign > 0 else "− ")))+label, fill=color)
            a = o.orientation_rad
            if o.orientation_enabled or self.field_view is None:
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
        if scenario.startswith('field:') or scenario == 'replay':
            try:
                self.field_view = self._get_field_view(scenario)
                self.sample = self.field_view.sample
            except Exception as exc:
                self.field_view = None
                self.sample = make_sample('invalid', self.transport.sim_time_s, self.sequence)
                self.status.set(f'场观察失败：{exc}')
        else:
            self.field_view = None
            self.sample = make_sample(scenario, self.transport.sim_time_s, self.sequence)
        self.sequence += 1
        field = self.field_view.field if self.field_view else None
        attention = Attention(self.center, self.radius.get(), self.background.get(),
                              field.extent_m if field else 1e-6, field.center_m[:2] if field else (0, 0))
        draw_attention = attention
        aggregation_pending = False
        if self.field_view is not None:
            from ..field_pipeline import apply_aggregation
            key = (id(self.field_view), attention, self.source_budget.get(), self.aggregation_mode.get())
            base = self.field_view
            adaptive = (self.aggregation_mode.get() == 'adaptive' and base.contributions is not None
                        and base.sample.validity == 'valid' and base.sample.coverage == 1
                        and not self.transport.playing)
            if adaptive:
                self._aggregation_worker.request(key, base, attention, self.source_budget.get())
                completed = self._aggregation_worker.poll()
                if completed is not None:
                    done_key, done_view, error = completed
                    if done_key[0] == id(base) and done_key[2:] == key[2:]:
                        if error is None:
                            self._aggregation_base = base
                            self._aggregation_key, self._aggregation_view = done_key, done_view
                        else:
                            self.status.set(f'空间聚合失败：{error}')
                compatible = (self._aggregation_base is base and self._aggregation_key is not None
                              and self._aggregation_key[2:] == key[2:])
                if compatible:
                    self.field_view = self._aggregation_view
                    attention = self._aggregation_key[1]
                else:
                    from dataclasses import replace
                    self.field_view = replace(base, sample=replace(base.sample, observations=(), validity='warming_up'))
                aggregation_pending = self._aggregation_key != key
            else:
                self.field_view = apply_aggregation(base, attention, self.source_budget.get(), self.aggregation_mode.get())
            self.sample = self.field_view.sample
        is_activity = self.field_view is not None and self.field_view.diagnostic['recipe'] == 'activity'
        is_band = self.field_view is not None and self.field_view.diagnostic['recipe'] == 'band'
        unsigned = is_activity or is_band
        for button in self.sign_buttons:
            button.state(['disabled'] if unsigned else ['!disabled'])
        mapped = map_sample_with_report(self.sample, attention, budget=self.source_budget.get(), mode='both' if unsigned else self.mode.get(), master_gain=self.master.get(), audible=self.transport.audible,
                                strength_reference=self.activity_reference if is_activity else (self.band_reference if is_band else 1.0))
        self.scene = mapped.scene
        self.selection_report = mapped.report
        if self.field_view:
            from ..field_pipeline import field_selection_report
            self.selection_report = field_selection_report(self.field_view, self.selection_report)
        self.selection_note.set(selection_summary(self.selection_report))
        if self.audio_ready and not self.opening:
            try:
                self.engine.update(self.scene)
                if now-self._last_diag > 0.5:
                    self.status.set(audio_status(self.engine.diagnostics()))
                    self._last_diag = now
            except Exception as exc:
                self.audio_ready = False
                self.status.set(f"音频故障：{exc} · 请重连")
        self.time_label.set(f"t = {self.transport.sim_time_s*1e9:.2f} ns · {self.transport.playback_rate:g}×")
        self.data_note.set('倍速只改变浏览速度；最大间隔只用于活动配方，留空不推断采样间隔。')
        positive = sum(o.strength for o in self.sample.observations if o.sign > 0)
        negative = sum(o.strength for o in self.sample.observations if o.sign < 0)
        orientation_note = " · 角标签仅可视化，尚未映射声音" if scenario == "orientation" else ""
        self.summary.set(f"标签强度 Σ+ {positive:.2f}   Σ− {negative:.2f}   净值 {positive-negative:.2f}   绝对和 {positive+negative:.2f}  · 非物理 Q   |   {self.sample.validity} · 覆盖 {self.sample.coverage:.0%}{orientation_note}")
        if self.field_view:
            self.summary.set(self.field_view.summary + f' | {self.sample.validity} · 覆盖 {self.sample.coverage:.0%}')
            self.subtitle.set(f'场观察 · {self.sample.source_kind} · {self.sample.observations[0].quantity if self.sample.observations else "有效域观测"} · 箭头 XY / 蓝橙为 ±z')
            self.time_label.set(f'帧 t = {self.sample.sim_time_s*1e9:.2f} ns · {self.transport.playback_rate:g}×')
            if is_activity:
                diagnostic = self.field_view.diagnostic
                dt = diagnostic['dt_s']
                reason = ACTIVITY_REASONS.get(diagnostic['reason'], diagnostic['reason'])
                note = f'物理 Δt：{dt*1e12:.4g} ps · {reason}' if dt is not None else f'活动状态：{reason}'
                if 'sequence_inferred:v1' in self.field_view.field.provenance:
                    note += ' · 旧文件未保存帧号，无法按序号检测缺帧'
                self.data_note.set(note)
                self.legend.set(f'活动强度：rad/s · 无正负号\n固定声音参考 {self.activity_reference:.3g} rad/s\n暂停时探听保留测量值')
            elif is_band:
                self.subtitle.set(f'频带观察 · {self.sample.source_kind} · 归一化方向的横向均方强度 · 箭头 XY / 蓝橙为 ±z')
                d = self.field_view.diagnostic
                df = d['frequency_resolution_hz']
                resolution = f'{df/1e9:.3g} GHz' if df is not None else '待定'
                span = d['window_span_s']
                span_text = f'{span*1e9:.3g} ns' if span is not None else '待定'
                reason = BAND_REASONS.get(d['reason'], d['reason'])
                self.data_note.set(f"历史 {d['samples_available']}/{d['samples_required']} 帧 · 窗宽 {span_text} · 分辨率 {resolution} · {reason}")
                self.legend.set(f'横向频带均方强度 · 无正负号\n固定声音参考 {self.band_reference:g}\n逐点求功率后汇总 · 非物理能量')
            elif self.field_view.diagnostic['recipe'] == 'direction':
                self.legend.set('φ：音高 + 起伏速率编码\n强度为横向投影面积权重\n非拓扑符号 · 使用正声部输出')
            else:
                self.legend.set(f'绿色 Q+ / 紫色 Q−\n圆为分块贡献，非粒子检测\n最多 {self.source_budget.get()} 路 · 单路增益含 1/预算')
            input_info = self.field_view.diagnostic.get('input', {})
            if input_info.get('format') == 'OVF2':
                self.subtitle.set(f"OVF · {input_info['origin']} · {self.sample.time_kind} · XYZ 完整分量 · z 层 {input_info['z_index']} · 源帧号 {self.sample.sequence} · 箭头 XY / 蓝橙 ±z")
        else:
            self.subtitle.set('P1 · Synthetic 标签演示 · 未计算物理拓扑荷')
            self.legend.set(f'绿色 + / 紫色 −\n位置表示方位，音色表示符号\n最多 {self.source_budget.get()} 路 · 单路增益含 1/预算')
        if aggregation_pending:
            self.data_note.set(self.data_note.get() + ' · 空间聚合更新中，声音使用上次完成的关注区域')
        self._draw(draw_attention)
        self.table.delete(*self.table.get_children())
        self.table.heading('strength', text='贡献 / rad/s' if is_activity else ('均方贡献' if is_band else '原始强度'))
        gains = {s.source_id: s.gain for s in self.scene.sources}
        for o in self.sample.observations:
            self.table.insert("", "end", values=(o.source_id, '—' if unsigned else ("+" if o.sign > 0 else "−"),
                f"{o.position_m[0]*1e6:.2f}, {o.position_m[1]*1e6:.2f}", f"{o.strength:.3g}",
                f"{gains.get(o.source_id, 0):.4f}", f"{math.degrees(o.orientation_rad):.1f}",
                ('圈内' if draw_attention.contains(o) else '圈外') + (' / 已选入' if gains.get(o.source_id, 0) > 0 else ' / 已选但输出静音' if o.source_id in self.selection_report['selected_ids'] else ' / 未选入')))
        self._tick_id = self.window.after(33, self._tick)

    def export_diagnostics(self):
        filename = filedialog.asksaveasfilename(defaultextension=".json", initialfile="sonic-diagnostics.json", filetypes=[("JSON", "*.json")])
        if filename:
            payload = {"source_kind": "synthetic", "physical_topology_computed": False,
                       "scenario": self._labels[self.scenario.get()], "gain": self.master.get(),
                       "roi": {"center": self.center, "radius": self.radius.get(), "background": self.background.get()},
                       "audio": self.engine.diagnostics() if self.engine else {"state": "not_open"}}
            if self.field_view:
                payload.update(source_kind=self.sample.source_kind,
                               physical_topology_computed=self.field_view.diagnostic['recipe'] == 'topology',
                               field=self.field_view.diagnostic)
            payload.update(playback_rate=self.transport.playback_rate, activity_reference_rad_s=self.activity_reference,
                           activity_max_dt_s=self.max_dt_s, band_reference=self.band_reference, selection=self.selection_report, source_budget=self.source_budget.get())
            Path(filename).write_text(report_json(payload), encoding="utf-8")

    def close(self):
        self.closing = True
        self._aggregation_worker.close()
        if self._tick_id:
            self.window.after_cancel(self._tick_id)
        if self.engine and not self.opening:
            self.engine.close()
        self.window.destroy()


def run(no_audio=False, dll_path=None, field_demo=None, replay_path=None, recipe='topology', max_dt_s=None, activity_reference=1e9, band_config=None, band_reference=0.005, source_budget=4, aggregation='fixed'):
    configure_dpi()
    root = tk.Tk()
    app = SonicApp(root, no_audio=no_audio, dll_path=dll_path)
    app.replay_recipe.set({'activity': '活动', 'band': '频带'}.get(recipe, '拓扑'))
    if band_config is not None:
        app.band_config = band_config
        app.band_low.set(f'{band_config.low_hz/1e9:g}')
        app.band_high.set(f'{band_config.high_hz/1e9:g}')
        app.band_window.set(str(band_config.window_samples))
        app.band_axis.set(','.join(str(v) for v in band_config.reference_axis))
    app.band_reference = band_reference
    app.source_budget.set(source_budget)
    app.aggregation_mode.set(aggregation)
    app.max_dt_s = max_dt_s
    app.max_dt_ns.set('' if max_dt_s is None else f'{max_dt_s*1e9:g}')
    app.activity_reference = activity_reference
    if field_demo:
        app.scenario.set(FIELD_SCENARIOS[f'field:{field_demo}'])
        app.reset()
    if replay_path:
        from ..sources.ovf_replay import load_field_replay
        app.replay = load_field_replay(replay_path)
        app._labels['回放 · 三分量场'] = 'replay'
        app.scenario_combo.configure(values=list(app._labels))
        app.scenario.set('回放 · 三分量场')
        app.reset()
    root.mainloop()


def configure_dpi():
    """Use physical window coordinates on Windows displays with scaling."""
    import ctypes
    import sys
    if sys.platform == "win32":
        ctypes.windll.user32.SetProcessDPIAware()
