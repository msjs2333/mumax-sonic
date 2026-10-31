"""Simulation picker. Discovery does not run on the Tk or audio threads."""
from pathlib import Path
import threading
import tkinter as tk
from tkinter import ttk, filedialog

from ..sources.discovery import discover_simulations
from ..sources.attachment import AttachedSimulation
from ..sources.live import LiveConfig


class AttachmentDialog:
    def __init__(self, app, roots=(), directory=None, **options):
        self.app = app
        self.roots = [Path(p) for p in roots] or [Path.cwd()]
        self.window = tk.Toplevel(app.window)
        self.window.title('接入已有仿真 · MuMax-Sonic')
        self.window.geometry('870x530')
        self.window.transient(app.window)
        self.result = None
        self.busy = False
        self.candidates = ()
        self.closed = False
        self.directory = tk.StringVar(value=str(directory or ''))
        self.all_material = tk.BooleanVar(value=options.get('all_material', False))
        self.mask = tk.StringVar(value=str(options.get('mask_path') or ''))
        self.layer = tk.StringVar(value='' if options.get('z_index') is None else str(options['z_index']))
        self.time_kind = tk.StringVar(value=options.get('time_kind', 'dynamics'))
        self.pattern = options.get('pattern', 'm[0-9]*.ovf')
        self.note = tk.StringVar(value='正在发现本机 MuMax3 进程及指定目录…')
        shell = ttk.Frame(self.window, padding=14)
        shell.pack(fill='both', expand=True)
        ttk.Label(shell, text='选择已有输出；不修改或启动求解器。每 5 秒重新发现。').pack(anchor='w')
        self.list = ttk.Treeview(shell, columns=('path', 'process', 'frames'), show='headings', height=7)
        for key, label, width in [('path', '输出目录', 520), ('process', '进程证据', 130), ('frames', 'OVF 文件', 90)]:
            self.list.heading(key, text=label)
            self.list.column(key, width=width)
        self.list.pack(fill='both', expand=True, pady=8)
        self.list.bind('<<TreeviewSelect>>', self.select)
        row = ttk.Frame(shell); row.pack(fill='x')
        ttk.Button(row, text='重新扫描', command=self.scan).pack(side='left')
        ttk.Button(row, text='添加搜索目录', command=self.add_root).pack(side='left', padx=6)
        ttk.Label(shell, text='也可直接填写输出目录（不存在时等待创建）').pack(anchor='w', pady=(8, 0))
        ttk.Entry(shell, textvariable=self.directory).pack(fill='x')
        row = ttk.Frame(shell); row.pack(fill='x', pady=8)
        ttk.Checkbutton(row, text='确认全网格为材料', variable=self.all_material).pack(side='left')
        ttk.Button(row, text='或选择材料 mask.npy', command=self.choose_mask).pack(side='left', padx=8)
        ttk.Label(row, textvariable=self.mask, wraplength=350).pack(side='left')
        row = ttk.Frame(shell); row.pack(fill='x')
        ttk.Label(row, text='多层时填写 z 层索引').pack(side='left')
        ttk.Entry(row, textvariable=self.layer, width=5).pack(side='left', padx=6)
        ttk.Label(row, text='时间类型').pack(side='left')
        ttk.Combobox(row, textvariable=self.time_kind, values=['dynamics', 'static', 'relaxation'], state='readonly', width=12).pack(side='left', padx=6)
        ttk.Button(row, text='接入 / 切换 / 重新连接', command=self.attach).pack(side='right')
        ttk.Label(shell, textvariable=self.note, wraplength=820).pack(anchor='w', pady=(10, 0))
        ttk.Label(shell, text='需有完整 m*.ovf 和物理时间；MuMax+ 无预设采样接口时不能外部附加。声音需在主窗口点击试听。', wraplength=820).pack(anchor='w', pady=(6, 0))
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        self.scan()
        self.timer = self.window.after(100, self.poll)
        self.rescan = self.window.after(5000, self.rescan_tick)

    def scan(self):
        if self.busy or self.closed:
            return
        self.busy = True
        roots = tuple(self.roots)
        def work():
            try:
                self.result = discover_simulations(roots)
            finally:
                self.busy = False
        threading.Thread(target=work, name='sonic-discovery', daemon=True).start()

    def poll(self):
        if self.closed:
            return
        if self.result is not None:
            result, self.result = self.result, None
            self.candidates = result.candidates
            self.list.delete(*self.list.get_children())
            for i, candidate in enumerate(self.candidates):
                self.list.insert('', 'end', iid=str(i), values=(str(candidate.directory),
                    ', '.join(map(str, candidate.pids)) or '未检测到进程', '有' if candidate.has_frames else '等待'))
            detail = '；'.join(result.warnings[:2])
            self.note.set(f'发现 {len(self.candidates)} 个候选；未检测到进程不代表仿真停止。' + detail)
        self.timer = self.window.after(100, self.poll)

    def rescan_tick(self):
        self.scan()
        self.rescan = self.window.after(5000, self.rescan_tick)

    def select(self, _event=None):
        selection = self.list.selection()
        if selection:
            self.directory.set(str(self.candidates[int(selection[0])].directory))

    def add_root(self):
        path = filedialog.askdirectory(parent=self.window, title='选择输出目录或包含 .out 的工作目录')
        if path:
            self.roots.append(Path(path))
            self.directory.set(path)
            self.scan()

    def choose_mask(self):
        path = filedialog.askopenfilename(parent=self.window, filetypes=[('Boolean NumPy mask', '*.npy')])
        if path:
            self.mask.set(path)
            self.all_material.set(False)

    def attach(self):
        try:
            if not self.directory.get().strip():
                raise ValueError('请选择或填写输出目录')
            recipe = {'活动': 'activity', '频带': 'band'}.get(self.app.replay_recipe.get(), 'topology')
            source = AttachedSimulation(self.directory.get().strip(), LiveConfig(recipe=recipe,
                max_dt_s=self.app.max_dt_s, band_config=self.app.band_config,
                stale_after_s=self.app.live_stale_s), all_material=self.all_material.get(),
                mask_path=None if self.all_material.get() else self.mask.get() or None,
                z_index=int(self.layer.get()) if self.layer.get().strip() else None,
                time_kind=self.time_kind.get(), pattern=self.pattern)
            self.app.mute()
            self.app.attach_live_source(source.start())
            self.app.status.set(f'已接入 {source.directory} · 等待完整帧 · 点击试听开启声音')
            self.close()
        except (OSError, ValueError) as exc:
            self.note.set(str(exc))

    def close(self):
        self.closed = True
        self.window.after_cancel(self.timer)
        self.window.after_cancel(self.rescan)
        self.window.destroy()
