# MuMax-Sonic

面向微磁与相关磁性连续场仿真的空间可听化工具：用可解释的声音辅助感知状态、运动、频带活动与拓扑结构，并通过可调关注区域分配听觉注意力。

**状态：P2b 活动观察与回放时间处理已实现。** 在拓扑、连续取向与 NPZ 回放基础上，新增局部角变化率、倍速、逐帧/跳转与缺帧检查。已用解析场验证，尚未验证真实 MuMax 输出；频带、自动壁追踪和头追仍待实现。见 [P2b 实现与验证](docs/p2b-validation.md)。

## 运行活动观察

```powershell
python launch.py --field-demo activity_localized
python launch.py --field-demo activity_rotation
```

活动量使用完整 XYZ 方向和源数据相邻两帧的物理时间差，单位 rad/s。首帧等待前一帧；点击“下一帧”或“播放”得到测量值。倍速只改变浏览速度，暂停探听保留当前区间的测量值。跳转重新取源数据中的相邻帧，不跨跳转位置计算差分。

```powershell
python scripts/make_field_demo.py local/activity.npz --scenario activity_rotation --frames 40 --drop-frame 15
python launch.py --replay local/activity.npz --recipe activity
python launch.py --inspect-field local/activity.npz --recipe activity --report local/activity.json
```

窗口的“回放配方”切换拓扑/活动；“最大间隔/ns”可限制允许计算的物理间隔，留空不推断采样周期，命令行对应 `--max-dt-ps`。真实序号缺口始终使该帧活动未就绪。默认听觉参考为 `1e9 rad/s`，可用 `--activity-reference-rad-s` 调整，不改物理值。活动没有正负符号，窗口停用正负 solo；圈大小使用同一听觉参考。

## 运行场观察

在仓库根目录安装数值依赖并启动：

```powershell
python -m pip install "numpy>=1.26,<3"
python launch.py --field-demo opposite_pair
python launch.py --field-demo wall_inplane
```

默认无声，点击“试听 / 重连”打开音频；取向场景点击“播放”后连续扫角。下拉列表也提供均匀场、单纹理及面外域轴。拓扑显示计算所得 Q+/Q−/Qnet/Qabs；取向模式用音高和起伏速率编码角度，属于未完成人工标定的初步声音表达。P1 的测试音色保持原样。

```powershell
python scripts/make_field_demo.py local/pair.npz
python launch.py --replay local/pair.npz
python launch.py --inspect-field local/pair.npz --method finite_difference --report local/pair.json
```

“加载场”打开 NPZ，当前 GUI 回放支持拓扑或活动。保存格式已升级到 v2，保留原始帧序号和逐帧来源；仍可读 v1，但 v1 没有帧序号，无法据此检测缺帧，界面会提示。文件生成不覆盖已有文件。格式见 [P2b 文档](docs/p2b-validation.md)，拓扑边界见 [P2a 文档](docs/p2a-validation.md)；NPZ 尚不直接读取 OVF。

## 运行 P1

在仓库根目录，使用带 Tk 的 64 位 Python 3.11+：

```powershell
python scripts/install_audio.py
python launch.py
```

安装脚本从官方获取固定版本 OpenAL Soft 1.25.2，校验 SHA256，仅放入 `local/`。不需要 GPU；P1 音频使用标准库，P2a 场计算需要 NumPy。已有 DLL 可用 `--dll` 指定完整路径。只预览界面用 `python launch.py --no-audio`。

窗口默认不发声。选择耳机输出设备，点击“开始试听 / 重连”，从较低音量开始；“播放”推进合成时间，“暂停时探听静态”决定暂停后是否仍能听当前状态。拖动画面移动关注区域，调节半径与背景比例，再比较正负 solo。切换设备或 HRTF 请求后点击重连，底部显示实际 HRTF 状态。HRTF 对照时应避免叠加其他空间音效。

合成场景包括单源移动、同位/异位正负等强、弱内源与强外源、连续角标签、过期和无效数据。位置控制方位；正负使用两种音高标签，共用强度标尺。标签净值为零时两声部仍保留。连续角当前仅显示箭头与数值，尚未编码进声音；数字 RMS 相同不代表感知等响。

```powershell
python launch.py --doctor --report local/doctor.json
python launch.py --audio-smoke 15 --report local/smoke.json
python scripts/listening_check.py --play --calibrate
python -m pytest -q
```

`--doctor` 无声检查设备；`--audio-smoke` 会播放低增益移动声源，最长 1800 秒。盲听工具见 [试听说明](examples/listening_trials/README.md)，每项默认 20 次，结果保存在 `local/`。开发测试需要 pytest（`python -m pip install pytest`）；启动窗口无需测试依赖。

## 核心方向

界面标记：**粗圈表示声源中心位于关注圈内，白点表示选入当前非零增益的输出候选**。候选在无声预览时也显示，设备是否开启看底部状态。当前最多 4 路，背景比例大于零时通常预留 1 路圈外背景，因此圈内不保证全部发声；正负 solo 也影响选源。关注圈边缘有平滑过渡，不是硬静音边界。表格分别显示范围和候选状态。

- Windows 耳机空间声音，第一版支持手动关注区域，后续加入头部姿态微调。
- 四个核心模式：活动、纹理与连续取向、频带、自旋纹理拓扑。模式是观察配方，可以组合。
- 状态浏览与动态监测并存：静止的拓扑结构和畴壁也可以被主动探听。
- 观测对象可为磁化、子晶格、有明确物理定义的序参量，以及后续的弹性或能量场。
- 仿真后端与物理观测、声音表达解耦；先做合成数据与回放，随后适配 MuMax3、MuMax+。

CDW 是需求来源和未来案例之一；通用代码不依赖其目录、易轴、几何、材料、阈值或科学分类。

## 文档

| 文档 | 内容 |
| --- | --- |
| [概念与架构](docs/concept.md) | 产品边界、模式、最小数据接口、时间与性能、关注机制 |
| [物理观测与符号约定](docs/observables.md) | 连续取向、壁、频带、拓扑正负通道及验证要求 |
| [MuMax+ 能力调研](docs/mumaxplus-research.md) | 官方能力、代表性问题、适配限制、来源 |
| [P1 细化执行计划](docs/p1-execution-plan.md) | 最小可试听原型、任务依赖、代理分工与分层验收 |
| [协作规则](AGENTS.md) | Luna/Terra 偏好、文件归属和等待策略 |
| [实施与验收计划](docs/implementation-plan.md) | 阶段、交付物、验证矩阵和未决实验 |

上述文档是可版本化的项目设计依据，纳入 Git。原始聊天、个人研究数据、仿真输出和本地试验文件不属于首个提交。
