# MuMax-Sonic

面向微磁与相关磁性连续场仿真的空间可听化工具：用可解释的声音辅助感知状态、运动、频带活动与拓扑结构，并通过可调关注区域分配听觉注意力。

**状态：P2f 自适应聚合与背景汇总已实现。** 拓扑正负、活动和频带贡献先逐位置计算，再按关注区域分配 1–16 路声源。固定分块仍为默认，可切换对比；空间误差与贡献覆盖分别报告。见 [P2f 实现与验证](docs/p2f-validation.md)。

```powershell
python launch.py --field-demo opposite_pair --aggregation adaptive --source-budget 4
python launch.py --field-demo activity_localized --aggregation adaptive --source-budget 8
```

窗口“空间聚合”中 `fixed` 为固定分块，`adaptive` 为关注区细分和背景汇总。虚线声源表示背景汇总；100% 贡献覆盖不表示空间细节完整保留。连续取向目前回退固定分块。自适应声音在分区变化时复用邻近同声部的播放循环，并采用 120 ms 时间常数平滑位置与增益，减少拖动重新起音；暂停探听的重聚合在后台运行。

## 调整声源与查看覆盖

```powershell
python launch.py --field-demo activity_rotation --source-budget 16
python launch.py --field-demo opposite_pair --source-budget 8
```

窗口可直接调整“声源上限 / 分摊音量”，物理观察量不变。提高预算会增加混音余量、降低单路增益；贡献覆盖使用原始强度，不使用响度。拓扑正负分别统计，Qnet 为零时仍可显示覆盖。覆盖不等于体元空间细节保留率或耳机可辨率，详见 [P2e 文档](docs/p2e-validation.md)。

## 运行 OVF 回放

```powershell
python scripts/make_ovf_demo.py local/ovf-demo --frames 320
python scripts/make_ovf_manifest.py local/ovf-demo local/ovf-demo.json --limit 320 --origin synthetic --time-kind dynamics --all-material
python launch.py --replay local/ovf-demo.json --recipe band
```

以上生成解析场，不运行求解器。真实输出可换成相应目录并声明 `--origin simulation`；全材料、材料掩膜、层号和动态/松弛时间必须与源数据一致。OVF 缺少可靠时间时，需手工填写有依据的逐帧时间清单。支持范围、清单示例和真实数据验证见 [P2d 文档](docs/p2d-validation.md)。

## 运行频带观察

```powershell
python launch.py --field-demo band_mixed
python launch.py --field-demo band_opposite
python scripts/make_field_demo.py local/band.npz --scenario band_mixed --frames 320
python launch.py --replay local/band.npz --recipe band
```

默认 8–12 GHz、256 帧、参考轴 +z，解析数据每 5 ps 采样。点击播放，或跳转到 1.5 ns 查看已有历史；窗口显示预热、物理窗宽和分辨率。逐体元求频带功率后再空间汇总，空间反相不会被平均抵消。该量是归一化方向的横向均方强度，不是能量，也不自动识别自旋波。设置与限制见 [P2c 文档](docs/p2c-validation.md)。

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

窗口的“回放配方”切换拓扑/活动/频带；“最大间隔/ns”可限制允许计算的物理间隔，留空不推断采样周期，命令行对应 `--max-dt-ps`。真实序号缺口始终使该帧活动未就绪。默认听觉参考为 `1e9 rad/s`，可用 `--activity-reference-rad-s` 调整，不改物理值。活动没有正负符号，窗口停用正负 solo；圈大小使用同一听觉参考。

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

“加载场”打开 NPZ 或 OVF JSON 清单，当前 GUI 回放支持拓扑、活动或频带。保存格式已升级到 v2，保留原始帧序号和逐帧来源；仍可读 v1，但 v1 没有帧序号，无法据此检测缺帧，界面会提示。文件生成不覆盖已有文件。格式见 [P2b 文档](docs/p2b-validation.md)，拓扑边界见 [P2a 文档](docs/p2a-validation.md)；OVF 通过 [显式清单](docs/p2d-validation.md) 导入。

## 运行 P1

在仓库根目录，使用带 Tk 的 64 位 Python 3.11+：

```powershell
python -m pip install "numpy>=1.26,<3"
python scripts/install_audio.py
python launch.py
```

安装脚本从官方获取固定版本 OpenAL Soft 1.25.2，校验 SHA256，仅放入 `local/`。不需要 GPU；音频模块使用标准库，当前场观察窗口需要 NumPy。已有 DLL 可用 `--dll` 指定完整路径。只预览界面用 `python launch.py --no-audio`。

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

界面标记：**粗圈表示声源中心位于关注圈内，白点表示选入当前非零增益的输出候选**。候选在无声预览时也显示，设备是否开启看底部状态。当前预算 1–16 路、默认 4 路；预算至少 3 且背景比例大于零时通常预留 1 路圈外背景，因此圈内不保证全部发声；正负 solo 也影响选源。关注圈边缘有平滑过渡，不是硬静音边界。表格分别显示范围和候选状态。

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
