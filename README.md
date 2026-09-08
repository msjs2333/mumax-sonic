# MuMax-Sonic

面向微磁与相关磁性连续场仿真的空间可听化工具：用可解释的声音辅助感知状态、运动、频带活动与拓扑结构，并通过可调关注区域分配听觉注意力。

**状态：P1 合成声源原型可运行，人工听觉验收待完成。** 已实现 Windows OpenAL Soft / HRTF、手动关注区域、正负声部和静态探听；物理观察器、真实仿真回放与头追尚未实现。详见 [P1 验证记录](docs/p1-validation.md)。

## 运行 P1

在仓库根目录，使用带 Tk 的 64 位 Python 3.11+：

```powershell
python scripts/install_audio.py
python launch.py
```

安装脚本从官方获取固定版本 OpenAL Soft 1.25.2，校验 SHA256，仅放入 `local/`。不需要 GPU；Python 运行部分使用标准库。已有 DLL 可用 `--dll` 指定完整路径。只预览界面用 `python launch.py --no-audio`。

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
