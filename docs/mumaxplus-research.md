# MuMax+ 能力调研与通用化依据

调研日期：2026-09-08。方法：核对官方 API、教程、例子及方法论文；未安装或执行 MuMax+，未跑 GPU 或音频基准。这里的“支持”指查阅来源中的上游能力，不指 MuMax-Sonic 已实现。

## 1. 版本与证据边界

查阅的[官方文档首页](https://mumax.github.io/plus/)标注 2026-06-08、v1.2.1。方法论文的演示基于 v1.1.0 / paper2025；主分支可能继续变化。适配时必须记录实际安装版本与构建标识，检测具体能力，不能把某个版本的文档列表视作所有安装都支持的承诺。[官方仓库](https://github.com/mumax/plus)

MuMax+ 是 C++/CUDA 求解器和 Python 接口；同一 World 中可以有多个磁体。论文覆盖更丰富的磁序和多物理建模，适合把 Sonic 的输入由“单张磁化图”扩展为“带实体身份的物理场集合”。[方法论文](https://www.nature.com/articles/s41524-025-01893-y)

## 2. 能力与代表性研究场景

“场景”是用于设计覆盖面的代表性任务，未做使用频率统计。下面的可听化均是本项目设计建议。

| 上游能力及来源 | 代表性场景 | Sonic 应观察什么 | 设计影响 |
| --- | --- | --- | --- |
| FM，交换、退磁、各向异性、Zhang–Li/Slonczewski 力矩：[Ferromagnet](https://mumax.github.io/plus/_api/mumaxplus/mumaxplus.Ferromagnet.html) | 磁化反转、电流驱动、振荡器 | 活动、方向、局部运动和窄带包络 | 驱动方式与观察模式分开 |
| 可设分量的 DMI 张量：[DmiTensor](https://mumax.github.io/plus/_api/mumaxplus/mumaxplus.DmiTensor.html) | 手性纹理、不同壁类型 | 局部旋转、纹理与拓扑 | 不硬编码某种 DMI 的手性或壁芯轴 |
| 共线 AFM / 亚铁磁，双子晶格：[Antiferromagnet](https://mumax.github.io/plus/_api/mumaxplus/mumaxplus.Antiferromagnet.html) | Néel 序动力学、补偿体系 | 子晶格、序参量、倾斜角、频带 | 总磁化接近零不能等价于安静 |
| 三子晶格非共线磁序：[NcAfm](https://mumax.github.io/plus/_api/mumaxplus/mumaxplus.NcAfm.html) | 非共线态和序参量切换 | 子晶格夹角、八极矩向量等 | 不把任意磁序压成一个单位 m |
| 双子晶格、各向异性交换：[Altermagnet](https://mumax.github.io/plus/_api/mumaxplus/mumaxplus.Altermagnet.html) | 交错磁体畴壁 | 子晶格/Néel 纹理及方向相关动力学 | 提供实体、序参量和局部坐标选择 |
| 弹性动力学、位移、速度、应变/应力及相关能量：[Magnet](https://mumax.github.io/plus/_api/mumaxplus/mumaxplus.Magnet.html) | 磁弹耦合、磁声波 | 两类场的频带强度与相位关系 | 单位和物理量类型必须明确；超声场仍需可听化 |
| 温度和热涨落：[Temperature](https://mumax.github.io/plus/tutorial/langevin.html) | 热噪声、热激活过程 | 背景统计、超出噪声的变化 | 活动基线、采样条件和误报需要标定 |
| 电势 Poisson 求解、含 AMR 的电导张量：[Ferromagnet 电输运接口](https://mumax.github.io/plus/_api/mumaxplus/mumaxplus.Ferromagnet.html#mumaxplus.Ferromagnet.electrical_potential) | 磁电读出 | 指定探针读数或其变化 | 后续用标量探针配方，不承诺任意输运模型 |

官方现成教程/示例包括[标准问题 4 的磁化反转](https://mumax.github.io/plus/tutorial/stdp4.html)、[磁弹波](https://mumax.github.io/plus/examples/magnetoelastics.html)、[声表面波驱动畴壁](https://mumax.github.io/plus/examples/DW_SAW.html)、[示例索引中的标准问题 2、Voronoi 和交错磁体畴壁](https://mumax.github.io/plus/examples.html)。这些可用于验证通用性，而无需把私人项目当成唯一基准。

[配置工具](https://mumax.github.io/plus/_api/util/mumaxplus.util.config.html)包含涡旋和 Néel/Bloch skyrmion 初态构造。存在初态生成器不证明给定材料中的稳定性，也不等于有完整拓扑识别器。初始化参数中的 charge/polarization 必须核对定义，不能直接替代 Sonic 的 Q 约定。

## 3. 已核对的数据接口

- [FieldQuantity](https://mumax.github.io/plus/_api/mumaxplus/mumaxplus.FieldQuantity.html)提供 name、unit、ncomp、grid、shape、eval、average 和 OVF 保存。NumPy 场采用分量优先布局，OVF 数据布局不同；适配层统一到显式维度与物理轴，不能靠数组 shape 猜测方向。
- [TimeSolver](https://mumax.github.io/plus/_api/mumaxplus/mumaxplus.TimeSolver.html)支持 run、steps、run_while，以及在指定 timepoints 收集量的 solve；solve 支持不保留输出字典。流式监听仍要验证量求值、回调和队列生命周期，不把 solve 宣称为开箱即用的零拷贝遥测。
- [AFM 的 neel_vector](https://mumax.github.io/plus/_api/mumaxplus/mumaxplus.Antiferromagnet.html#mumaxplus.Antiferromagnet.neel_vector)定义为 (Ms1*m1−Ms2*m2)/(Ms1+Ms2)。它不是自动归一化的单位方向。拓扑计算若选它，必须记录加权定义、模长和无效区；也允许显式选择 (m1−m2)/2 的另一配方。
- [NcAfm 的 octupole_vector](https://mumax.github.io/plus/_api/mumaxplus/mumaxplus.NcAfm.html#mumaxplus.NcAfm.octupole_vector)是特定定义的序参量；其物理态空间不能由一个三分量输出形式推断成普通 FM 的 S²。

拟优先在原脚本已有采样处接入，输出不可变样本；音频线程不推进求解器、不调用 GPU。监测器不擅自改变阻尼、时间步、外场或输出协议。内部自适应积分步、观测采样间隔和音频采样率是不同概念。

## 4. 未确认、不可预设的能力

本次未验证通用 GPU 零拷贝、任意观察器 GPU 插件、现成拓扑对象追踪、Windows AirPods 姿态通道或任意进程实时挂接。它们不能成为 MVP 依赖。宿主 eval 后裁剪不会消除前面的完整传输；后续若做 GPU 侧滤波，需以源代码接口与基准证明收益。

relax/minimize 得到的是优化过程；不能把优化迭代中的方向变化速度解释成真实 LLG 动力学。此类输入使用 convergence 配方，保留步骤/状态标记，不默认开启物理频谱。

## 5. 拓扑与音频依据

- [Kim & Mulkers：晶格拓扑荷计算](https://arxiv.org/abs/2006.13336)：支持把连续导数与离散几何估计交叉检查，尤其注意离散化和热扰动影响；不意味着任意网格都能可靠分类。
- [Theory of antiskyrmions in magnets](https://www.nature.com/articles/ncomms10542)：拓扑荷、绕数、背景/核心方向和 helicity 有不同角色，不应把正 Q 固定命名为 skyrmion、负 Q 固定命名为 antiskyrmion。
- [OpenAL Soft](https://github.com/kcat/openal-soft)：Windows 原型的首选渲染候选，具备 HRTF/空间声源及 C 接口。尚未在本机试听与验证绑定。
- [Windows Spatial Audio](https://learn.microsoft.com/en-us/windows/win32/coreaudio/render-spatial-sound-using-spatial-audio-objects)：原生候选，需验证设备和动态声源能力。
- [Apple Core Motion](https://developer.apple.com/videos/play/wwdc2023/10179/)：公开耳机姿态接口面向 Apple 宿主平台，不据此承诺 Windows 直接读取。

## 6. 泛化结论

泛化单位应是“实体 × 物理量/序参量 × 观测算子 × 听觉配方”。FM、AFM、磁弹耦合可以共享时间、ROI 和音频系统；物理定义及有效条件仍由配方声明。四个核心模式先覆盖 FM/二维向量纹理，AFM 的实体和序参量结构从首个数据契约保留，其他能力按验证结果逐步开放。
