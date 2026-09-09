# P3b / P3c：增量读取与两个真实后端

2026-09-09。P3b、P3c 的实现与小型真实后端验证已完成；长时间运行、充分重复的开销基准和极限负载属于 P3d，不将本批结果外推成生产级实时保证。

## P3b：增量读取

`IncrementalOVFReader` 沿用原 OVF 校验器，只对新增帧解码/哈希，保留活动所需最近 2 帧或频带所需 N 帧。全部已发布记录的轻量指纹仍保留，检查哈希声明、物理时间、单位、材料域、网格和序号不能被悄悄改写。旧文件 stat 变化会触发重新校验。载入失败不会提交部分缓存。

原始场数组按指定尾部数量保留，不为满足内存限制静默缩短频带窗口；超过保留内存预算即报错。旧帧不反复复制，实时结果只将当前 FieldView 标记 live。诊断导出包含解码总帧数、缓存命中、保留帧数/字节数和源帧数。

当前仍限制每段清单 4096 帧/8 MiB，文件 stat 和清单元数据检查仍随历史增长；长运行需显式换新 segment 和清单，并重新预热频带。增量缓存不是无限长归档，也没有取消原文件不可变约定。总进程峰值内存还包含校验中的新帧、事务旧尾部和观察器工作数组，保留字节数不是进程 RSS。

## P3b：MuMax3 输出桥接

```powershell
python scripts/follow_mumax3.py path/to/run.out local/mumax3-live.json --entity m --segment run-01 --all-material --duration-s 60
python launch.py --follow local/mumax3-live.json --recipe activity --aggregation adaptive --source-budget 8
```

桥接只读 `m*.ovf`（可用 `--pattern` 调整），文件名数字后缀只代表源序号，物理时间必须来自可信 OVF 时间字段。缺少时间、单位或明确 XYZ 标签时拒绝发布，不用文件时间猜测。跨输出阶段用新的 segment 和清单，不自动把求解器重启解释为连续动力学。

每个文件至少经过两次稳定 stat 检查，再完整解析含结束标记与载荷校验；大小不变但未写完的文件继续等待。较早未完成帧会阻止跳到更晚帧；已经存在的序号缺口如实保留，交给观察器的缺帧规则处理。完整前缀可先发布。

清单位于输出目录之外，以独占 `.lock` 管理单写者，先完整写临时文件再原子发布/替换。桥接不会覆盖已有清单来启动新会话；已有源文件被改写、删除或清单被其他写者修改时失败可见。进程崩溃可能留下锁文件，只有确认原发布进程已结束后才手工清理，不自动抢锁。桥接只保留帧元数据，不缓存每帧大数组。

材料域必须显式声明：`--all-material` 仅用于全网格确为材料的情况；否则用 `--mask mask.npy` 的布尔材料掩膜。多层场用 `--z-index` 指定切片。脚本不会从零磁化推断材料边界。单次稳定扫描可用 `--once`，持续运行时用明确的 `--duration-s`，诊断可写 `--report`。

MuMax3 的 Save/AutoSave 与物理时间约定依据 [官方 API](https://mumax.github.io/api.html)，输出格式由现有严格 OVF 读取器解释。

## P3c：MuMax+ 直接采样

```powershell
python launch.py --backend-info
python scripts/run_mumaxplus_live.py --frames 60
python scripts/run_mumaxplus_live.py --headless --frames 20 --interval-s 0.03 --report local/plus-check.json
```

`--backend-info` 只探测可执行文件/可导入包和 API 表面，不创建 World，不把“能导入”视为 GPU 可用。示例脚本则会明确运行一个真实 16×16×1 的均匀进动仿真：5 nm 网格、Ms=8e5 A/m、A=13 pJ/m、alpha=0.02、Bz=0.1 T、关闭退磁场、从 +x 初始化、每 5 ps 采样。它是独立验证案例，不改用户已有项目。

可把采样器嵌入已有求解线程：

```python
from mumax_sonic.sources.mumaxplus import MuMaxPlusSampler
from mumax_sonic.sources.stream import FrameStream
from mumax_sonic.sources.live import LiveConfig

stream = FrameStream(LiveConfig(recipe='activity')).start()
sampler = MuMaxPlusSampler(world, magnet, entity_id='m', segment_id='run-01')
# 在创建 sampler 的同一求解线程中，按研究所需的物理采样间隔执行：
world.timesolver.run(dt_s)
stream.submit(sampler.sample(sequence=0))
```

`sample()` 不推进求解器；创建与调用必须在同一生产线程。读取 eval 前后核对 timesolver.time，不接受采样中途推进。捕获完整 `(3,nz,ny,nx)`，转换为显式 XY 层的 `(ny,nx,3)`；坐标来自后端 meshgrid，保留真实起点，不套用其他后端的半格偏移。材料 mask 来自 geometry 或显式布尔掩膜，多层必须选层。保存主机数据/坐标/掩膜哈希、版本、实体与序参量声明，原始磁化幅值不被采样器归一化。

默认采样给定 magnet 的 magnetization。自定义量必须声明 `quantity_semantics`，并提供完整三分量与正确单位；不会把 AFM 两个子晶格自动相加。支持类的导入检测不代表 AFM/NcAfm 多实体语义已验收，相关泛化仍在 P4。

接口依据 [FieldQuantity](https://mumax.github.io/plus/_api/mumaxplus/mumaxplus.FieldQuantity.html)、[磁化与 meshgrid 教程](https://mumax.github.io/plus/tutorial/magnetization.html)、[World](https://mumax.github.io/plus/_api/mumaxplus/mumaxplus.World.html)，并检查了本机 1.2.1 的 Python 实现。

`FrameStream` 使用有界队列与物理历史；不在 Tk/音频线程调用 GPU。生产过快时替换最旧待处理帧，明确累计 dropped_frames；观察器仍检查源序号/物理时间，不伪造中间帧，频带缺口重新预热。另记 skipped_observation_frames（未单独展示的已接收帧），不与输入丢帧混淆。采集失败、处理超时或关闭均可见，陈旧结果静音。队列和历史配置超出预设主机保留预算时拒绝接收，不自动缩小频带窗口。

GUI 可用 `mumax_sonic.ui.app.run(live_source=stream)`；Tk 在主线程，求解线程拥有 World 和 sampler，参见完整示例。直接采样的观察配方由生产端的 LiveConfig 定义；界面修改回放配方不会重新配置已有求解线程。

## 验证结果

本机：Windows，MuMax3 3.12、MuMax+ 1.2.1，NVIDIA RTX 5060 Laptop 8 GiB。所有生成文件、绝对私有路径、日志和原始报告仅在忽略的 local 目录，未提交。

- 真实 MuMax3：独立 16×16×1、5 nm 网格生成 9 帧，退出码 0，末帧 40 ps；桥接完整发布 9 帧并由增量跟随器计算活动，末区间均值约 1.7255e10 rad/s。该链路验证使用真实已完成输出，不等于持续求解中的极限吞吐基准。
- 真实 MuMax+：20 帧直接采样，末时间约 95 ps（采用求解器浮点实际值），20 次观察、队列丢帧 0，活动均值约 1.7582e10 rad/s。实际 Tk 直接采样窗口最终追上序号 19，valid，最多 8 个声音候选；窗口检查未开启音频设备。
- MuMax+ 一组成对验证：关闭/开启采样的最终 XYZ 最大绝对差为 0。捕获包含主机复制、坐标/材料检查及哈希封装，中位约 0.83 ms；两个 20 步总耗时约 364/352 ms，出现负表观开销，说明预热/顺序噪声不可忽略，不能据此宣称 5% 指标达标。此对照未计 UI 和完整音频链路。
- 同一 20 帧 CPU 发布清单演示，末次读取/观察由 P3a 约 498 ms 降到约 31.7 ms，保留等待/有效/超时静音行为。这是工程样本对比，不是严格统计性能结论。
- 完整测试 `python -m pytest -q -rs`：219 项通过，无跳过；随后直接采样界面参数保护修改的相关 UI 测试 15 项通过。测试覆盖缓存只解码新增帧、历史不可变、内存与物理窗口、时间/网格回退、半写恢复、连续原子发布、owner-thread 捕获、采样时间竞争、单位/坐标错误、队列溢出及失效状态。

P3d 仍需长时间运行、更多网格/频带/后端配置、随机顺序重复成对开销基准、处理积压和恢复、实际设备端到端延迟，以及真正在线求解下的监测干扰测量。
