# P4b：双子晶格语义与真实 AFM 接入

2026-09-12。本批完成有序双子晶格配对模块、物理定义、独立合成测试，以及一个小型真实 MuMax+ 案例。P3e 仍待启动，P4 的全部泛化任务尚未完成。

## 实现与使用

[sublattices.py](../src/mumax_sonic/sublattices.py) 提供 `combine_sublattices`，不推进求解器，不介入音频线程。输入 A/B 的完整 XYZ、明确的 MsA/MsB 和配对 ID，返回原子晶格、原始等权 Néel 差、有量纲净磁化及可用于既有方向观察器的 FieldFrame。

公式和退化、符号约定见 [物理契约](observables.md)。配对要求同网格、同掩膜、同物理时间和同帧身份；保持材料域，不将缺失数据当零，不增加哈希。派生实体 ID 同时区分子晶格顺序与所选观察量。

```python
from mumax_sonic.sublattices import combine_sublattices
from mumax_sonic.field_pipeline import observe_field

pair = combine_sublattices(frame_a, frame_b, pair_id="sample-afm",
                          msat_a_A_m=200e3, msat_b_A_m=200e3)
view = observe_field(pair.neel, "activity", previous=previous_pair.neel)
# pair.net_A_m 保留 A/m；pair.neel_raw 保留无量纲模长和真实零值。
```

第一帧无前驱时传 `previous=None`。改变权重定义、阈值或坐标约定应开启新 segment。当前输出可进入既有 Activity/Topology 观察器，未增加面向用户的多实体切换面板。

## 合成回归

[test_sublattices.py](../tests/test_sublattices.py) 包含 25 项检查：净磁化抵消时的非零 Néel 活动、实际 Δt、等权/不等权/逐点权重、三分量原始模长、方向退化、缺失配对、材料内 NaN、掩膜/时间/身份错配。二维合成纹理验证交换 A/B 时 Q 变号、Qabs 不变，双帧活动不变；有序实体交换或 n/net 切换不能跨身份做角差。

完整回归 `python -m pytest -q`：322 项通过。合成拓扑纹理不是本次真实 AFM 求解器产生的拓扑态。

## 真实 MuMax+ 检查

复现与回放命令见 [AFM 案例](../examples/antiferromagnet/README.md)。最终运行在 `local/p4b-afm-final-20260912/`，脚本、参数、41 帧原始 A/B、派生 n、原始物理数组和报告保存在本地，不参与提交。MuMax+ 1.2.1，8×8×1 网格，两子晶格 Ms 均为 200 kA/m，零外场、关闭退磁、有限阻尼；以稍偏离易轴的反平行态开始真实动力学。

| 检查 | 实测 |
| --- | --- |
| 配对采样 | 41 帧，初始 1 帧预热、40 对完整有效活动 |
| 实际末帧时间 | 1.96250047 ps；每步请求 50 fs，计算使用后端实际时间，未假装累计为 2 ps |
| 初始净磁化 | 严格为 0 A/m |
| 净磁化 vs 后端 full_magnetization | 最大分量差 0.0078011 A/m（对子晶格 200000 A/m 的单精度归约）；绝对容差 0.1 A/m、相对 2×10⁻⁶ |
| 等权 n vs 后端 neel_vector | 最大差 1.63×10⁻⁷；本例等 Ms 才适用此对照 |
| 平均 Néel 活动范围 | 5.46×10⁹–1.88×10¹⁰ rad/s |
| 活动 → 声音参数 | 40 帧，固定参考 10¹¹ rad/s，自适应聚合，最多 8 路 |
| 导出回放的活动差 | 0 rad/s |
| 模拟逐帧发布 | 41 帧与离线读取完全一致，解码 41、跳过 0，保留 2 帧 |

最终运行约 10.16 秒，包含初始化、求解、采样、检查和保存；不作为监测开销或实时延迟指标。真实输出同时用于增量回放检查，但没有验收求解器写入与播放同时运行的长期链路。

本批不验证网格收敛、AFM 共振频率、拓扑态稳定性、实际设备/HRTF 或人耳可辨性。MuMax+ 后端输出属于同一求解器内部对照，不能替代独立解析解或跨求解器验证。热噪声、非共线三子晶格、异网格和部分缺子晶格体系仍需后续配方。

## 下一步

建议进入 **P4c：真实频带小型案例**。选一个采样足够的公开自旋波/进动输入，冻结物理时间窗、分量基底、带内/带外参考与固定听觉尺度，验证“逐点频带强度再聚合”及反相不相消。不立即增加耦合物理或读取架构；内存不足时再将相应任务连接到 P3e-3。
