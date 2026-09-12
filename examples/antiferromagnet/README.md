# P4b：小型双子晶格 AFM

运行已有 MuMax+ Antiferromagnet 后端，在求解器所属线程顺序采样 A、B，再派生等权 Néel 场和有量纲净磁化。子晶格参数参考[官方 AFM 共振示例](https://github.com/mumax/plus/blob/master/examples/antiferromagnetic-resonance.py)；这里改为 8×8×1、有限阻尼、零外场短时采样，并显式关闭退磁场。它是接入与语义小样，不是完整共振谱实验，也不宣称得到共振频率。

根目录执行，需已安装 MuMax+ 和可用 GPU：

```powershell
python scripts/run_mumaxplus_afm.py local/afm-small
python scripts/check_replay_incremental.py local/afm-small/neel.json --output local/afm-small/incremental.json
python launch.py --replay local/afm-small/neel.json --recipe activity --aggregation adaptive --source-budget 8 --activity-reference-rad-s 1e11
```

输出目录必须不存在；脚本、参数、原始 A/B、Néel OVF 清单、`physical-arrays.npz` 中的有量纲净磁化/原始序参量及检查报告保存在本地，不加入 Git，不计算哈希。可以将回放文件改为 `A.json` 或 `B.json` 独立观察子晶格。窗口点击“试听 / 重连”开启音频。

采样 41 次，相邻 `run` 请求 50 fs；全部使用后端实际返回的物理时间，不用请求间隔乘帧号。各次配对同一时刻，求解器不在音频线程运行。本阶段没有 AFM 多实体切换 UI 或长期在线跟随验收；提供可复现脚本和既有回放入口。

此例 MsA=MsB，Néel 与后端加权定义可直接对照。净磁化初始严格抵消不意味着 n 无效；反之 n 退化也不能强造方向。定义和限制见[物理契约](../../docs/observables.md)，实际检查见[P4b 记录](../../docs/p4b-validation.md)。
