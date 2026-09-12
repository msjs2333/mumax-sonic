# 公开案例：标准问题 4 · Field 1

本案例用于 MuMax-Sonic 的数据接入与观察器验证，独立于 CDW。参数依据 [NIST 标准问题 4](https://www.ctcms.nist.gov/~rdm/std4/spec4.html)，网格与松弛初始化采用 [MuMax3 官方示例](https://mumax.github.io/examples.html)。本批只运行 Field 1 的前 1 ns。

| 设置 | 值 |
| --- | --- |
| 全材料矩形、无周期边界 | 500 × 125 × 3 nm |
| 网格 | 128 × 32 × 1 |
| 饱和磁化 / 交换常数 / 阻尼 | 800 kA/m / 13 pJ/m / 0.02 |
| 外场 B | (−24.6, 4.3, 0) mT |
| 动态保存 | 0–1 ns，每 10 ps，共 101 帧；每帧同时 TableSave |

`field1.mx3` 先松弛 `uniform(1, 0.1, 0)`，将其单独保存为 `relaxed.ovf`，再从动态时间零开始保存编号序列。默认编号匹配排除松弛快照。时间从 OVF 头部读取，并与表格交叉核对。

NIST 描述了从饱和场制备平衡 S 态的方法，并要求离散化独立性。本脚本采用官方 MuMax3 示例的简化制备过程；未进行网格收敛、跨求解器参考曲线比较或 Field 2 验证，不能视作完整标准题认证。

## 复现

在仓库根目录执行，要求已安装 MuMax3 且 GPU 可用。运行目录必须不存在；输出留在忽略提交的 `local/` 中。

```powershell
python scripts/run_public_case.py local/sp4-field1
python scripts/validate_public_case.py local/sp4-field1/replay.json local/sp4-field1/simulation.out/table.txt --output local/sp4-field1/validation.json
python scripts/check_replay_incremental.py local/sp4-field1/replay.json --output local/sp4-field1/incremental.json
python launch.py --replay local/sp4-field1/replay.json --recipe activity --aggregation adaptive --source-budget 8 --activity-reference-rad-s 1e10
```

MuMax3 不在 PATH 时，第一条命令增加 `--mumax3 "完整可执行文件路径"`。运行器保留实际脚本、求解器日志、退出码与命令；不计算内容哈希。后两项验证只读取原仿真输出，模拟增量发布使用临时清单。

回放窗口点击“试听 / 重连”出声。`1e10 rad/s` 是固定听觉参考尺度，不改变物理活动量；不要把活动配方的音高当作自旋波频率。拓扑查看可将 `--recipe activity` 改为 `--recipe topology`；开放矩形上的拓扑积分不要求整数，也不能据此自动判定存在斯格明子。101 帧和 1 ns 窗口不满足默认 256 点频带配方，本案例没有验收频带模式。

本批实际结果与后续任务见 [P4a 验证记录](../../docs/p4a-validation.md)。
