# P4c：真实进动的频带与反相验证

本案例运行真实 MuMax3 求解器，用可解析的独立磁矩进动检查频带链路。脚本依据 [MuMax3 官方 API](https://mumax.github.io/api.html) 的外场、旋磁比和固定步长接口编写；它是本仓库公开的测试案例，不是 µMAG 标准题或传播自旋波实验。

8×4×1 全材料网格，胞元 5 nm；Ms=800 kA/m，交换、各向异性、退磁和阻尼为零。单位磁化初态横向幅值 0.1，纵向为 √0.99。显式设置 γ=2π×28 GHz/T，恒定 z 外场产生 f=γB/(2π) 的等锥角进动。左右反相时只反转初始横向分量，纵向不变。

| 案例 | 频率 | 8–12 GHz 的预期局域均方强度 |
| --- | --- | --- |
| `in_phase` 同相 | 9.375 GHz | 0.01 |
| `opposite_phase` 左右反相 | 9.375 GHz | 0.01，横向空间均值约零 |
| `out_band` 带外 | 18.75 GHz | 接近零，保留实测泄漏 |

RK4 积分步长 0.5 ps，每 10 步保存一次，320 帧覆盖 0–1.595 ns。全部分析使用 OVF 实际物理时间。256 点周期 Hann 窗、z 参考轴，频率分辨率 0.78125 GHz、Nyquist 100 GHz；DFT 周期长 1.28 ns，首末样本跨度 1.275 ns。带内包含 8.59375、9.375、10.15625、10.9375、11.71875 GHz 五个频点。两个载频都落在整数频点上，因此本例不能代表任意离栅频率的泄漏水平。

在仓库根目录运行（MuMax3 不在 PATH 时给运行器增加 `--mumax3 "完整路径"`）：

```powershell
python scripts/run_band_case.py local/band-small
python scripts/validate_band_case.py local/band-small --output local/band-small/validation.json
python launch.py --replay local/band-small/opposite_phase/replay.json --recipe band --band-low-ghz 8 --band-high-ghz 12 --band-window 256 --band-axis 0 0 1 --band-reference 0.01 --aggregation adaptive --source-budget 8
```

输出目录必须不存在；保存每个实际 `.mx3`、求解器日志、命令、退出码和全部 OVF。运行目录在 `local/`，不参与提交，不计算哈希。点击“试听 / 重连”开启声音。前 255 帧显示预热，历史足够后才输出频带强度；暂停后可探听完整窗口。将 `opposite_phase` 换为其余案例可比较，固定听觉参考 0.01，不为带外残余自动增益。

本例先逐体元求频带均方，再空间聚合。先空间平均再分析会抹去反相信号，不能替代当前流程。响度表示所选频带的强度，不是把 GHz 原始波形直接播放，也不能据此确定传播方向。

实际检查和限制见 [P4c 记录](../../docs/p4c-validation.md)。
