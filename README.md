# MuMax-Sonic

把微磁仿真的空间分布与演化转换成可解释的空间声音。支持 MuMax3 已有 OVF 输出的自动发现与只读接入、MuMax+ 生产线程采样，以及离线场回放。

- **监听已有仿真**：发现本机 MuMax3 进程及有限目录中的输出，选择任务后持续读取；可重新扫描、切换和重新连接。
- **观察物理场**：二维切片的拓扑、相邻帧角活动和时间窗频带；保留物理时间、单位、材料掩膜和缺帧状态。
- **空间试听**：可拖动关注区域、控制背景和声源数量；OpenAL Soft HRTF。默认无声，点击“试听 / 重连”才打开音频。

这是研究原型。数值验证、实际设备测试与人工听觉验收分开记录；100 ms 端到端延迟和普适纹理分类不是当前保证。

## 安装

使用带 Tk 的 64 位 Python 3.11+，在仓库根目录执行：

```powershell
python -m pip install -e ".[test]"
python launch.py --attach --no-audio
```

需要耳机试听时，在 Windows 上安装固定版本的官方 OpenAL Soft（下载到本仓库 `local/`，校验 SHA256），再去掉 `--no-audio`：

```powershell
python scripts/install_audio.py
python launch.py --attach
```

已有 OpenAL Soft DLL 可用 `--dll 路径` 指定。求解器另行安装；本项目不安装或打包 MuMax3/MuMax+ 引擎。

## 接入正在运行的仿真

仿真可以先于 MuMax-Sonic 启动。运行一条命令：

```powershell
python launch.py --attach --recipe activity
```

窗口每 5 秒发现本机 MuMax3 进程及当前目录的直接 `.out` 子目录；点击候选，确认材料域，选择“接入 / 切换 / 重新连接”。主窗口的“接入仿真”可再次打开列表。程序不向求解器注入代码、不改运行脚本、不停止或重启任务。

若进程权限、WSL/容器边界或相对路径阻止发现，可添加搜索根目录，或直接给出现有输出目录：

```powershell
python launch.py --attach --search-root D:/simulations --recipe activity
python launch.py --attach D:/simulations/run.out --all-material --recipe activity
python launch.py --discover --search-root D:/simulations
```

`--discover` 输出候选和扫描诊断 JSON，不打开窗口或音频。扫描不递归遍历磁盘；最多检查 256 个目录、每目录 4096 个条目，达到上限会提示。PID 是进程证据，存在输出目录或旧帧并不证明任务仍在运行。

材料不占满网格时以 `--mask material.npy` 替代 `--all-material`，使用布尔 NPY 掩膜；多层场增加 `--z-index 0` 等显式切片。默认选取 `m[0-9]*.ovf`，其他数字序列用 `--pattern`；静态或松弛输出分别使用 `--time-kind static` / `relaxation`，不将其解释为动力学速率。

接入从最近 2 帧（频带为所选窗口）开始，避免读取全部历史场。桥接清单在系统临时目录中自动管理，源目录只读；每 4096 帧轮换观察段，可能短暂重新预热。它是实时观察，不是完整归档。

半写帧会等待完成；目录失联会显示错误并保持重试，同一输出序列恢复追加后自动继续。超过 `--live-stale-s`（默认 2 秒）没有新物理帧时显示 `stale` 并静音；慢速生产可增大该值。求解器重启、同名文件重写或物理时间重置后，点击重新连接以建立新观察段。切换任务默认再次静音，需主动开启试听。

## 支持范围

| 输入 | 已有任务接入条件 |
| --- | --- |
| MuMax3 | 可读的数字序列 OVF，含完整 XYZ 磁化、单位和真实物理时间；明确材料域与切片 |
| WSL / 远程任务 | 通过本机可读输出目录接入；本机进程发现不跨 WSL、容器或远程主机 |
| MuMax+ | 已在生产线程设置 `MuMaxPlusSampler` / `FrameStream`，或已有符合上述约定的 OVF 输出；不支持从任意 Python 进程中提取私有 GPU 状态 |
| 无 OVF / 无采样接口 | 无法从进程存在本身恢复磁化场。当前任务保持原样；下次运行前配置一次所需的 `AutoSave(m, ...)` 或采样接口 |

不从截图、文件修改时间或编号猜造物理数据，不从零矢量猜材料域。频带需要连续动态时间窗；缺帧、无效输入或采样不足会显示等待/无效。附加窗口使用主窗口当前配方；更改配方、频带或最大间隔会建立新的观察会话并静音，需再次点击试听。

已有发布清单仍可用 `python launch.py --follow path/to/live.json`；两终端桥接方式、MuMax+ 采样示例和离线回放见[既有用法](docs/usage-and-development.md)与[后端接入说明](docs/p3bc-validation.md)。

## 验证与研究记录

```powershell
python -m pytest -q
python launch.py --backend-info
python launch.py --field-demo activity_rotation --no-audio
```

- [自动接入验证](docs/attachment-validation.md)：本次功能、测试和已知限制。
- [既有用法与阶段记录](docs/usage-and-development.md)：原首页的完整演示命令、性能记录和研究链接。
- [概念与架构](docs/concept.md)、[物理观测约定](docs/observables.md)、[实施计划](docs/implementation-plan.md)。
- [P4 收尾](docs/p4-closeout.md)、[试听材料](examples/p4_listening/README.md)：数值结果不等于人工听觉验收。
- [代理协作约定](AGENTS.md)。

## 许可

项目采用 [GNU GPL-3.0-only](LICENSE)。MuMax3、MuMax+、OpenAL Soft 及其他依赖保留各自版权与许可，详见[第三方说明](THIRD_PARTY_NOTICES.md)。本项目不把这些第三方组件改授权为自身作品。
