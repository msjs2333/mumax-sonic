# P2d：OVF 序列与真实数据验证

日期：2026-09-09。已接入 OVF 2.0 规则网格的 Text、Binary 4 和 Binary 8，支持清单导入、二维层选择、显式材料掩膜、完整 XYZ、真实时间与来源哈希。真实 MuMax+ 小型回放和 MuMax3 单文件解析已检查；本阶段没有启动求解器或修改外部仿真。

## 快速使用

可复现的独立解析案例，不依赖私人项目：

```powershell
python scripts/make_ovf_demo.py local/ovf-demo --frames 320
python scripts/make_ovf_manifest.py local/ovf-demo local/ovf-demo.json --limit 320 --origin synthetic --time-kind dynamics --all-material
python launch.py --replay local/ovf-demo.json --recipe band
python launch.py --inspect-field local/ovf-demo.json --recipe band --report local/ovf-band.json
```

生成器写入新目录，清单生成器不覆盖文件。默认清单最多选前 32 个匹配文件；`--limit` 可提高至 4096。它报告匹配数、选入数和选入范围内的序号缺口，不宣称原始仿真完整。`m000000.ovf` 这类数字后缀作为源序号保留，绝不用作物理时间。多个命名检查点或非帧号命名需手工清单，不能按字典序混入同一阶段。

对真实 MuMax3 序列，将目录换成实际输出目录，声明 `--origin simulation --time-kind dynamics`。只有确知全网格为材料时使用 `--all-material`；有几何缺口时提供 `--mask-file material.npy`，不能由磁化零值自动推断。多层输入必须 `--z-index 0` 等显式选层。

窗口“加载场”可选 NPZ 或 OVF JSON 清单，回放配方仍为拓扑/活动/频带。窗口显示 OVF、来源声明、时间类型、所选层与源帧号；诊断保留输入单位、分量标签、网格、时间来源和文件哈希。源数据中的序号缺口继续触发活动/频带的预热规则。

## 清单契约

最小示例（SHA256 需替换为实际值）：

```json
{
  "schema_version": 1,
  "entity_id": "m",
  "segment_id": "dynamic-stage-1",
  "origin": "simulation",
  "time_kind": "dynamics",
  "quantity": "magnetization_direction",
  "value_unit": "1",
  "components": ["x", "y", "z"],
  "z_index": 0,
  "mask": "all",
  "frames": [
    {"file": "m000000.ovf", "sha256": "实际文件哈希", "sequence": 0},
    {"file": "m000002.ovf", "sha256": "实际文件哈希", "sequence": 2}
  ]
}
```

文件路径相对清单目录解析，也可显式使用本地绝对路径；清单生成器使用绝对路径以便读取外部输出，迁移后须更新。内容身份由 SHA256 绑定，路径不证明物理有效性。清单只读取文件，不执行命令或下载资源；不要把不可信清单当作已审阅的数据选择。

- `origin` 是用户对 simulation/synthetic/unknown 的声明，不是程序认证。`time_kind` 明确区分 dynamics/static/relaxation；一个清单只容纳一个实体、阶段及固定网格，阶段重置应拆成另一清单。
- `quantity` 支持 `magnetization_direction` 或 `order_parameter_direction`。单位方向接受 `1`，磁化也接受 `A/m`，后续方向观察器归一化。原始模长保留，但目前不输出有单位的磁化功率或磁能。
- `components` 声明文件实际存储次序，导入时重排为 XYZ；可识别的 x/y/z 标签不得与声明冲突。单位与三分量标签也须跨帧一致，不自动把任意三分量力/位移场解释成磁化。
- `mask` 必须明确为 `"all"`，或 `{"file":"material.npy","sha256":"..."}`。NPY 须为布尔二维选中层或三维完整网格，格式 1/2；不能用数值标签代替。零矢量和 NaN 保留，交给观察器判定无效，缺数据不变成零活动。
- OVF `xbase/ybase/zbase` 是第一个采样中心，转换为 SI；网格数组顺序为 z/y/x/component，x 最快。多层只取声明层，层位置包含 zbase 与步长；不平均或合并拓扑荷。使用 OVF 实际记录的坐标，不猜测原求解器未导出的世界原点。
- 目前仅支持单段、规则网格、三分量 OVF2；不支持 OVF1、不规则网格、标量或直接三维拓扑。XY 至少 3×3。单文件及其解码矢量各不超过 64 MiB，序列输入和选中层数据各不超过 256 MiB。

## 物理时间的来源

正常 MuMax3 的 `Desc: Total simulation time: ... s` 可读取为秒。若清单帧也提供 `time_s`，必须与头信息一致，否则拒绝。时间严格递增；不自动排序时间、不插值、不按文件修改时间或播放速度重建动力学。

本次真实 MuMax+ 样本的头包含重复、无单位的 `Total simulation time: 0`。这类记录保留为 `header_time_hint`，`header_time_s` 为 null，不能直接作为物理时间。清单生成器会拒绝自动生成这类回放，要求手工填写从可靠来源取得的逐帧 `time_s`。

手工清单可增加 `time_evidence`：

```json
{
  "time_evidence": {
    "file": "original-frame-manifest.json",
    "sha256": "原时间证据文件的实际哈希",
    "description": "每个帧记录的 solver elapsed_s，单位秒"
  }
}
```

将此字段合入完整清单，并在各帧写入 `time_s`。导入会校验时间证据文件哈希，记录描述；它不会自动理解任意外部清单的字段意义。证据与帧的对应仍需按源保存代码核对。真实案例使用原清单的 index 和 elapsed_s，未从含时间数字的文件名推导序号或时间。

格式依据：[NIST OVF 2.0](https://math.nist.gov/oommf/doc/userguide21a0/userguide/OVF_2.0_format.html)、[MuMax3 OVF 写入源码](https://github.com/mumax/3/blob/master/oommf/ovf2.go)、[MuMax+ FieldQuantity 输出约定](https://mumax.github.io/plus/_api/mumaxplus/mumaxplus.FieldQuantity.html)。无单位占位头的结论仅来自本次具体文件，不外推到所有版本。

## 验证证据

**自动验证**：`python -m pytest -q` → **154 passed**。覆盖非方形多层 XYZ、Text/Binary4/Binary8 一致性、SI/采样中心、注释/大小写、完整尾部与字节序、无单位时间、缺帧与跳转、哈希冲突、单位/标签冲突、掩膜、NPY 分配前边界、重复 JSON 字段、清单生成与不覆盖、NPZ 来源往返和真实 Tk 控件路径。

**独立解析案例**：实际生成并读取 320 帧 Binary4 OVF；逐帧清单校验和频带 JSON 导出通过。该案例与 P2c 同为 10/30 GHz 分区振荡，不是求解器运行。二进制量化在 float32 精度范围内。

**打包与格式**：`pip wheel . --no-deps --no-build-isolation` 通过，新模块随包分发，无新增依赖；`git diff --check` 通过。

**真实 MuMax+ 回放**：只读用户备份中的两帧 508×508×1 磁化，约 5.91 MiB，总时间分别 50/100 ps。核对保存代码、全矩形材料构造、原帧清单哈希及原清单内容签名后导入；导入耗时本次约 45 ms，仅作单次记录。原始 XYZ 与独立二进制读取逐值相等；50 ps 区间活动均值约 `9.26508e6 rad/s`，与独立 acos 差分一致，覆盖 100%。

同一真实帧的周期边界中心差分 Q 为 `0.9947502578`，原项目记录 `0.9947501448`，绝对差约 `1.13e-7`。这是同类数值估计的交叉核对，不是普适拓扑分类证明。该对照显式选择 periodic；窗口当前拓扑配方默认 open，不把二者当作同一积分设置。两帧不足以做频带估计，程序正确报告 warming_up。

**真实 MuMax3 格式验证**：另读一帧 1500×500×1、2 nm 网格、Binary4、时间 5 ps 的实际输出，约 8.58 MiB。其源代码有缺口几何，文件含 1250 个零矢量；仅验证格式与元数据，未重建材料掩膜，也未将该帧报告为完整活动/拓扑验证。

**窗口**：用真实 MuMax+ 清单检查活动显示、完整 XYZ 图例、源帧号、时间与控件布局；单窗口截图通过。私有文件路径、原始数据、哈希与详细诊断均留在 `local/`，不进入 Git；通用实现不含私人项目参数或路径。

本阶段未重跑实际音频设备检查、未进行人工耳机试听、未验证真实自旋波频带、公共独立仿真、大网格持续回放或自动实时跟随。音频模块未修改，不能将 P2c 设备结果写成 P2d 真实数据试听通过。

## 下一步

P2e 先使声源预算可配置，并报告选入与省略的物理贡献、总体覆盖；对当前固定分块作基线比较。真实数据已经能进入流程，之后再根据密集场景和性能证据引入自适应聚合与背景汇总。另补公共独立案例和具有足够采样历史的真实频带验证；完整壁追踪与更舒适音色仍是后续工作。
