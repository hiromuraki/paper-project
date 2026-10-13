# UI-PRMD 数据集说明与用法

> 当前工具为独立的 `scripts/convert_ui_prmd.py`，使用方法见 [UI-PRMD-转换-v2.md](UI-PRMD-转换-v2.md)。旧体检、生成和打包脚本已移除。以下数据规模和质量记录来自历史本地扫描。

本文基于对 `dataset-raw/UI-PRMD/` 的全量扫描与逐文件实测，以及原论文
[Vakanski et al., *Data* 3(1):2, 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC5773117/)
（[doi:10.3390/data3010002](https://doi.org/10.3390/data3010002)）写成。
凡标“实测”的数字为历史本地扫描结果，旧体检命令已停用。

## 1. 这是什么

University of Idaho 的物理康复动作数据集。10 名健康受试者（s01–s10）各做 10 种
康复动作（m01–m10）各 10 次，同时用 **Vicon 光学动捕（100 Hz）** 与
**Microsoft Kinect（30 Hz）** 记录，另有一份**故意做错**的版本。

| m | 动作 | 本项目 action_id | 本项目是否直接可用 |
|---|---|---|---|
| m01 | Deep squat 深蹲 | `squat` | ✅ 与 demo 的 `squat` 任务同名同义 |
| m02 | Hurdle step | `hurdle_step` | 需改动作定义 |
| m03 | Inline lunge | `inline_lunge` | 需改动作定义 |
| m04 | Side lunge | `side_lunge` | 需改动作定义 |
| m05 | Sit to stand | `sit_to_stand` | 需改动作定义 |
| m06 | Standing active straight leg raise | `straight_leg_raise` | 需改动作定义 |
| m07 | Standing shoulder abduction | `shoulder_abduction` | 需改动作定义（单臂） |
| m08 | Standing shoulder extension | `shoulder_extension` | 需改动作定义（单臂） |
| m09 | Standing shoulder internal-external rotation | `shoulder_internal_external_rotation` | 需改动作定义（单臂） |
| m10 | Standing shoulder scaption | `shoulder_scaption` | 需改动作定义（单臂） |

**标签只来自目录**：`Movements*` = 正确（label 1），`Incorrect Movements*` = 非最优
（label 0）。论文明确说明错误样本**没有**标注具体错误类型，也没有严重程度，而且
受试者只是“随意地以次优方式完成”，并非模拟某种具体损伤。m01 论文给出的非最优判据是：
躯干未能保持竖直、下蹲未过水平、膝外翻（knee valgus）、躯干前屈超过 30°。

每个动作的正确/错误各 100 次重复（m01 就是 100 正 + 100 负），按受试者划分后是
60/20/20 人级别的小数据集，适合做流程验证，不足以训练一个稳健的通用判别器。

## 2. 目录与文件

```text
dataset-raw/UI-PRMD/
  Movements/{Kinect,Vicon}/{Positions,Angles}/            # 正确，整体（每人每动作 1 个文件）
  Segmented Movements/{Kinect,Vicon}/{Positions,Angles}/  # 正确，切成 10 段
  Incorrect Movements/...                                 # 错误，整体
  Incorrect Segmented Movements/...                       # 错误，切成 10 段
  *.zip                                                   # 同样是这四份，解压后可直接删
```

文件命名：

- 整体：`m01_s01_positions.txt` / `m01_s01_angles.txt`
- 分片：`m01_s01_e03_positions.txt`（第 3 次重复）
- 错误：文件名末尾加 `_inc`

### 实测规模

| 划分 | Kinect 文件 | Vicon 文件 | 说明 |
|---|---|---|---|
| Movements | 100 | **99** | Vicon 缺 `m03_s03`（论文说明当时 marker 采集失败，Kinect 侧完整） |
| Segmented Movements | 1000 | **990** | 每个 trial 恰好 10 段，无缺段 |
| Incorrect Movements | 100 | **99** | 同上 |
| Incorrect Segmented Movements | 1000 | **990** | 每 trial 10 段 |

每份数据都有 `Positions` 和 `Angles` 两套，两者行数完全一致。

### 帧数与时长（实测）

深蹲正确数据 m01，整体文件帧数（Vicon / Kinect）：`1620–4176`（均值 2562.3）/ `525–1370`（均值 797.7），
比值 3.09；分片文件 Vicon 均值 245.0 帧、Kinect 均值 74.6 帧，比值 3.28 —— 与论文的
100 Hz : 30 Hz ≈ 3.33 一致，可用作交叉校验。即**一次动作约 2–3 秒**。

## 3. 文件格式

纯文本、**无表头**、每行一帧、列数恒为 `关节数 × 3`：Kinect 66 列、Vicon 117 列。

> **分隔符不统一（实测）**：`Movements/` 与 `Incorrect Movements/` 下的整体文件用
> **空格**分隔，`Segmented Movements/` 与 `Incorrect Segmented Movements/` 下的分片
> 文件用**逗号**分隔。论文正文只写了“逗号分隔”，照论文写解析器会在整体文件上报错。
> `scripts/convert_ui_prmd.py` 两种都接受。文件为 CRLF 换行。

### 坐标系与单位

| | Kinect | Vicon |
|---|---|---|
| 帧率 | 30 Hz（实测帧数比 = 3.33） | 100 Hz |
| 列数 | 22 关节 × 3 = 66 | 39 标记点 × 3 = 117 |
| 位置含义 | 关节 1 绝对，其余 21 个按论文“相对父关节” | 39 个标记点**全部绝对** |
| 米/单位 | 未标注；实测数值量级明显不是毫米（见 §5） | 毫米 |
| 竖直轴 | 论文未写。实测根部三轴都随深蹲变化，无法唯一确定 | Z 轴（足跟≈60–90 mm，头≈1300–1600 mm） |
| 角度 | 每关节 YXZ 欧拉角，单位度 | 同 |

Vicon 的 39 个标记点顺序（论文 Table 2，`Positions` 与 `Angles` 同序）：
`LFHD RFHD LBHD RBHD C7 T10 CLAV STRN RBAK LSHO LUPA LELB LFRM LWRA LWRB LFIN
RSHO RUPA RELB RFRM RWRA RWRB RFIN LASI RASI LPSI RPSI LTHI LKNE LTIB LANK LHEE LTOE
RTHI RKNE RTIB RANK RHEE RTOE`

Kinect 的 22 个关节（论文 Table 2）：
`Waist Spine Chest Neck Head HeadTip LeftCollar LeftUpperArm LeftForearm LeftHand
RightCollar RightUpperArm RightForearm RightHand
LeftUpperLeg LeftLowerLeg LeftFoot LeftToes
RightUpperLeg RightLowerLeg RightFoot RightToes`

## 4. 关键坑（都是实测结论）

1. **Vicon 缺 `m03_s03`**（190 个文件中的 2 个）。做全量对齐时不能假设 10×10 全齐。
2. **Vicon 有零值标记点**：某些分片里某个标记点整段恒为 `[0,0,0]`，是**采集丢失**而不是
   真实的静止。实测出现位置包括 `m02_s05_e01`（STRN）、`m03_s01_e01`（RBHD）、
   `m03_s01_e04–e09`（LTOE）等；正确数据里出现过 `LASI/LWRB/LTOE/STRN/RBHD`。
   直接当坐标用会被理解成“标记点在原点”，必须按缺失处理。
3. **存在损坏/截断分片**：`Incorrect Segmented Movements/Vicon/Positions/m10_s05_e09_positions_inc.txt`
   只有 **1 帧**（同一 trial 整体文件 2156 帧，其余 9 段合计 2150 帧），属异常样本应剔除。
4. **整体文件帧数 > 该 trial 分片帧数之和**（实测差 67–228 帧）。分片只覆盖每次重复本身，
   重复之间的过渡帧没进分片；所以“整体文件 = 10 个分片拼接”是错的。反过来，分片总数
   990 = 99 trial × 10 段，说明是留白而非丢段。
5. **正确与错误的 `e01…e10` 编号不对应同一次物理重复**。两侧都是各自 10 次，编号只是序号。
   好在每一段自带标签，不需要配对。
6. **Kinect 的尺度对不上（重要）**。论文说位置是毫米，但实测 Kinect 位置列的量级约为人体
   真实尺寸的 1/10–1/15，且**每段的比例还不一致**；按论文假设把这些列当“相对父关节偏移”
   累加之后，得到的骨架高度只有约 140 mm，髋到膝的距离在同一段里还会变化 20% 以上，
   躯干、四肢长度比例都不成立。用角度+正向运动学重建也不能解决。**结论：Kinect 位置
   不能按字面直接拿来当物理坐标用，必须先做交叉验证**（用同 trial 的 Vicon 校准尺度与
   关节对应，或直接只用 Vicon）。本仓库的工具**不对 Kinect 做转换**，避免产出看似可用的错数据。
7. **Kinect 有整段静止的关节**：正确分片里 `Neck` 有 132 段恒定、`LeftToes/RightToes`
   各 6–7 段恒定，`Incorrect Segmented Movements/Kinect` 里 `Neck` 有 32 段恒定。
   这些段缺少该关节的运动信息。
8. **角度有 ±180° 回绕修正**：论文说明只做过这一项预处理（角度限制在 (−180°, 180°)）。
   直接用角度做特征时要自己处理跳变。

## 5. 怎么用

当前转换器为单文件 `scripts/convert_ui_prmd.py`，无默认转换参数：

```powershell
uv run scripts/convert_ui_prmd.py --help
uv run scripts/convert_ui_prmd.py --root dataset-raw/UI-PRMD --out dataset-ui-prmd-new --movement all --segmented --view front --max-gap 10 --fps 30
```

新版输出为米单位、Y-up H36M-17 关节和对应的派生 2D 投影，并记录缺失点修复与跳过原因，详见 [UI-PRMD-转换-v2.md](UI-PRMD-转换-v2.md)。输出目录必须为空，不会覆盖以前的结果。

- 标签仅区分正确 / 非最优，不提供具体错误类别。
- 2D 来自 3D 投影，不是真实相机观测。
- 不混入 Kinect，不将正确 / 错误的同编号重复当作同一次动作。
- 不同时使用整体与分片；按受试者划分，避免同源泄漏。

## 6. 相关文件

- 转换工具：[convert_ui_prmd.py](../scripts/convert_ui_prmd.py)
- 数据格式约定：[dataset/README.md](dataset/README.md)
- 已有的本地评估与选型记录：[docs/可行性与训练数据评估-2026-10-12.md](docs/可行性与训练数据评估-2026-10-12.md)
