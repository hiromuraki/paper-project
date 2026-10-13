# Skeleton Demo

为基于骨骼角度的规则评估系统增加目标动作筛选，减少非目标动作造成的错误判定，同时保留对不标准目标动作的评估能力。

骨架序列二分类 demo：输入 `[T, 17, 2]`、`[T, 17, 3]` 或两者，输出动作是否良好（True / False）。

## 1. 环境信息

本项目使用 uv 作为环境管理

- Python 3.11.14
- torch 2.14.0+cu130
- numpy 2.4.6
- pyyaml 6.0.3

如果 `.venv` 不存在，用 `uv venv --python 3.11` 配合 `uv sync` 重建；重建后确认 `python -c "import skeleton_demo"` 可用，若不可用再执行 `uv pip install -e .`（或临时 `export PYTHONPATH=src`）。

## 2. 快速开始

```bash
# 训练
uv run -m skeleton_demo.train --config configs/squat.yaml

# 评估
uv run -m skeleton_demo.evaluate --config configs/squat.yaml --ckpt runs/squat/best.pt

# 单条推理
uv run -m skeleton_demo.infer --ckpt runs/squat/best.pt --input data/squat_val.pt
```

1. 运行 `uv run scripts/pack_dataset.py --dataset ./dataset` 转化数据集，默认输出到 `./data/dataset/`。输入须有 train/val 子目录，test 可选。打包保持原有标签。详细规则见 [数据格式](dataset/README.md)。
2. 再执行 `uv run -m skeleton_demo.train --config data/dataset/train.yaml`。现有 mock 的质量标签不能直接代表“目标动作/非目标动作”。
  
**训练输出概览:**

```bash
epoch=001 loss=0.6500 val_acc=0.7000 val_f1=0.7200 val_auroc=0.7800
```

| 输出                 | 含义                             |
| -------------------- | -------------------------------- |
| `val_acc=0.5000`     | 验证集准确率                     |
| `val_f1=0.6667`      | 对正类的精确率和召回率的综合指标 |
| `val_auroc=1.0000`   | 正样本得分高于负样本的排序表现   |
| `best val f1=1.0000` | 训练期间最高验证 F1              |

> 只想确认管线是否连通时，改用 `configs/smoke.yaml`（2 epoch、`d_model=64`，结果写到 `runs/smoke`）。

## 3. 目录结构

| 目录           | 说明                                                     |
| -------------- | -------------------------------------------------------- |
| `tools_app`    | 带有 web 页面的工具应用                                  |
| `tools_script` | 纯脚本工具，`*_tests.py` 为其对应的测试脚本              |
| `.scratch`     | 概念验证目录，此目录专门用于存放各类验证可行性的临时代码 |

### 3.1. tools_app

统一采用 `./tools_app/<tool_name>/run.sh` / `.\tools_app\<tool_name>\run.ps1` 的形式运行

| 工具            | 说明                                                 |
| --------------- | ---------------------------------------------------- |
| `kps_viewer`    | 骨骼查看工具，用于查看 dataset 中的 2D / 3D 骨骼序列 |
| `kps_extractor` | 骨骼提取工具，用于从视频中提取 2D / 3D 骨骼序列      |

### 3.2. tools_script

统一采用 `uv run tools_script/<tool_name>.py <args>` 的形式运行

| 工具                 | 说明                                          |
| -------------------- | --------------------------------------------- |
| `convert_ui_prmd.py` | 将 UI-PRMD 的 Vicon 数据转换为 H36M-17 资源包 |
| `pack_dataset.py`    | 校验已划分的 NPZ 样本，打包 PT 并生成训练配置 |

**convert_ui_prmd.py**:

```bash
uv run scripts/convert_ui_prmd.py --help
uv run scripts/convert_ui_prmd.py --root "dataset-raw\UI-PRMD" --out "dataset-ui-prmd" --movement all --whole-trial --view front --max-gap 0 --fps 30
```

`--segmented` 与 `--whole-trial` 二选一；`--out` 必须为空，已有结果不会覆盖。转换说明见 [UI-PRMD-转换-v2.md](docs/UI-PRMD-转换-v2.md)。

**pack_dataset.py**:

```bash
uv run scripts/pack_dataset.py --dataset ./dataset # 默认输出到 ./data/dataset
```

## 4. 训练流程

| 阶段     | 当前代码做什么                                                          |
| -------- | ----------------------------------------------------------------------- |
| 准备标签 | 从 `meta.json` 读取 `label`；模型本身不理解这个标签代表质量还是动作类别 |
| 数据划分 | 准备好数据划分（训练、测试、验证）                                      |
| 读取样本 | 读取一条 `[T,17,2]` 或 `[T,17,3]` 的骨骼序列                            |
| 预处理   | 髋部对齐、尺度归一化、重采样为配置指定的帧数                            |
| 训练增强 | 随机缩放、镜像、噪声、3D 旋转、模态丢弃                                 |
| 特征构造 | 拼接坐标、骨向量和相邻帧位置差                                          |
| 分批训练 | 多条样本组成一批，模型输出每条样本的预测值                              |
| 参数更新 | 用标签计算二分类损失，通过反向传播和 AdamW 更新参数                     |
| 验证     | 关闭随机增强和训练时的 dropout，计算验证指标，不更新参数                |
| 保存     | 验证 F1 严格提高时保存 `best.pt`                                        |

## 5. 数据格式

推荐原始数据目录：

```text
dataset/
  train/
    clip_good_0001/
      kp2d.npz
      kp3d.npz
      meta.json
      remark.txt
    clip_bad_0001/
      ...
  val/
  test/
```

`kp2d.npz`：

```text
keypoints: [T, 17, 2] float32
```

`kp3d.npz`：

```text
keypoints: [T, 17, 3] float32
```

完整格式说明见：

- [dataset/README.md](dataset/README.md)

## 6. 训练文件格式

打包后的训练文件可以是 `.pt`，需要包含：

```text
x2d: list of [T, 17, 2] 或 None
x3d: list of [T, 17, 3] 或 None
y:   [N]，1 表示动作良好，0 表示动作错误
```

可选元信息：

```text
clip_id: [N]
subject: [N]
view: [N]
error_type: [N]
```

至少提供 `x2d` 和 `x3d` 中的一个。

## 7. 当前假设

- 骨骼由外部模型提供，本 demo 不负责姿态估计；
- 当前只做单个动作的 True/False 判断；
- 2D、3D 或两者同时输入均可；
- 输入骨骼序列在进入模型前会被归一化到固定长度。
