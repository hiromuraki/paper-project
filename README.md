# Skeleton Demo

骨架序列二分类 demo：输入 `[T, 17, 2]`、`[T, 17, 3]` 或两者，输出动作是否良好（True / False）。

## 1. 目录结构

| 目录       | 说明                                                     |
| ---------- | -------------------------------------------------------- |
| `tools`    | 带有 web 页面的工具应用                                  |
| `scripts`  | 工具脚本，`*_tests.py` 为其对应的测试脚本                |
| `.scratch` | 概念验证目录，此目录专门用于存放各类验证可行性的临时代码 |

### 1.1. tools

统一采用 `./tools/<tool_name>/run.sh` / `.\tools\<tool_name>\run.ps1` 的形式运行

| 工具         | 说明                                                 |
| ------------ | ---------------------------------------------------- |
| `kps_viewer` | 骨骼查看工具，用于查看 dataset 中的 2D / 3D 骨骼序列 |

### 1.2. scripts

统一采用 `uv run scripts/<tool_name>.py <args>` 的形式运行

| 工具                 | 说明                                          |
| -------------------- | --------------------------------------------- |
| `convert_ui_prmd.py` | 将 UI-PRMD 的 Vicon 数据转换为 H36M-17 资源包 |

**convert_ui_prmd.py**:

```powershell
uv run scripts/convert_ui_prmd.py --help
uv run scripts/convert_ui_prmd.py --root "dataset-raw\UI-PRMD" --out "dataset-ui-prmd" --movement all --whole-trial --view front --max-gap 0 --fps 30 
```

`--segmented` 与 `--whole-trial` 二选一；`--out` 必须为空，已有结果不会覆盖。转换说明见 [UI-PRMD-转换-v2.md](docs/UI-PRMD-转换-v2.md)。

## 2. 快速开始

建议使用 Python 3.11 或 3.12（PyTorch 对 Python 3.14 的支持通常滞后）。当前 `.venv` 为 Python 3.11，已按可编辑模式安装本包，因此下面命令在项目根目录直接执行即可，不需要设置 `PYTHONPATH`。

```bash
cd /home/user/paper-project

# 训练
.venv/bin/python -m skeleton_demo.train --config configs/squat.yaml

# 评估
.venv/bin/python -m skeleton_demo.evaluate --config configs/squat.yaml --ckpt runs/squat/best.pt

# 单条推理
.venv/bin/python -m skeleton_demo.infer --ckpt runs/squat/best.pt --input data/squat_val.pt
```

训练前需要另行准备配置指定的 `.pt` 训练文件；当前转换工具只输出 NPZ 资源包，不提供打包脚本。

只想确认管线是否连通时，改用 `configs/smoke.yaml`（2 epoch、`d_model=64`，结果写到 `runs/smoke`）。

当前环境已验证：Python 3.11.14、torch 2.14.0+cu130、numpy 2.4.6、pyyaml 6.0.3，CUDA 可用（RTX 4050 Laptop），训练与评估会自动使用 GPU。

如果 `.venv` 不存在，用 `uv venv --python 3.11` 配合 `uv sync` 重建；重建后确认 `python -c "import skeleton_demo"` 可用，若不可用再执行 `uv pip install -e .`（或临时 `export PYTHONPATH=src`）。

## 3. 数据格式

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

## 4. 训练文件格式

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

## 5. 当前假设

- 骨骼由外部模型提供，本 demo 不负责姿态估计；
- 当前只做单个动作的 True/False 判断；
- 2D、3D 或两者同时输入均可；
- 输入骨骼序列在进入模型前会被归一化到固定长度。
