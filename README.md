# Skeleton Demo

骨架序列二分类 demo：输入 `[T, 17, 2]`、`[T, 17, 3]` 或两者，输出动作是否良好（True / False）。

## 快速开始

建议使用 Python 3.11 或 3.12（PyTorch 对 Python 3.14 的支持通常滞后）。当前 `.venv` 为 Python 3.11，已按可编辑模式安装本包，因此下面命令在项目根目录直接执行即可，不需要设置 `PYTHONPATH`。

```bash
cd /home/user/paper-project

# 已经存在 .venv 时，直接使用
.venv/bin/python scripts/make_synthetic_data.py --out data

# 训练
.venv/bin/python -m skeleton_demo.train --config configs/squat.yaml

# 评估
.venv/bin/python -m skeleton_demo.evaluate --config configs/squat.yaml --ckpt runs/squat/best.pt

# 单条推理
.venv/bin/python -m skeleton_demo.infer --ckpt runs/squat/best.pt --input data/squat_val.pt
```

只想确认管线是否连通时，改用 `configs/smoke.yaml`（2 epoch、`d_model=64`，结果写到 `runs/smoke`）。

当前环境已验证：Python 3.11.14、torch 2.14.0+cu130、numpy 2.4.6、pyyaml 6.0.3，CUDA 可用（RTX 4050 Laptop），训练与评估会自动使用 GPU。

如果 `.venv` 不存在，用 `uv venv --python 3.11` 配合 `uv sync` 重建；重建后确认 `python -c "import skeleton_demo"` 可用，若不可用再执行 `uv pip install -e .`（或临时 `export PYTHONPATH=src`）。

## 数据格式

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

## 打包为训练文件

```bash
cd /home/user/paper-project
.venv/bin/python scripts/make_mock_dataset.py --out dataset/mock
.venv/bin/python scripts/pack_dataset.py --dataset dataset/mock --out data --action squat
```

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

## 当前假设

- 骨骼由外部模型提供，本 demo 不负责姿态估计；
- 当前只做单个动作的 True/False 判断；
- 2D、3D 或两者同时输入均可；
- 输入骨骼序列在进入模型前会被归一化到固定长度。
