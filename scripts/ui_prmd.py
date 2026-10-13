"""UI-PRMD 读取与体检工具（只读，不修改 dataset-raw）。

UI-PRMD 原始文件为逗号分隔的 ASCII txt，没有表头。Kinect 与 Vicon 的列含义、
单位、坐标系都不同，见 docs/UI-PRMD-数据集说明.md。这里只做三件事：

1. ``load_ui_prmd`` 把任意一个 txt 文件读成 ``[T, J, D]`` 数组；
2. ``kinect_absolute`` 把 Kinect 的相对坐标累加成绝对坐标；
3. ``survey`` 扫描整个数据集，输出规模、形状、缺失文件与统计摘要。

命令行用法::

    python scripts/ui_prmd.py                      # 全量体检，打印摘要
    python scripts/ui_prmd.py --json out.json      # 同时写出 JSON 报告
    python scripts/ui_prmd.py --quick              # 抽样体检（每个划分只查 3 个文件）
    python scripts/ui_prmd.py --dump m01_s01_e01 --system Kinect --kind positions \
        --out tmp/m01_s01_e01_abs.npy              # 导出单条绝对坐标
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- 常量与关节表

# 原论文 Table 2：Kinect 位置与角度共用同一套 22 关节顺序。
KINECT_JOINTS = [
    "Waist", "Spine", "Chest", "Neck", "Head", "HeadTip",
    "LeftCollar", "LeftUpperArm", "LeftForearm", "LeftHand",
    "RightCollar", "RightUpperArm", "RightForearm", "RightHand",
    "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes",
    "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes",
]

# Kinect 骨骼父子关系（按上表下标）。论文只说明“相对父关节”，没给显式父表；
# 这里按关节链常识给出，供 kinect_absolute 使用。
KINECT_PARENTS = [-1, 0, 1, 2, 3, 4, 2, 6, 7, 8, 2, 10, 11, 12, 0, 14, 15, 16, 0, 18, 19, 20]

# 原论文 Table 2：Vicon 39 个反光标记名称（位置与角度列数相同）。
VICON_MARKERS = [
    "LFHD", "RFHD", "LBHD", "RBHD", "C7", "T10", "CLAV", "STRN", "RBAK",
    "LSHO", "LUPA", "LELB", "LFRM", "LWRA", "LWRB", "LFIN",
    "RSHO", "RUPA", "RELB", "RFRM", "RWRA", "RWRB", "RFIN",
    "LASI", "RASI", "LPSI", "RPSI", "LTHI", "LKNE", "LTIB", "LANK", "LHEE", "LTOE",
    "RTHI", "RKNE", "RTIB", "RANK", "RHEE", "RTOE",
]

# 本数据集里 Vicon 实际缺失的一个文件（原论文明确说明：marker 测量缺失）。
KNOWN_MISSING = {"m03_s03"}

VICON_FPS = 100
KINECT_FPS = 30

_NAME_RE = re.compile(
    r"^(?P<movement>m\d{2})_(?P<subject>s\d{2})"
    r"(?:_(?P<episode>e\d{2}))?"
    r"_(?P<kind>positions|angles)_inc$|"
    r"^(?P<movement2>m\d{2})_(?P<subject2>s\d{2})"
    r"(?:_(?P<episode2>e\d{2}))?"
    r"_(?P<kind2>positions|angles)$"
)

# 数据集顶层四个划分：目录名 -> (label, segmented)
SUBSETS = {
    "Movements": (1, False),
    "Segmented Movements": (1, True),
    "Incorrect Movements": (0, False),
    "Incorrect Segmented Movements": (0, True),
}

SYSTEMS = ("Kinect", "Vicon")
KINDS = ("positions", "angles")


# ---------------------------------------------------------------- 基础读取

def parse_name(stem: str) -> dict:
    """把 ``m01_s01_e02_positions_inc`` 解析成结构化字段。"""
    incorrect = stem.endswith("_inc")
    body = stem[: -len("_inc")] if incorrect else stem
    match = re.match(
        r"^(?P<movement>m\d{2})_(?P<subject>s\d{2})"
        r"(?:_(?P<episode>e\d{2}))?_"
        r"(?P<kind>positions|angles)$",
        body,
    )
    if not match:
        raise ValueError(f"无法解析文件名: {stem}")
    parts = match.groupdict()
    return {
        "clip_id": stem,
        "movement_id": parts["movement"],
        "subject_id": parts["subject"],
        "episode_id": parts["episode"],
        "kind": parts["kind"],
        "incorrect": incorrect,
    }


def load_ui_prmd(path: str | Path, expect_cols: int | None = None) -> np.ndarray:
    """读取一个 UI-PRMD txt，返回 ``[T, J, D]`` float64 数组。

    列数除以 3 得到关节数：Kinect 为 22，Vicon 为 39。空白行会被忽略。

    分隔符实测不统一：``Movements`` / ``Incorrect Movements`` 下的整体文件用空格，
    ``Segmented Movements`` / ``Incorrect Segmented Movements`` 下的分片文件用逗号
    （原论文正文写的是逗号分隔）。这里两种都接受。
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    rows = [line for line in text.replace("\r\n", "\n").split("\n") if line.strip()]
    if not rows:
        raise ValueError(f"空文件: {path}")
    table = np.array(
        [[float(value) for value in row.replace(",", " ").split()] for row in rows]
    )
    if table.shape[1] % 3 != 0:
        raise ValueError(f"{path.name}: 列数 {table.shape[1]} 不是 3 的倍数")
    if expect_cols is not None and table.shape[1] != expect_cols:
        raise ValueError(f"{path.name}: 期望 {expect_cols} 列，实际 {table.shape[1]} 列")
    return table.reshape(table.shape[0], table.shape[1] // 3, 3)


def kinect_absolute(relative: np.ndarray,
                    parents: list[int] = KINECT_PARENTS) -> np.ndarray:
    """把 Kinect 的“根关节绝对 + 其余关节相对父关节”累加成全绝对坐标。

    Args:
        relative: ``[T, 22, 3]``，第 0 关节为绝对坐标，其余为相对父关节的偏移。

    Returns:
        ``[T, 22, 3]`` 绝对坐标，单位与输入一致。

    注意：这只是按论文描述做的累加。实测这些“相对坐标”的模长并不恒定
    （同一 trial 内髋-膝距离变化可达 20% 以上），所以不能保证累加得到的骨架
    在物理上正确；使用前请按 docs/UI-PRMD-数据集说明.md 的方法先验证。
    """
    if relative.ndim != 3 or relative.shape[1] != len(parents):
        raise ValueError(f"期望 [T, {len(parents)}, 3]，实际 {relative.shape}")
    absolute = np.empty_like(relative)
    for joint, parent in enumerate(parents):
        if parent < 0:
            absolute[:, joint] = relative[:, joint]
        else:
            absolute[:, joint] = absolute[:, parent] + relative[:, joint]
    return absolute


# H36M 17 关节下标（与 src/skeleton_demo/data.py 一致）
H36M_NAMES = [
    "Hip", "RHip", "RKnee", "RFoot", "LHip", "LKnee", "LFoot",
    "Spine", "Thorax", "Neck", "Head",
    "LShoulder", "LElbow", "LWrist", "RShoulder", "RElbow", "RWrist",
]


def vicon_to_h36m(markers: np.ndarray) -> np.ndarray:
    """把 Vicon 的 39 个标记点转换成 H36M 17 关节位置。

    本项目管线只吃 H36M 17 关节，而 UI-PRMD 的 Vicon 是 39 个反光标记点，
    数量与拓扑都不同，必须显式转换。对应关系（标记点名称见 VICON_MARKERS）：

    =================  ==============================================
    H36M 关节          来源
    =================  ==============================================
    Hip                LASI 与 RASI 的中点
    RHip/RKnee/RFoot   RASI / RKNE / RANK
    LHip/LKnee/LFoot   LASI / LKNE / LANK
    Spine              T10
    Thorax             C7 与 STRN 的中点
    Neck               C7（颈根，位于肩峰与头部标记之间）
    Head               四个头部标记 LFHD/RFHD/LBHD/RBHD 的中点（颅骨中心）
    LShoulder          LSHO
    LElbow / LWrist    LELB / LWRA 与 LWRB 的中点
    RShoulder          RSHO
    RElbow / RWrist    RELB / RWRA 与 RWRB 的中点
    =================  ==============================================

    这是本项目采用的约定，不是唯一正确的映射；肩、颈、头三个关节可以按需调整。

    Args:
        markers: ``[T, 39, 3]`` Vicon 标记点绝对坐标。

    Returns:
        ``[T, 17, 3]``，单位与坐标系与输入相同（毫米、Z 轴向上）。
    """
    if markers.ndim != 3 or markers.shape[1] != len(VICON_MARKERS):
        raise ValueError(f"期望 [T, {len(VICON_MARKERS)}, 3]，实际 {markers.shape}")
    index = {name: i for i, name in enumerate(VICON_MARKERS)}

    def marker(*names: str) -> np.ndarray:
        return np.mean([markers[:, index[name], :] for name in names], axis=0)

    return np.stack([
        marker("LASI", "RASI"),   # 0  Hip
        marker("RASI"),           # 1  RHip
        marker("RKNE"),           # 2  RKnee
        marker("RANK"),           # 3  RFoot
        marker("LASI"),           # 4  LHip
        marker("LKNE"),           # 5  LKnee
        marker("LANK"),           # 6  LFoot
        marker("T10"),            # 7  Spine
        marker("C7", "STRN"),     # 8  Thorax
        marker("C7"),             # 9  Neck
        marker("LFHD", "RFHD", "LBHD", "RBHD"),         # 10 Head
        marker("LSHO"),           # 11 LShoulder
        marker("LELB"),           # 12 LElbow
        marker("LWRA", "LWRB"),   # 13 LWrist
        marker("RSHO"),           # 14 RShoulder
        marker("RELB"),           # 15 RElbow
        marker("RWRA", "RWRB"),   # 16 RWrist
    ], axis=1)


def project_2d(points: np.ndarray, mode: str = "front") -> np.ndarray:
    """把 3D 关节投影成归一化图像坐标 ``[T, J, 2]``（仅供 smoke 与 2D 分支使用）。

    UI-PRMD 的 2D 只在 Kinect 侧、且关节定义与 Vicon 不同，无法与 Vicon 逐帧配对。
    这里用固定的虚拟相机把 Vicon 3D 投到 2D：x 向右、y 向下、原点在画面中心，
    量纲按整段 Y 方向跨度归一化。这是派生数据，`meta.json` 会标记 ``derived_2d``，
    不能用它报告真实的 2D 指标。
    """
    axes = {"front": (0, 1), "side": (2, 1)}[mode]
    view = points[:, :, axes]
    center = view.reshape(-1, 2).mean(axis=0)
    scale = float(np.ptp(points[:, :, 1])) or 1.0
    image = (view - center) / scale
    image[:, :, 1] *= -1.0  # 图像 y 轴向下
    return image.astype(np.float32)


def joint_names(system: str) -> list[str]:
    return KINECT_JOINTS if system == "Kinect" else VICON_MARKERS


# ---------------------------------------------------------------- 体检

def _stats(array: np.ndarray) -> dict:
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def survey_subset(base: Path, subset: str, quick: bool = False,
                  sample: int = 3) -> dict:
    """扫描一个划分，返回文件数、行数/形状统计与缺失情况。"""
    root = base / subset
    report: dict = {"subset": subset, "exists": root.exists(), "systems": {}}
    if not root.exists():
        return report

    for system in SYSTEMS:
        entry: dict = {"kinds": {}, "missing": [], "orientation": {}}
        for kind in KINDS:
            folder = root / system / kind
            if not folder.exists():
                continue
            files = sorted(folder.glob("*.txt"))
            expected_joints = len(joint_names(system))
            rows: list[int] = []
            cols_ok = True
            bad: list[str] = []
            vertical_axis = None
            seen_ids: set[str] = set()

            selected = files
            if quick:
                selected = files[:: max(1, len(files) // sample)][:sample]

            for path in selected:
                meta = parse_name(path.stem)
                seen_ids.add(f"{meta['movement_id']}_{meta['subject_id']}")
                try:
                    data = load_ui_prmd(path, expect_cols=expected_joints * 3)
                except ValueError as error:
                    cols_ok = False
                    bad.append(f"{path.name}: {str(error)[:80]}")
                    continue
                rows.append(int(data.shape[0]))
                if kind == "positions" and vertical_axis is None:
                    # 竖直轴 = 关节绝对位置均值绝对值最大的那一维。
                    head = data[: min(200, len(data))]
                    center = head[:, 0, :] if system == "Kinect" else head.reshape(-1, 3)
                    vertical_axis = int(np.argmax(np.abs(np.asarray(center).mean(axis=0))))

            entry["kinds"][kind] = {
                "files": len(files),
                "checked": len(selected) if quick else len(files),
                "rows_total": int(sum(rows)),
                "rows_min": int(min(rows)) if rows else None,
                "rows_max": int(max(rows)) if rows else None,
                "rows_mean": float(np.mean(rows)) if rows else None,
                "joints": expected_joints,
                "cols_ok": cols_ok,
                "bad_files": bad[:10],
            }
            if kind == "positions":
                entry["orientation"]["vertical_axis_index"] = vertical_axis
                entry["coordinate_convention"] = (
                    "Y-up（Kinect 传感器坐标系）" if system == "Kinect"
                    else "Z-up（Vicon 实验室坐标系）"
                )
                entry["unit"] = "millimetre"

            if not quick:
                expected_ids = {
                    f"m{movement:02d}_s{subject:02d}"
                    for movement in range(1, 11)
                    for subject in range(1, 11)
                }
                missing = sorted(expected_ids - seen_ids)
                entry["missing"] = [
                    name for name in missing if name not in KNOWN_MISSING
                ]
                entry["missing_known"] = sorted(expected_ids & set(KNOWN_MISSING) & (
                    expected_ids - seen_ids
                ))
        report["systems"][system] = entry
    return report


def geometry_check(base: Path, system: str = "Kinect") -> dict:
    """用第一条深蹲（m01）检查骨段长度是否近似恒定，验证父子表是否合理。"""
    path = base / "Segmented Movements" / system / "Positions" / (
        "m01_s01_e01_positions.txt" if system == "Kinect"
        else "m01_s01_e01_positions.txt"
    )
    data = load_ui_prmd(path)
    if system == "Kinect":
        absolute = kinect_absolute(data)
        parents = KINECT_PARENTS
        names = KINECT_JOINTS
    else:
        absolute = data
        parents = list(range(-1, len(VICON_MARKERS) - 1))  # Vicon 全为绝对坐标
        names = VICON_MARKERS

    if system == "Kinect":
        lengths = {}
        for joint, parent in enumerate(parents):
            if parent < 0:
                continue
            bone = np.linalg.norm(absolute[:, joint] - absolute[:, parent], axis=-1)
            lengths[f"{names[parent]}->{names[joint]}"] = {
                "mean_mm": float(bone.mean()),
                "std_mm": float(bone.std()),
                "drift_mm": float(np.ptp(bone)),
            }
        worst = max(lengths.items(), key=lambda item: item[1]["drift_mm"])
        return {"system": system, "bones": lengths, "worst_bone": worst[0],
                "worst_drift_mm": worst[1]["drift_mm"]}
    # Vicon 位置为标记点绝对坐标，不做骨长检查。
    return {"system": system, "note": "Vicon 为 39 个标记点绝对坐标，无父子约束"}


def survey(base: Path, quick: bool = False) -> dict:
    result: dict = {
        "root": str(base),
        "units": "millimetre",
        "fps": {"Kinect": KINECT_FPS, "Vicon": VICON_FPS},
        "joints": {"Kinect": len(KINECT_JOINTS), "Vicon": len(VICON_MARKERS)},
        "subsets": {},
        "notes": [
            "Vicon m03_s03 全划分缺失（原论文说明：marker 测量缺失，Kinect 侧完整）",
            "Segmented 的文件名 episode 编号在正确/错误两份数据里不保证指向同一物理重复",
            "Kinect 位置除根关节外都按论文记为相对父关节的偏移；但实测偏移模长不恒定，"
            "累加出的骨架物理上不可信，使用前必须单独验证（详见 docs）",
            "Vicon 位置列为 39 个标记点的绝对坐标，单位毫米，Z 轴向上",
        ],
    }
    for subset in SUBSETS:
        result["subsets"][subset] = survey_subset(base, subset, quick=quick)
    return result


# ---------------------------------------------------------------- 转换

# UI-PRMD 10 个动作（论文 Table 1），本项目把它映射成 action_id。
MOVEMENT_NAMES = {
    "m01": "squat",
    "m02": "hurdle_step",
    "m03": "inline_lunge",
    "m04": "side_lunge",
    "m05": "sit_to_stand",
    "m06": "straight_leg_raise",
    "m07": "shoulder_abduction",
    "m08": "shoulder_extension",
    "m09": "shoulder_internal_external_rotation",
    "m10": "shoulder_scaption",
}

# 按受试者划分，避免同一人的不同重复同时出现在训练和测试里。
SUBJECT_SPLIT = {
    **{f"s{index:02d}": "train" for index in range(1, 7)},
    "s07": "val",
    "s08": "val",
    "s09": "test",
    "s10": "test",
}


def convert_vicon(root: Path, out: Path, movement: str = "m01",
                  segmented: bool = True, project: bool = True,
                  project_mode: str = "front") -> None:
    """把 UI-PRMD 转换成本项目 `dataset/` 目录要求的 clip 结构。

    输出 ``<out>/<split>/<clip_id>/{kp3d.npz,kp2d.npz,meta.json,remark.txt}``，
    之后可以直接跑 ``scripts/pack_dataset.py``。

    只有 Vicon 侧会做 39 标记点 -> H36M 17 关节的重映射；Kinect 侧的 22 关节
    与 H36M 17 拓扑不同、尺度也存在疑点，这里不转换，避免产生看似可用的错误数据。
    """
    system = "Vicon"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，拒绝覆盖: {out}")
    if out.resolve() == root.resolve() or root.resolve() in out.resolve().parents:
        raise SystemExit(f"输出目录不能落在原始数据目录内: {out}")
    subsets = ["Movements", "Incorrect Movements"] if not segmented else \
              ["Segmented Movements", "Incorrect Segmented Movements"]
    counts: dict[str, int] = {}
    for subset in subsets:
        folder = root / subset / system / "Positions"
        if not folder.exists():
            raise SystemExit(f"缺少目录: {folder}")
        for path in sorted(folder.glob("*.txt")):
            parsed = parse_name(path.stem)
            if parsed["movement_id"] != movement:
                continue
            split = SUBJECT_SPLIT[parsed["subject_id"]]
            markers = load_ui_prmd(path, expect_cols=len(VICON_MARKERS) * 3)
            joints = vicon_to_h36m(markers)
            if not np.isfinite(joints).all():
                raise ValueError(f"{path.name}: 转换后存在非法数值")

            clip_id = f"{movement}_{parsed['subject_id']}_" \
                      f"{parsed['episode_id'] or 'full'}_" \
                      f"{'bad' if parsed['incorrect'] else 'good'}"
            clip_dir = out / split / clip_id
            if clip_dir.exists():
                raise FileExistsError(f"拒绝覆盖已存在的 clip: {clip_dir}")
            clip_dir.mkdir(parents=True)

            kp3d = (joints / 1000.0).astype(np.float32)  # 毫米 -> 米
            np.savez_compressed(clip_dir / "kp3d.npz", keypoints=kp3d,
                                fps=np.int32(VICON_FPS))
            if project:
                np.savez_compressed(clip_dir / "kp2d.npz",
                                    keypoints=project_2d(kp3d, project_mode),
                                    fps=np.int32(VICON_FPS))

            meta = {
                "clip_id": clip_id,
                "action_id": MOVEMENT_NAMES[movement],
                "label": 0 if parsed["incorrect"] else 1,
                "label_name": "bad" if parsed["incorrect"] else "good",
                "subject_id": parsed["subject_id"],
                "view": "front",
                "fps": VICON_FPS,
                "num_frames": int(kp3d.shape[0]),
                "error_type": "non_optimal" if parsed["incorrect"] else None,
                "has_2d": bool(project),
                "has_3d": True,
                "joint_format": "h36m_17",
                "units_3d": "metres",
                "up_axis": "Z",
                "coordinate_system": "vicon_lab_z_up",
                "derived_2d": bool(project),
                "projection": f"virtual_{project_mode}_orthographic" if project else None,
                "source": "UI-PRMD",
                "source_file": f"{subset}/{system}/Positions/{path.name}",
                "source_markers": "vicon_39",
                "label_source": "dataset_folder_correct_vs_incorrect",
                "notes": [
                    "Vicon 39 标记点按约定重映射为 H36M 17 关节",
                    "错误样本未标注具体错误类型，只区分正确/非最优",
                ],
            }
            (clip_dir / "meta.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            (clip_dir / "remark.txt").write_text(
                f"来源: {subset}/{system}/Positions/{path.name}\n"
                f"Vicon 39 标记点 -> H36M 17 关节（约定见 scripts/ui_prmd.py）\n",
                encoding="utf-8")
            counts[split] = counts.get(split, 0) + 1

    if not counts:
        raise SystemExit(f"没有找到动作 {movement} 的文件")
    print(f"已写入 {out}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print("下一步: python scripts/pack_dataset.py "
          f"--dataset {out} --out <打包目录> --action {MOVEMENT_NAMES[movement]}")


# ---------------------------------------------------------------- CLI

def _print_summary(report: dict) -> None:
    print(f"根目录: {report['root']}")
    print(f"单位: {report['units']}   Kinect {report['joints']['Kinect']} 关节 / "
          f"Vicon {report['joints']['Vicon']} 标记点")
    for subset, entry in report["subsets"].items():
        label, segmented = SUBSETS[subset]
        print(f"\n=== {subset} (label={label}, segmented={segmented}) ===")
        if not entry["exists"]:
            print("  目录不存在")
            continue
        for system, sys_entry in entry["systems"].items():
            for kind, info in sys_entry["kinds"].items():
                rows = (f"{info['rows_min']}/{info['rows_mean']:.1f}/{info['rows_max']}"
                        if info["rows_mean"] is not None else "n/a")
                print(f"  {system}/{kind}: {info['files']} 文件, "
                      f"关节={info['joints']}, 行数 min/mean/max={rows}, "
                      f"列数正确={info['cols_ok']}")
                for bad in info["bad_files"]:
                    print(f"    ! {bad}")
            if sys_entry["missing"]:
                print(f"  {system} 缺失: {sys_entry['missing']}")
            if sys_entry.get("missing_known"):
                print(f"  {system} 已知缺失: {sys_entry['missing_known']}")


def main() -> None:
    try:  # Windows 控制台默认 GBK，中文与报告里的符号会乱码。
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path,
                        default=Path("dataset-raw/UI-PRMD"))
    parser.add_argument("--json", type=Path, default=None,
                        help="把体检报告写到该 JSON 文件")
    parser.add_argument("--quick", action="store_true",
                        help="每个划分只抽样若干文件")
    parser.add_argument("--geometry", action="store_true",
                        help="额外做 Kinect 骨长恒定检查")
    parser.add_argument("--dump", type=str, default=None,
                        help="导出单条序列，可用 'm01_s01_e01' 或 'm01_s01_e01_positions'")
    parser.add_argument("--system", choices=SYSTEMS, default="Kinect")
    parser.add_argument("--kind", choices=KINDS, default="positions")
    parser.add_argument("--subset", choices=list(SUBSETS), default="Segmented Movements")
    parser.add_argument("--out", type=Path, default=Path("tmp/ui_prmd_dump.npy"))
    parser.add_argument("--convert", action="store_true",
                        help="把 Vicon 数据转换成本项目 dataset/ 的 clip 结构")
    parser.add_argument("--dataset-out", type=Path,
                        default=Path("dataset/ui-prmd"),
                        help="--convert 的输出目录")
    parser.add_argument("--movement", default="m01",
                        help="要转换的动作编号，默认 m01 深蹲")
    parser.add_argument("--whole-trial", action="store_true",
                        help="--convert 时用整体文件而不是分片（每个 trial 一个 clip）")
    parser.add_argument("--no-2d", action="store_true",
                        help="--convert 时不生成派生的 2D 投影")
    args = parser.parse_args()

    if not args.root.exists():
        raise SystemExit(f"找不到数据集目录: {args.root}")

    if args.convert:
        convert_vicon(args.root, args.dataset_out, movement=args.movement,
                      segmented=not args.whole_trial, project=not args.no_2d)
        return

    if args.dump:
        kind_dir = args.kind.capitalize()  # positions -> Positions
        folder = args.root / args.subset / args.system / kind_dir
        candidates = [folder / f"{args.dump}.txt", folder / f"{args.dump}_{args.kind}.txt"]
        path = next((item for item in candidates if item.exists()), None)
        if path is None:
            raise SystemExit(f"找不到文件，试过: "
                             + ", ".join(str(item) for item in candidates))
        data = load_ui_prmd(path)
        if args.system == "Kinect" and args.kind == "positions":
            data = kinect_absolute(data)
            print("Kinect 相对坐标已按论文累加为绝对坐标（是否物理可信见 docs 说明）")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.out, data)
        print(f"{path.name}: shape={data.shape} -> {args.out}")
        print(f"  范围: min={data.min():.3f} max={data.max():.3f}")
        return

    report = survey(args.root, quick=args.quick)
    _print_summary(report)
    if args.geometry:
        geometry = geometry_check(args.root, "Kinect")
        print(f"\nKinect 骨长检查: 漂移最大的骨段 {geometry['worst_bone']} "
              f"ptp={geometry['worst_drift_mm']:.3f} mm")
        report["geometry"] = geometry
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
        print(f"\n报告已写入 {args.json}")


if __name__ == "__main__":
    main()
