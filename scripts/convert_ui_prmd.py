"""UI-PRMD 转换工具：Vicon → H36M-17，Y-up 3D 与对应的 2D 投影。

用法：uv run scripts/convert_ui_prmd.py --help
所有转换参数均需显式指定；无参数运行只显示帮助。
输出目录必须为空；再次转换请用 --out 指定新目录。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

VICON_MARKERS = [
    "LFHD", "RFHD", "LBHD", "RBHD", "C7", "T10", "CLAV", "STRN", "RBAK",
    "LSHO", "LUPA", "LELB", "LFRM", "LWRA", "LWRB", "LFIN",
    "RSHO", "RUPA", "RELB", "RFRM", "RWRA", "RWRB", "RFIN",
    "LASI", "RASI", "LPSI", "RPSI", "LTHI", "LKNE", "LTIB", "LANK", "LHEE", "LTOE",
    "RTHI", "RKNE", "RTIB", "RANK", "RHEE", "RTOE",
]

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

SUBJECT_SPLIT = {
    **{f"s{index:02d}": "train" for index in range(1, 7)},
    "s07": "val",
    "s08": "val",
    "s09": "test",
    "s10": "test",
}

SOURCE_FPS = 100
INDEX = {name: index for index, name in enumerate(VICON_MARKERS)}
H36M_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
# Optical surface landmarks approximate joint centres, not an anatomical calibration.
JOINT_MARKERS = [
    ('LASI', 'RASI'), ('RASI',), ('RKNE',), ('RANK',),
    ('LASI',), ('LKNE',), ('LANK',), ('T10',), ('C7', 'STRN'),
    ('C7',), ('LFHD', 'RFHD', 'LBHD', 'RBHD'),
    ('LSHO',), ('LELB',), ('LWRA', 'LWRB'),
    ('RSHO',), ('RELB',), ('RWRA', 'RWRB'),
]


def read_positions(path: Path) -> np.ndarray:
    with path.open(encoding='utf-8-sig') as stream:
        first = next((line for line in stream if line.strip()), '')
    data = np.loadtxt(path, delimiter=',' if ',' in first else None, ndmin=2, encoding='utf-8-sig')
    if data.shape[1] != 117:
        raise ValueError(f'expected 117 Vicon position columns, got {data.shape}')
    if len(data) < 2:
        raise ValueError(f'truncated sequence: {len(data)} frame(s)')
    return data.reshape(-1, 39, 3)


def repair_markers(markers: np.ndarray, max_gap: int = 10) -> tuple[np.ndarray, dict]:
    """Only interpolate short internal gaps; never fabricate absent whole markers."""
    repaired = np.array(markers, dtype=np.float64, copy=True)
    valid = np.isfinite(repaired).all(axis=-1) & (np.linalg.norm(np.nan_to_num(repaired), axis=-1) > 1e-6)
    repaired[~valid] = np.nan
    report = {'missing_samples': int((~valid).sum()), 'interpolated_samples': 0,
              'absent_markers': [VICON_MARKERS[j] for j in range(39) if not valid[:, j].any()]}
    for j in range(39):
        mask = ~valid[:, j]
        runs = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)))
        for start, end in runs.reshape(-1, 2):
            if start == 0 or end == len(markers) or end - start > max_gap:
                continue
            weights = np.arange(1, end - start + 1)[:, None] / (end - start + 1)
            repaired[start:end, j] = repaired[start - 1, j] * (1 - weights) + repaired[end, j] * weights
            report['interpolated_samples'] += int(end - start)
    return repaired, report


def map_h36m(markers: np.ndarray) -> tuple[np.ndarray, dict]:
    points = []
    fallback = {}
    for j, names in enumerate(JOINT_MARKERS):
        values = markers[:, [INDEX[name] for name in names]]
        valid = np.isfinite(values).all(axis=-1)
        count = valid.sum(axis=1)
        if (count == 0).any():
            raise ValueError(f'H36M joint {j} has unrecoverable missing markers: {names}')
        points.append(np.where(valid[..., None], values, 0).sum(axis=1) / count[:, None])
        if (count < len(names)).any():
            fallback[str(j)] = int((count < len(names)).sum())
    return np.stack(points, axis=1), fallback


def canonical_coordinates(joints_mm: np.ndarray) -> tuple[np.ndarray, dict]:
    """One rigid transform per clip: anatomical right, lab vertical, forward.

    Vicon's lab Z is vertical. Use the horizontal pelvis line for anatomical X,
    then X cross Y for Z. This is a rotation (det=+1), never a mirror/per-frame warp.
    Keep motion relative to the first pelvis position, not a framewise reset.
    """
    right = np.median(joints_mm[:, 1] - joints_mm[:, 4], axis=0)
    right[2] = 0
    if np.linalg.norm(right) < 20:
        raise ValueError('pelvis left/right baseline is degenerate')
    right /= np.linalg.norm(right)
    up = np.array([0., 0., 1.])
    forward = np.cross(right, up)
    basis = np.stack([right, up, forward], axis=1)
    origin = joints_mm[0, 0].copy()
    points = ((joints_mm - origin) @ basis / 1000).astype(np.float32)
    if np.median(points[:, 10, 1] - (points[:, 3, 1] + points[:, 6, 1]) / 2) < 0.5:
        raise ValueError('head/feet vertical geometry is invalid')
    lengths = np.stack([np.linalg.norm(points[:, j] - points[:, p], axis=-1)
                        for j, p in enumerate(H36M_PARENTS) if p >= 0], axis=1)
    median = np.median(lengths, axis=0)
    if (median < 0.015).any() or (median > 1.0).any():
        raise ValueError(f'implausible bone lengths (m): {median.tolist()}')
    info = {'lab_origin_mm': origin.tolist(), 'lab_to_canonical_rotation': basis.tolist(),
            'bone_median_m': median.tolist(),
            'bone_length_cv': (lengths.std(axis=0) / np.maximum(lengths.mean(axis=0), 1e-6)).tolist()}
    return points, info


def resample_frames(points: np.ndarray, fps: int) -> tuple[np.ndarray, dict]:
    """Uniform output timestamps k/fps, within the original [0,(N-1)/100].

    Linear interpolation in 3D; never append an off-grid final timestamp or
    extrapolate. The unsampled tail is shorter than one output frame interval.
    """
    if not isinstance(fps, (int, np.integer)) or not 1 <= fps <= 1000:
        raise ValueError('fps must be an integer from 1 to 1000')
    count = (len(points) - 1) * fps // SOURCE_FPS + 1
    position = np.arange(count, dtype=np.float64) * SOURCE_FPS / fps
    low = np.floor(position).astype(int)
    high = np.minimum(low + 1, len(points) - 1)
    weight = (position - low)[:, None, None]
    sampled = (points[low] * (1 - weight) + points[high] * weight).astype(np.float32)
    return sampled, {
        'method': 'timestamp_linear_interpolation', 'source_fps': SOURCE_FPS,
        'target_fps': fps, 'source_num_frames': len(points), 'output_num_frames': count,
        'source_last_timestamp_s': (len(points) - 1) / SOURCE_FPS,
        'output_last_timestamp_s': (count - 1) / fps,
        'discarded_tail_s': (len(points) - 1) / SOURCE_FPS - (count - 1) / fps,
        'anti_alias_filter': False,
    }


def project_2d(points: np.ndarray, view: str = 'front') -> np.ndarray:
    plane = points[..., [0 if view == 'front' else 2, 1]].astype(np.float64)
    low = plane.min(axis=(0, 1)); high = plane.max(axis=(0, 1))
    span = float(np.max(high - low))
    if span <= 1e-6:
        raise ValueError('empty projection extent')
    image = (plane - (low + high) / 2) / (span / 0.9)
    image[..., 1] *= -1  # screen Y is down; anatomical Y is up
    return (image + 0.5).astype(np.float32)


def convert(root: Path, out: Path, movement: str, whole_trial: bool,
            view: str, max_gap: int, fps: int) -> dict:
    if not isinstance(fps, (int, np.integer)) or not 1 <= fps <= 1000:
        raise ValueError('fps must be an integer from 1 to 1000')
    root = root.resolve(); out = out.resolve()
    if out == root or out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError('output and raw dataset must not contain one another')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'refusing to overwrite non-empty output: {out}')
    subsets = [('Movements', 1), ('Incorrect Movements', 0)] if whole_trial else [
        ('Segmented Movements', 1), ('Incorrect Segmented Movements', 0)]
    files = []
    for subset, label in subsets:
        folder = root / subset / 'Vicon' / 'Positions'
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for path in sorted(folder.glob('*.txt')):
            parsed = parse_name(path.stem)
            if movement == 'all' or parsed['movement_id'] == movement:
                files.append((path, subset, label, parsed))
    if not files:
        raise ValueError('no matching Vicon files')
    out.mkdir(parents=True, exist_ok=True)
    report = {'input_root': str(root), 'output_root': str(out), 'input_files': len(files),
              'converted': 0, 'skipped': [], 'counts': {}, 'repaired_clips': 0,
              'mapping': [list(names) for names in JOINT_MARKERS], 'version': 2,
              'source_fps': SOURCE_FPS, 'output_fps': fps,
              'sampling_method': 'timestamp_linear_interpolation'}
    for path, subset, label, parsed in files:
        try:
            markers, repair = repair_markers(read_positions(path), max_gap)
            joints, fallback = map_h36m(markers)
            points, geometry = canonical_coordinates(joints)
            points, sampling = resample_frames(points, fps)
            image = project_2d(points, view)
        except (ValueError, OSError) as exc:
            report['skipped'].append({'source_file': str(path.relative_to(root)), 'reason': str(exc)})
            continue
        clip_id = f"{parsed['movement_id']}_{parsed['subject_id']}_{parsed['episode_id'] or 'full'}_{'good' if label else 'bad'}"
        split = SUBJECT_SPLIT[parsed['subject_id']]
        clip = out / split / clip_id
        clip.mkdir(parents=True, exist_ok=False)
        for name, array in [('kp3d', points), ('kp2d', image)]:
            np.savez_compressed(clip / f'{name}.npz', keypoints=array, fps=np.int32(fps))
        meta = {'clip_id': clip_id, 'action_id': MOVEMENT_NAMES[parsed['movement_id']],
                'movement_id': parsed['movement_id'], 'subject_id': parsed['subject_id'],
                'episode_id': parsed['episode_id'], 'label': label, 'label_name': 'good' if label else 'bad',
                'fps': fps, 'source_fps': SOURCE_FPS, 'resampling': sampling,
                'num_frames': len(points), 'start_frame': 0, 'end_frame': len(points)-1,
                'error_type': 'non_optimal' if not label else None,
                'severity': None, 'has_2d': True, 'has_3d': True, 'joint_format': 'h36m_17',
                'units_3d': 'metres', 'up_axis': 'Y', 'coordinate_system': 'anatomical_right_y_up',
                'view': view, 'derived_2d': True, 'coordinates_2d': 'normalized_0_1',
                'projection': f'virtual_{view}_orthographic', 'source': 'UI-PRMD',
                'source_system': 'Vicon', 'source_file': str(path.relative_to(root)),
                'label_source': 'dataset_folder_correct_vs_incorrect', 'conversion_version': 2,
                'quality': {**repair, 'averaged_marker_fallback_frames': fallback, **geometry}}
        (clip / 'meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
        (clip / 'remark.txt').write_text(
            f"UI-PRMD / {subset} / {path.name}\nVicon optical landmarks approximate H36M-17 joints.\n"
            f"3D: rigid rotation, metres, Y-up; 2D: synthetic {view} projection, not camera observations.\n"
            f"Sampling: {SOURCE_FPS} -> {fps} FPS, timestamp linear interpolation; no anti-alias filter.\n"
            f"Missing marker samples: {repair['missing_samples']}; interpolated: {repair['interpolated_samples']}.\n",
            encoding='utf-8')
        report['converted'] += 1
        report['repaired_clips'] += int(repair['missing_samples'] > 0)
        key = f"{parsed['movement_id']}/{split}/{label}"
        report['counts'][key] = report['counts'].get(key, 0) + 1
        if report['converted'] % 200 == 0:
            print(f"Converted {report['converted']} / {len(files)}", flush=True)
    (out / 'conversion_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(f"Converted {report['converted']}; skipped {len(report['skipped'])}; output: {out}", flush=True)
    return report


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="将 UI-PRMD 的 Vicon 数据转换为 H36M-17 资源包（Y-up 3D + 派生 2D）。所有参数均需显式指定。",
        epilog=("示例：\n"
                "  uv run scripts/convert_ui_prmd.py --root dataset-raw/UI-PRMD --out dataset-ui-prmd-new --movement all --segmented --view front --max-gap 10 --fps 30\n"
                "输出目录必须为空，不覆盖已有结果。"),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True,
                        metavar='目录', help='原始 UI-PRMD 数据目录')
    parser.add_argument('--out', type=Path, required=True,
                        metavar='目录', help='输出目录，必须为空')
    parser.add_argument('--movement', choices=['all', *MOVEMENT_NAMES], required=True,
                        help='动作编号，m01 为深蹲；all 表示全部动作')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--segmented', action='store_const', const=False, dest='whole_trial',
                      help='使用已切分的重复动作')
    mode.add_argument('--whole-trial', action='store_const', const=True, dest='whole_trial',
                      help='使用整段 trial（与 --segmented 二选一）')
    parser.add_argument('--view', choices=['front', 'side'], required=True,
                        help='派生 2D 的视角：front 正面，side 侧面')
    parser.add_argument('--max-gap', type=int, required=True, metavar='帧数',
                        help='允许插值的内部缺口长度，0 禁用（100 FPS）')
    parser.add_argument('--fps', type=int, required=True, metavar='帧率',
                        help='输出帧率，1–1000；原始为 100 FPS，按时间戳线性插值')
    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()
    if not 1 <= args.fps <= 1000:
        parser.error('--fps 必须为 1–1000 的整数')
    if args.max_gap < 0:
        parser.error('--max-gap 必须为非负整数')
    if args.out.is_file() or (args.out.is_dir() and any(args.out.iterdir())):
        parser.exit(2, f"输出目录已存在且非空：{args.out.resolve()}\n"
                       "已有结果未修改。请用 --out 指定一个新的空目录，例如：\n"
                       "  --out dataset-ui-prmd-new\n"
                       "查看参数：uv run scripts/convert_ui_prmd.py --help\n")
    try:
        report = convert(args.root, args.out, args.movement, args.whole_trial, args.view, args.max_gap, args.fps)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"转换失败：{exc}\n查看用法：uv run scripts/convert_ui_prmd.py --help\n")
    if not report['converted']:
        parser.exit(1, '没有可用片段，请查看输出目录内的 conversion_report.json\n')


if __name__ == '__main__':
    main()
