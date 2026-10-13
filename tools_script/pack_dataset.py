"""校验已划分的 NPZ 资源包，打包 PT 数据并生成训练配置。

帮助：uv run scripts/pack_dataset.py --help
输入：<数据集>/{train,val,test}/<资源包>/{meta.json,kp2d.npz,kp3d.npz}
标签与坐标保持原样，--action 只控制输出文件名前缀。
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def load_keypoints(path: Path, dim: int):
    if not path.exists():
        return None, None
    with np.load(path, allow_pickle=False) as z:
        if 'keypoints' not in z:
            raise ValueError(f'{path}: missing keypoints')
        x = z['keypoints']
        if x.ndim != 3 or x.shape[0] < 2 or x.shape[1:] != (17, dim):
            raise ValueError(f'{path}: expected [T>=2,17,{dim}], got {x.shape}')
        if not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
            raise ValueError(f'{path}: coordinates must be real numbers')
        x = x.astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError(f'{path}: nonfinite coordinates')
        fps = float(z['fps'].item()) if 'fps' in z else None
    if fps is not None and (not math.isfinite(fps) or fps <= 0):
        raise ValueError(f'{path}: invalid fps')
    return torch.from_numpy(x), fps


def collect_split(root: Path):
    folders = sorted(p for p in root.iterdir() if p.is_dir())
    if not folders:
        raise ValueError(f'{root}: no clip directories')
    table = {k: [] for k in ('x2d', 'x3d', 'y', 'clip_id', 'subject', 'view',
                             'error_type', 'action_id', 'has2d', 'has3d', 'fps', 'metadata')}
    for folder in folders:
        meta = json.loads((folder / 'meta.json').read_text(encoding='utf-8-sig'))
        if meta.get('label') not in (0, 1):
            raise ValueError(f'{folder}: explicit numeric label 0 or 1 required')
        if meta.get('joint_format', 'h36m_17') != 'h36m_17':
            raise ValueError(f'{folder}: expected h36m_17')
        x2, f2 = load_keypoints(folder / 'kp2d.npz', 2)
        x3, f3 = load_keypoints(folder / 'kp3d.npz', 3)
        if x2 is None and x3 is None:
            raise ValueError(f'{folder}: no skeleton')
        if x2 is not None and x3 is not None and len(x2) != len(x3):
            raise ValueError(f'{folder}: 2D/3D frame count mismatch')
        rates = [v for v in (f2, f3, meta.get('fps')) if v is not None]
        if rates and any(not math.isfinite(float(v)) or float(v) <= 0 or
                         not math.isclose(float(v), float(rates[0]), rel_tol=1e-5) for v in rates):
            raise ValueError(f'{folder}: inconsistent or invalid fps')
        frames = len(x2 if x2 is not None else x3)
        if 'num_frames' in meta and meta['num_frames'] != frames:
            raise ValueError(f'{folder}: num_frames mismatch')
        row = dict(x2d=x2, x3d=x3, y=float(meta['label']),
                   clip_id=str(meta.get('clip_id', folder.name)), subject=meta.get('subject_id'),
                   view=meta.get('view'), error_type=meta.get('error_type'),
                   action_id=meta.get('action_id'), has2d=x2 is not None, has3d=x3 is not None,
                   fps=float(rates[0]) if rates else None, metadata=meta)
        for key, value in row.items():
            table[key].append(value)
    table['y'] = torch.tensor(table['y'], dtype=torch.float32)
    table['schema_version'] = 2
    return table


def pack(dataset: Path, out: Path, action: str):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', action):
        raise ValueError('--action must be a simple filename prefix')
    if not dataset.is_dir():
        raise ValueError(f'Dataset not found: {dataset}')
    tables = {}
    for split in ('train', 'val', 'test'):
        path = dataset / split
        if path.is_dir():
            tables[split] = collect_split(path)
        elif split != 'test':
            raise ValueError(f'Missing {path}; split by subject before packaging')
    seen_ids, seen_sources, seen_subjects = {}, {}, {}
    for split, table in tables.items():
        for clip, subject, meta in zip(table['clip_id'], table['subject'], table['metadata']):
            if clip in seen_ids:
                raise ValueError(f'Duplicate clip_id: {clip}')
            seen_ids[clip] = split
            source = meta.get('source_motion_id')
            if source:
                if source in seen_sources and seen_sources[source] != split:
                    raise ValueError(f'Source motion crosses splits: {source}')
                seen_sources[source] = split
            if subject and subject != 'unknown':
                if subject in seen_subjects and seen_subjects[subject] != split:
                    raise ValueError(f'Subject crosses splits: {subject}')
                seen_subjects[subject] = split
    targets = [out / f'{action}_{s}.pt' for s in tables] + [out / 'train.yaml', out / 'summary.json']
    for path in targets:
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite {path}; choose another --out')
    base = Path(__file__).resolve().parents[1] / 'configs' / 'squat.yaml'
    cfg = yaml.safe_load(base.read_text(encoding='utf-8'))
    for split in tables:
        cfg['data'][split] = str((out / f'{action}_{split}.pt').resolve())
    cfg['train']['output_dir'] = str((out / 'checkpoints').resolve())
    summary = dict(source=str(dataset.resolve()), label_policy='Copied unchanged from meta.json',
                   preprocessing='float32 only; no normalization, resampling or augmentation', splits={})
    out.mkdir(parents=True, exist_ok=True)
    for split, table in tables.items():
        torch.save(table, out / f'{action}_{split}.pt')
        summary['splits'][split] = dict(samples=len(table['y']), positive=int(table['y'].sum()),
                                        has2d=sum(table['has2d']), has3d=sum(table['has3d']))
    (out / 'train.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'Config: {(out / "train.yaml").resolve()}')
    print('Labels copied unchanged. Old good/bad quality labels are NOT target-action labels.')
    return tables


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(
        description='校验 NPZ 资源包，打包为 PT，并生成 train.yaml 与 summary.json。',
        epilog=(
            '输入结构：<数据集>/{train,val,test}/<资源包>/，train、val 必须存在，test 可选。\n'
            '每个包需要 meta.json（label 为 0 或 1）及至少一种骨骼 NPZ（keypoints: [T>=2,17,2或3]）。\n'
            '校验坐标、帧数、FPS 及跨划分的重复样本/受试者；已有目标文件不会覆盖。\n'
            '仅转换坐标存储类型为 float32，不归一化、不重采样、不增广、不改写标签。\n\n'
            '示例：\n'
            '  uv run scripts/pack_dataset.py --dataset dataset-ui-prmd\n'
            '  uv run scripts/pack_dataset.py --dataset dataset-ui-prmd --out data/ui-prmd --action ui_prmd\n'
            '输出：<action>_train.pt、<action>_val.pt、可选 <action>_test.pt、train.yaml、summary.json。'),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', type=Path, required=True, metavar='目录',
                        help='输入数据集目录，直接包含 train/val（及可选 test）')
    parser.add_argument('--out', type=Path, metavar='目录',
                        help='输出目录；省略时使用 data/<输入数据集目录名>')
    parser.add_argument('--action', default='squat', metavar='前缀',
                        help='输出文件名前缀（默认：squat）；仅允许字母、数字、下划线或连字符，不筛选动作、不改标签')
    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()
    pack(args.dataset, args.out or Path('data') / args.dataset.resolve().name, args.action)


if __name__ == '__main__':
    main()
