"""Generate toy kinematic squat pairs, NOT expert-labelled human motion.

Y is up, Z is forward, units are metres. Equal-length two-link leg IK
keeps ankles planted and bones constant; a rigid torso cannot model lumbar
curvature. Labels express this generator's rules only, not clinical truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]


def sequence(frames, scale, width, bottom, lean, descent):
    t = np.linspace(0, 1, frames)
    phase = np.where(t <= descent, t / descent, (1 - t) / (1 - descent))
    depth = 0.5 - 0.5 * np.cos(np.pi * phase)
    x = np.zeros((frames, 17, 3), dtype=np.float64)
    x[:, 0, 1] = 0.84 - (0.84 - bottom) * depth
    x[:, 0, 2] = -0.25 * depth
    for side, hip, knee, ankle in [(-1, 1, 2, 3), (1, 4, 5, 6)]:
        x[:, hip] = x[:, 0] + [side * width, 0, 0]
        x[:, ankle] = [side * width, 0, 0]
        delta = x[:, hip] - x[:, ankle]
        distance = np.linalg.norm(delta, axis=-1)
        if np.any(distance >= 0.86):
            raise ValueError('Leg target is unreachable')
        perpendicular = np.stack([np.zeros(frames), -delta[:, 2], delta[:, 1]], -1)
        perpendicular /= distance[:, None]
        height = np.sqrt(0.43 ** 2 - (distance / 2) ** 2)
        x[:, knee] = (x[:, hip] + x[:, ankle]) / 2 + height[:, None] * perpendicular
    angle = 0.06 + lean * depth
    up = np.stack([np.zeros(frames), np.cos(angle), np.sin(angle)], -1)
    for child, parent, length in [(7, 0, .18), (8, 7, .22), (9, 8, .10), (10, 9, .13)]:
        x[:, child] = x[:, parent] + length * up
    # Arms held forward, with a fixed elbow bend.
    for side, shoulder, elbow, wrist in [(1, 11, 12, 13), (-1, 14, 15, 16)]:
        x[:, shoulder] = x[:, 8] + [side * .18, 0, 0]
        x[:, elbow] = x[:, shoulder] + [0, -.07, .26]
        x[:, wrist] = x[:, elbow] + [0, .04, .24]
    return (scale * x).astype(np.float32)


def project(x, yaw):
    # Perspective camera looking along +Z; return normalized image coordinates.
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    camera = x @ rotation.T
    camera += [0, -.85, 4]
    return np.stack([.5 + 1.2 * camera[..., 0] / camera[..., 2],
                     .5 - 1.2 * camera[..., 1] / camera[..., 2]], -1).astype(np.float32)


def generate(out: Path, seed: int):
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite existing dataset: {out}')
    rng = np.random.default_rng(seed)
    counts = {'train': 0, 'val': 0, 'test': 0}
    max_bone_drift = 0.0
    for subject in range(10):
        split = 'train' if subject < 6 else ('val' if subject < 8 else 'test')
        scale, width = rng.uniform(.88, 1.12), rng.uniform(.11, .16)
        for take in range(3):
            frames = int(rng.integers(75, 121))
            descent = float(rng.uniform(.42, .60))
            yaw = float(rng.uniform(-1.15, 1.15))
            family = f's{subject:02d}_take{take:02d}'
            variants = [
                ('good_a', 1, float(rng.uniform(.34, .39)), float(rng.uniform(.25, .42))),
                ('good_b', 1, float(rng.uniform(.34, .39)), float(rng.uniform(.25, .42))),
                ('insufficient_depth', 0, float(rng.uniform(.61, .69)), float(rng.uniform(.25, .42))),
                ('excessive_trunk_lean', 0, float(rng.uniform(.34, .39)), float(rng.uniform(.94, 1.08))),
            ]
            for variant, label, bottom, lean in variants:
                x3d = sequence(frames, scale, width, bottom, lean, descent)
                x2d = project(x3d, yaw)
                bones = np.stack([np.linalg.norm(x3d[:, j] - x3d[:, p], axis=-1)
                                  for j, p in enumerate(PARENTS) if p >= 0], -1)
                drift = float(np.max(np.ptp(bones, axis=0)))
                max_bone_drift = max(max_bone_drift, drift)
                assert drift < 1e-5
                assert np.isfinite(x3d).all() and np.isfinite(x2d).all()
                assert np.max(np.abs(x3d[:, [3, 6], 1])) == 0
                clip_id = family + '_' + variant
                folder = out / split / clip_id
                folder.mkdir(parents=True)
                np.savez_compressed(folder / 'kp3d.npz', keypoints=x3d, fps=np.int32(30))
                np.savez_compressed(folder / 'kp2d.npz', keypoints=x2d, fps=np.int32(30))
                meta = dict(clip_id=clip_id, action_id='squat', label=label,
                            label_name='good' if label else 'bad', subject_id=f's{subject:02d}',
                            source='synthetic_kinematic_v1', label_source='generator_rules_not_expert',
                            source_motion_id=family, synthetic=True, seed=seed,
                            error_type=None if label else variant, fps=30, num_frames=frames,
                            joint_format='h36m_17', has_2d=True, has_3d=True,
                            view=f'yaw_{yaw:.4f}', units_3d='metres', up_axis='Y',
                            coordinates_2d='normalized_image_xy_down',
                            parameters=dict(scale=float(scale), half_hip_width=float(width),
                                            bottom_hip_height=bottom, trunk_lean_amplitude=lean,
                                            descent_fraction=descent, camera_yaw=yaw))
                (folder / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
                counts[split] += 1
    report = dict(seed=seed, counts=counts, max_bone_length_drift_m=max_bone_drift,
                  checks=['finite coordinates', 'constant 3D bone lengths', 'planted ankles'],
                  limitations=['No dynamics, balance or collision validation',
                               'No lumbar curvature, toes, tracking noise or real human variation',
                               'Labels are procedural assumptions, not expert judgements',
                               'Synthetic split performance does not measure real-world accuracy'])
    (out / 'validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('runs/kinematic_squat_v1/raw'))
    parser.add_argument('--seed', type=int, default=20261012)
    args = parser.parse_args()
    generate(args.out, args.seed)
