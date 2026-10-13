"""Run: uv run python -m unittest discover -s scripts -p pack_dataset_tests.py"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pack_dataset import collect_split, pack
from skeleton_demo.data import SkeletonDataset, collate_fn
from skeleton_demo.infer import first_sequence


class PackTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / 'tmp' / 'pack_tests'
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=scratch)
        assert Path(self.tmp.name).resolve().is_relative_to(scratch.resolve())
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def clip(self, split, name, dims=(2, 3)):
        folder = self.root / 'raw' / split / name
        folder.mkdir(parents=True)
        meta = dict(label=1, clip_id=split + name, subject_id=split,
                    joint_format='h36m_17', custom_provenance='keep me')
        (folder / 'meta.json').write_text(json.dumps(meta))
        for dim in dims:
            x = np.arange(8*17*dim, dtype=np.float32).reshape(8,17,dim)
            np.savez(folder / f'kp{dim}d.npz', keypoints=x, fps=30)
        return folder

    def test_mixed_modalities_roundtrip(self):
        self.clip('train','a',(2,)); self.clip('train','b',(3,)); self.clip('train','c')
        self.clip('val','d')
        tables = pack(self.root/'raw', self.root/'out', 'squat')
        ds = SkeletonDataset(self.root/'out/squat_train.pt', target_len=12)
        batch = collate_fn([ds[i] for i in range(3)])
        self.assertEqual(batch['has2d'].tolist(),[1,0,1])
        self.assertEqual(batch['has3d'].tolist(),[0,1,1])
        self.assertEqual(tuple(batch['x3d'].shape),(3,12,17,9))
        self.assertEqual(tables['train']['metadata'][0]['custom_provenance'],'keep me')
        self.assertIsNone(first_sequence(tables['train']['x3d']))
        with self.assertRaises(FileExistsError):
            pack(self.root/'raw',self.root/'out','squat')

    def test_bad_coordinates_and_timing(self):
        folder = self.clip('train','a')
        for x in [np.zeros((7,17,3)), np.full((8,17,3),np.nan), np.zeros((8,16,3))]:
            with self.subTest(shape=x.shape):
                np.savez(folder/'kp3d.npz',keypoints=x)
                with self.assertRaises(ValueError): collect_split(self.root/'raw/train')

    def test_cross_split_subject_rejected_before_writes(self):
        self.clip('train','a'); folder=self.clip('val','b')
        p=folder/'meta.json'; m=json.loads(p.read_text()); m['subject_id']='train'
        p.write_text(json.dumps(m))
        with self.assertRaises(ValueError): pack(self.root/'raw',self.root/'out','squat')
        self.assertFalse((self.root/'out').exists())

    def test_old_whole_modality_missing(self):
        p=self.root/'old.pt'
        torch.save(dict(x2d=None,x3d=torch.ones(2,8,17,3),y=torch.tensor([0.,1.])),p)
        self.assertEqual(SkeletonDataset(p)[0]['has2d'].item(),0)
        torch.save(dict(x2d=None,x3d=[torch.ones(8,17,3)],y=torch.tensor([0.,1.])),p)
        with self.assertRaises(ValueError): SkeletonDataset(p)


if __name__ == '__main__':
    unittest.main()
