"""Regression tests for coordinate, projection and missing-data semantics."""
from convert_ui_prmd import (INDEX, VICON_MARKERS, canonical_coordinates,
                             convert, map_h36m, project_2d, read_positions, repair_markers, resample_frames)
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def fixture():
    values = np.full((4, 39, 3), 10., dtype=float)
    positions = {
        'LASI': (-100, 150, 1000), 'RASI': (-100, -150, 1000),
        'LKNE': (-100, 180, 550), 'RKNE': (-100, -180, 550),
        'LANK': (-100, 180, 80), 'RANK': (-100, -180, 80),
        'T10': (-180, 0, 1250), 'C7': (-180, 0, 1550), 'STRN': (-50, 0, 1350),
        'LFHD': (-100, 60, 1750), 'RFHD': (-100, -60, 1750),
        'LBHD': (-220, 60, 1750), 'RBHD': (-220, -60, 1750),
        'LSHO': (-150, 230, 1500), 'RSHO': (-150, -230, 1500),
        'LELB': (-140, 400, 1300), 'RELB': (-140, -400, 1300),
        'LWRA': (-100, 400, 1050), 'LWRB': (-120, 400, 1050),
        'RWRA': (-100, -400, 1050), 'RWRB': (-120, -400, 1050),
    }
    for name, point in positions.items():
        values[:, INDEX[name]] = point
    values[1:, :, 0] += np.arange(1, 4)[:, None] * 10
    return values


class ConversionTests(unittest.TestCase):
    def test_y_up_rotation_preserves_geometry_and_root_motion(self):
        joints, _ = map_h36m(fixture())
        points, info = canonical_coordinates(joints)
        self.assertAlmostEqual(np.linalg.det(info['lab_to_canonical_rotation']), 1)
        np.testing.assert_allclose(np.linalg.norm(points[:, 2]-points[:, 3], axis=-1),
                                   np.linalg.norm(joints[:, 2]-joints[:, 3], axis=-1)/1000, rtol=1e-6)
        np.testing.assert_allclose(points[0, 0], 0, atol=1e-6)
        self.assertGreater(np.linalg.norm(points[-1, 0]), 0)
        self.assertGreater(points[0, 1, 0], points[0, 4, 0])
        self.assertGreater(points[0, 10, 1], points[0, 3, 1]+1)

    def test_front_projection_has_height_not_lab_top_view(self):
        joints, _ = map_h36m(fixture())
        points, _ = canonical_coordinates(joints)
        image = project_2d(points)
        self.assertEqual(image.shape, (4, 17, 2))
        self.assertGreater(image[0, 3, 1]-image[0, 10, 1], .7)
        self.assertTrue((image >= .049).all() and (image <= .951).all())
        np.testing.assert_allclose(image[0, 1, 1], image[0, 4, 1], atol=1e-6)

    def test_short_internal_gaps_only(self):
        markers = fixture()
        j = INDEX['LKNE']
        markers[1:3, j] = 0
        repaired, report = repair_markers(markers, max_gap=2)
        self.assertTrue(np.isfinite(repaired[:, j]).all())
        self.assertEqual(report['interpolated_samples'], 2)
        markers[0, j] = 0
        repaired, _ = repair_markers(markers, max_gap=2)
        self.assertTrue(np.isnan(repaired[:3, j]).all())
        with self.assertRaises(ValueError):
            map_h36m(repaired)

    def test_missing_redundant_marker_uses_available_member(self):
        markers = fixture()
        markers[:, INDEX['LWRB']] = 0
        repaired, _ = repair_markers(markers)
        joints, fallback = map_h36m(repaired)
        np.testing.assert_array_equal(joints[:, 13], markers[:, INDEX['LWRA']])
        self.assertEqual(fallback['13'], 4)

    def test_end_to_end_format_labels_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'raw'
            out = Path(temp)/'dataset-ui-prmd'
            for subset, suffix in [('Segmented Movements', ''), ('Incorrect Segmented Movements', '_inc')]:
                folder = root/subset/'Vicon/Positions'
                folder.mkdir(parents=True)
                np.savetxt(folder/f'm01_s01_e01_positions{suffix}.txt', fixture().reshape(4, 117), delimiter=',')
            report = convert(root, out, 'm01', False, 'front', 10, 100)
            self.assertEqual(report['converted'], 2)
            meta = json.loads((out/'train/m01_s01_e01_bad/meta.json').read_text())
            self.assertEqual(meta['label'], 0)
            self.assertEqual(meta['up_axis'], 'Y')
            self.assertTrue(meta['derived_2d'])
            with np.load(out/'train/m01_s01_e01_good/kp3d.npz', allow_pickle=False) as data:
                self.assertEqual(data['keypoints'].shape, (4, 17, 3))
                self.assertEqual(data['fps'], 100)
            with self.assertRaises(FileExistsError):
                convert(root, out, 'm01', False, 'front', 10, 100)
            with self.assertRaises(ValueError):
                convert(root, root/'output', 'm01', False, 'front', 10, 100)

    def test_cli_help_and_existing_output(self):
        script = Path(__file__).resolve().parents[1] / "scripts/convert_ui_prmd.py"
        help_result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, encoding="utf-8")
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("输出目录", help_result.stdout)
        self.assertIn("--movement", help_result.stdout)
        with tempfile.TemporaryDirectory() as temp:
            sentinel = Path(temp) / "keep.txt"
            sentinel.write_text("unchanged", encoding="utf-8")
            result = subprocess.run([sys.executable, str(script), "--root", temp, "--out", temp, "--movement", "all",
                                    "--segmented", "--view", "front", "--max-gap", "10", "--fps", "30"], capture_output=True, encoding="utf-8")
            self.assertEqual(result.returncode, 2)
            self.assertIn("--out", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertEqual(sentinel.read_text(), "unchanged")

    def test_cli_no_arguments_only_help_and_missing_arguments_rejected(self):
        script = Path(__file__).resolve().parents[1] / "scripts/convert_ui_prmd.py"
        result = subprocess.run([sys.executable, str(script)], capture_output=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0)
        self.assertIn("--root", result.stdout)
        self.assertNotIn("Converted", result.stdout)
        result = subprocess.run([sys.executable, str(script), "--movement", "m01"],
                                capture_output=True, encoding="utf-8")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--out", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_resampling_30fps_is_uniform_and_time_aligned(self):
        points = np.repeat(np.arange(101, dtype=np.float32)[:, None, None], 17, axis=1)
        points = np.repeat(points, 3, axis=2)
        sampled, info = resample_frames(points, 30)
        self.assertEqual(sampled.shape, (31, 17, 3))
        np.testing.assert_allclose(sampled[:, 0, 0], np.arange(31)*100/30, rtol=1e-6)
        self.assertEqual(info['discarded_tail_s'], 0)
        unchanged, _ = resample_frames(points, 100)
        np.testing.assert_array_equal(unchanged, points)
        high, _ = resample_frames(points, 200)
        self.assertEqual(high.shape[0], 201)
        self.assertAlmostEqual(float(high[1, 0, 0]), .5)
        for fps in (0, -1, 1001, 30.5):
            with self.assertRaises(ValueError):
                resample_frames(points, fps)

    def test_resampling_tail_is_not_extrapolated(self):
        sampled, info = resample_frames(np.zeros((100, 17, 3)), 30)
        self.assertEqual(len(sampled), 30)
        self.assertGreaterEqual(info['discarded_tail_s'], 0)
        self.assertLess(info['discarded_tail_s'], 1/30)
        self.assertLessEqual(info['output_last_timestamp_s'], info['source_last_timestamp_s'])

    def test_conversion_saves_matching_resampled_2d_and_3d(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'raw'
            out = Path(temp)/'output'
            for subset in ('Segmented Movements', 'Incorrect Segmented Movements'):
                folder = root/subset/'Vicon/Positions'
                folder.mkdir(parents=True)
            markers = np.repeat(fixture()[:1], 101, axis=0)
            markers[:, :, 0] += np.arange(101)[:, None]
            np.savetxt(root/'Segmented Movements/Vicon/Positions/m01_s01_e01_positions.txt',
                       markers.reshape(101, 117), delimiter=',')
            convert(root, out, 'm01', False, 'front', 10, 30)
            clip = out/'train/m01_s01_e01_good'
            with np.load(clip/'kp2d.npz') as a, np.load(clip/'kp3d.npz') as b:
                self.assertEqual(int(a['fps']), 30)
                self.assertEqual(int(b['fps']), 30)
                self.assertEqual(a['keypoints'].shape[0], 31)
                np.testing.assert_allclose(a['keypoints'], project_2d(b['keypoints']), atol=1e-7)
            meta = json.loads((clip/'meta.json').read_text())
            self.assertEqual(meta['fps'], 30)
            self.assertEqual(meta['num_frames'], 31)
            self.assertEqual(meta['resampling']['source_num_frames'], 101)

    def test_truncated_and_whitespace_files(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'positions.txt'
            np.savetxt(p, fixture().reshape(4, 117))
            self.assertEqual(read_positions(p).shape, (4, 39, 3))
            np.savetxt(p, fixture()[:1].reshape(1, 117), delimiter=',')
            with self.assertRaises(ValueError):
                read_positions(p)


if __name__ == '__main__':
    unittest.main()
