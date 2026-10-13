"""数据包发现、槽位解析、原文与访问边界的回归检查。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException
import kps_viewer as viewer


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / "dataset"
        self.root.mkdir()
        self.patch = patch.object(viewer, "DATASET_ROOT", self.base)
        self.patch.start()
        self.folder = self.root / "test" / "clip_bad_2001"
        self.folder.mkdir(parents=True)

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def test_discovery_and_semantic_slots(self):
        frames = np.ones((3, 17, 3), dtype=np.float32)
        frames[..., 2] = 0.8
        for slot in ("2d", "3d"):
            np.savez(self.folder / f"kp{slot}.npz", keypoints=frames, fps=25)
        self.assertEqual(viewer.packages()["packages"][0]["path"], "test/clip_bad_2001")
        result = viewer.package("test/clip_bad_2001")
        self.assertEqual(result["tracks"]["2d"]["frames"][0][0][2], 0)
        self.assertEqual(result["tracks"]["3d"]["frames"][0][0][2], 0.8)
        self.assertTrue(result["tracks"]["2d"]["has_confidence"])
        self.assertEqual(result["tracks"]["3d"]["fps"], 25)

    def test_corrupt_track_does_not_hide_other_files(self):
        (self.folder / "kp2d.npz").write_bytes(b"broken")
        np.savez(self.folder / "kp3d.npz", keypoints=np.zeros((2, 17, 3)))
        result = viewer.package("test/clip_bad_2001")
        self.assertIn("3d", result["tracks"])
        self.assertNotIn("2d", result["tracks"])
        self.assertTrue(result["errors"][0].startswith("kp2d.npz:"))

    def test_raw_text_and_optional_video(self):
        text = '<script>raw</script>\r\n  不改排版\r\n'
        (self.folder / "remakr.txt").write_bytes(text.encode("utf-8"))
        (self.folder / "meta.json").write_bytes(b'{\r\n  "label": 0\r\n}\r\n')
        (self.folder / "video.mp4").write_bytes(b"video fixture")
        result = viewer.package("test/clip_bad_2001")
        self.assertEqual(result["texts"]["remakr.txt"], text)
        self.assertIn("\r\n", result["texts"]["meta.json"])
        self.assertTrue(result["video"].startswith("/api/video?source=dataset&path="))
        self.assertEqual(viewer.package_video("test/clip_bad_2001").path, self.folder / "video.mp4")

    def test_missing_dataset_and_files(self):
        self.assertEqual(viewer.packages()["packages"][0]["files"], [])
        result = viewer.package("test/clip_bad_2001")
        self.assertEqual(result["tracks"], {})
        self.assertIsNone(result["video"])
        with patch.object(viewer, "DATASET_ROOT", self.base / "missing"):
            self.assertEqual(viewer.sources()["sources"], [])

    def test_fixed_depth_includes_empty_packages_but_not_nested_files(self):
        (self.folder / "nested").mkdir()
        np.savez(self.root / "test" / "kp2d.npz", keypoints=np.zeros((2, 17, 2)))
        np.savez(self.folder / "nested" / "kp3d.npz", keypoints=np.zeros((2, 17, 3)))
        result = viewer.packages()
        self.assertEqual(result["groups"], ["test"])
        self.assertEqual([item["path"] for item in result["packages"]], ["test/clip_bad_2001"])
        self.assertEqual(result["packages"][0]["files"], [])
        for path in ("test", "test/clip_bad_2001/nested"):
            with self.assertRaises(HTTPException) as error:
                viewer.package(path)
            self.assertEqual(error.exception.status_code, 400)

    def test_sources_are_isolated_and_video_keeps_source(self):
        mock_root = self.base / "dataset-mock"
        mock_folder = mock_root / "test" / "clip_bad_2001"
        mock_folder.mkdir(parents=True)
        (mock_folder / "remark.txt").write_text("mock", encoding="utf-8")
        (mock_folder / "video.mp4").write_bytes(b"fixture")
        (self.folder / "remark.txt").write_text("real", encoding="utf-8")
        self.assertEqual(viewer.packages("dataset-mock")["source"], "dataset-mock")
        self.assertEqual(viewer.package("test/clip_bad_2001", "dataset-mock")["texts"]["remark.txt"], "mock")
        self.assertEqual(viewer.package("test/clip_bad_2001")["texts"]["remark.txt"], "real")
        result = viewer.package("test/clip_bad_2001", "dataset-mock")
        self.assertIn("source=dataset-mock", result["video"])
        self.assertEqual(viewer.package_video("test/clip_bad_2001", "dataset-mock").path, mock_folder / "video.mp4")
        with self.assertRaises(HTTPException) as error:
            viewer.dataset_path("../test/clip_bad_2001", "dataset-mock")
        self.assertEqual(error.exception.status_code, 403)

    def test_source_discovery_and_custom_root(self):
        for name in ("dataset-mock", "dataset-ui-prmd", "dataset-raw", "not-a-dataset"):
            (self.base / name).mkdir()
        self.assertEqual(viewer.sources()["sources"], ["dataset", "dataset-mock", "dataset-raw", "dataset-ui-prmd"])
        self.assertEqual(viewer.dataset_root("dataset-ui-prmd"), self.base / "dataset-ui-prmd")
        for invalid in ("../dataset", "dataset/child", "dataset-mock/../dataset", "not-a-dataset"):
            with self.assertRaises(HTTPException) as error:
                viewer.packages(invalid)
            self.assertEqual(error.exception.status_code, 400)

    def test_video_source_name_is_url_encoded(self):
        from urllib.parse import parse_qs, urlparse
        source = "dataset-case & 测试"
        folder = self.base / source / "test" / "clip"
        folder.mkdir(parents=True)
        (folder / "video.mp4").write_bytes(b"fixture")
        url = viewer.package("test/clip", source)["video"]
        query = parse_qs(urlparse(url).query)
        self.assertEqual(query["source"], [source])
        self.assertEqual(query["path"], ["test/clip"])

    def test_unknown_source_rejected(self):
        with self.assertRaises(HTTPException) as error:
            viewer.packages("../dataset")
        self.assertEqual(error.exception.status_code, 400)

    def test_access_boundary_and_missing_package(self):
        for path in ("../outside", str(self.root.parent / "outside")):
            with self.assertRaises(HTTPException) as error:
                viewer.dataset_path(path)
            self.assertEqual(error.exception.status_code, 403)
        with self.assertRaises(HTTPException) as error:
            viewer.package("test/missing")
        self.assertEqual(error.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
