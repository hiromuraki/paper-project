import asyncio
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException, UploadFile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import web_app as web
spec = importlib.util.spec_from_file_location("extract_pipeline", ROOT / "src/main.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class FakeProcess:
    def __init__(self, code, lines):
        self.code = code
        self.stdout = io.StringIO(lines)
    def wait(self, **kwargs):
        return self.code
    def poll(self):
        return self.code


class ExtractorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.root = Path(self.temp.name)
        self.weight = self.root / "model.pth"
        self.weight.touch()
        self.patchers = [patch.object(web, "OUTPUTS", self.root / "outputs"),
                         patch.object(web, "WEIGHT", self.weight)]
        for p in self.patchers:
            p.start()
        web.JOBS.clear()
        web.ACTIVE = None
        web.STOPPING = False

    def tearDown(self):
        for p in self.patchers:
            p.stop()
        web.JOBS.clear()
        web.ACTIVE = None
        self.temp.cleanup()

    def upload(self, data=b"video", name="测试 视频.mp4"):
        with patch.object(web, "Thread"):
            return asyncio.run(web.upload(UploadFile(filename=name, file=io.BytesIO(data)), "cpu"))["id"]

    def make_track(self, job_id, slot):
        suffix = "2d_h36m" if slot == "2d" else "3d"
        file = web.JOBS[job_id]["folder"] / ("video_" + suffix + "_kps.npz")
        np.savez_compressed(file, keypoints=np.ones((3, 17, 3)), fps=29.97, width=640, height=480)
        return file

    def test_upload_paths_and_busy(self):
        job_id = self.upload(name="../../evil.mp4")
        self.assertEqual(web.JOBS[job_id]["video"].name, "video.mp4")
        with self.assertRaises(HTTPException) as caught:
            self.upload()
        self.assertEqual(caught.exception.status_code, 409)

    def test_upload_limits_and_recovery(self):
        for data, name, code in [(b"", "clip.mp4", 400), (b"x", "clip.txt", 400)]:
            with self.assertRaises(HTTPException) as caught:
                self.upload(data, name)
            self.assertEqual(caught.exception.status_code, code)
            self.assertIsNone(web.ACTIVE)
        with patch.object(web, "MAX_UPLOAD", 2):
            with self.assertRaises(HTTPException) as caught:
                self.upload(b"123")
            self.assertEqual(caught.exception.status_code, 413)
        self.assertIsNone(web.ACTIVE)
        self.upload()

    def test_missing_weight(self):
        self.weight.unlink()
        with self.assertRaises(HTTPException) as caught:
            self.upload()
        self.assertEqual(caught.exception.status_code, 503)

    def test_worker_success_and_download_boundaries(self):
        job_id = self.upload()
        self.make_track(job_id, "2d")
        self.make_track(job_id, "3d")
        with patch.object(web.subprocess, "Popen", return_value=FakeProcess(0, "STAGE:2d\nSTAGE:3d\nSTAGE:done\n")) as spawn:
            web.run_job(job_id)
        self.assertEqual(spawn.call_args.args[0][0], sys.executable)
        self.assertEqual(web.status(job_id)["status"], "done")
        self.assertIsNone(web.ACTIVE)
        self.assertAlmostEqual(web.track(job_id, "2d")["fps"], 29.97)
        response = web.archive(job_id)
        import zipfile
        with zipfile.ZipFile(response.path) as bundle:
            self.assertIn("kp2d.npz", bundle.namelist())
            self.assertIn("kp3d.npz", bundle.namelist())
        with self.assertRaises(HTTPException):
            web.download(job_id, "../../model.pth")
        with self.assertRaises(HTTPException):
            web.status("unknown")

    def test_worker_error_retains_2d(self):
        job_id = self.upload()
        self.make_track(job_id, "2d")
        with patch.object(web.subprocess, "Popen", return_value=FakeProcess(1, "STAGE:3d\nCUDA error\n")):
            web.run_job(job_id)
        self.assertEqual(web.status(job_id)["status"], "error")
        self.assertIn("CUDA error", web.status(job_id)["logs"])
        self.assertEqual(len(web.track(job_id, "2d")["frames"]), 3)
        with self.assertRaises(HTTPException):
            web.track(job_id, "3d")
        self.assertIsNone(web.ACTIVE)

    def test_worker_progress_and_sse_reconnect(self):
        job_id = self.upload()
        records = [dict(stage="2d", current=2, total=8, unit="帧", message="检测 2D"),
                   dict(stage="3d", current=1, total=1, unit="窗口", message="重建 3D")]
        lines = "\n".join("KPS_PROGRESS:" + json.dumps(r, ensure_ascii=False) for r in records)
        with patch.object(web.subprocess, "Popen", return_value=FakeProcess(0, lines)) as spawn:
            web.run_job(job_id)
        self.assertEqual(spawn.call_args.kwargs["env"]["KPS_PROGRESS"], "1")
        self.assertEqual(web.status(job_id)["progress"], records[-1])
        self.assertNotIn("KPS_PROGRESS:", web.status(job_id)["logs"])
        self.assertIsNone(web.parse_progress('KPS_PROGRESS:{"stage":"2d","current":-1}'))

        class Request:
            async def is_disconnected(self):
                return False

        async def collect():
            # A reconnect after completion receives the complete final snapshot, then closes.
            response = await web.events(job_id, Request())
            chunks = [chunk async for chunk in response.body_iterator]
            self.assertEqual(response.media_type, "text/event-stream")
            self.assertIn("retry:", chunks[0])
            payload = json.loads(chunks[1].split("data: ", 1)[1])
            self.assertEqual(payload["status"], "done")
            self.assertEqual(payload["progress"]["current"], 1)
            self.assertGreater(payload["revision"], 0)
        asyncio.run(collect())

    def test_sse_live_updates_error_and_disconnect(self):
        job_id = self.upload()
        class Request:
            disconnected = False
            async def is_disconnected(self):
                return self.disconnected
        request = Request()

        async def exercise():
            response = await web.events(job_id, request)
            iterator = response.body_iterator
            await iterator.__anext__()  # reconnect interval
            first = await iterator.__anext__()
            self.assertIn('"status": "queued"', first)
            job = web.JOBS[job_id]
            with web.LOCK:
                job.update(status="running", progress=dict(stage="2d", current=3, total=8), revision=1)
            second = await iterator.__anext__()
            self.assertIn('"current": 3', second)
            with web.LOCK:
                job.update(status="error", error="model failed", revision=2)
            final = await iterator.__anext__()
            self.assertIn('"error": "model failed"', final)
            with self.assertRaises(StopAsyncIteration):
                await iterator.__anext__()
            response = await web.events(job_id, request)
            request.disconnected = True
            self.assertEqual([chunk async for chunk in response.body_iterator], ["retry: 1500\n\n"])
        asyncio.run(exercise())

    def test_short_and_long_3d_progress_and_pelvis_origin(self):
        import torch
        from core.MHFormerInferencer import MHFormerInferencer
        model = MHFormerInferencer.__new__(MHFormerInferencer)
        model.device = torch.device("cpu")
        model.window, model.stride = 27, 10
        model.joints_left, model.joints_right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
        # Deliberately offset all model coordinates; both paths must still produce a zero pelvis.
        def prediction(input_tensor):
            result = torch.ones((2, 27, 17, 3)) * torch.tensor([10., 20., 30.])
            result[:, :, :, 1] += torch.arange(17)
            return result
        model.model = prediction
        for length in (8, 70):
            with patch.object(sys.modules[MHFormerInferencer.__module__], "report_progress") as report:
                points = model.run_3d_keypoints_inference(np.ones((length, 17, 3)), 640, 480)
            self.assertEqual(points.shape, (length, 17, 3))
            np.testing.assert_array_equal(points[:, 0], 0)
            self.assertEqual(report.call_args.args[1], report.call_args.args[2])
            self.assertEqual(report.call_args.args[3], "窗口")

    def test_pipeline_path_output_and_metadata(self):
        import cv2
        source = self.root / "输入 视频.mp4"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 29.97, (64, 48))
        for _ in range(3):
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
        writer.release()
        # Real conversion/serialization; inference alone is mocked.
        src = ROOT / "src"
        sys.path.insert(0, str(src))
        from common import DataConverter, Serializer, VideoUtils
        core = types.ModuleType("core")
        class Pose:
            def __init__(self, **kwargs):
                pass
            def run_2d_keypoints_inference(self, video):
                return np.ones((1, 3, 17, 3), dtype=np.float32)
        class Lift:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            def run_3d_keypoints_inference(self, points, **kwargs):
                return np.zeros((len(points), 17, 3), dtype=np.float32)
        core.RTMPoseInferencer, core.MHFormerInferencer = Pose, Lift
        with patch.dict(sys.modules, {"core": core}):
            output = self.root / "骨架 输出"
            paths = pipeline.extract_3d_keypoints(source, output, "cpu", self.weight)
            self.assertTrue(all(p.parent == output for p in paths.values()))
            with np.load(paths["3d"]) as data:
                self.assertAlmostEqual(float(data["fps"]), 29.97, places=2)
                self.assertEqual(data["keypoints"].shape, (3, 17, 3))
            with patch.object(Pose, "run_2d_keypoints_inference", return_value=np.zeros((0, 3, 17, 3))):
                with self.assertRaisesRegex(ValueError, "未检测"):
                    pipeline.extract_3d_keypoints(source, output, "cpu", self.weight)


if __name__ == "__main__":
    unittest.main()
