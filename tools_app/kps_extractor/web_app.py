"""Local upload UI. Run via run.ps1 / run.sh in this independent uv project."""
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock, Thread
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
import zipfile

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Form, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
WEIGHT = ROOT / "src/MHFormer/checkpoint/pretrained/351/model_4294.pth"
MAX_UPLOAD = 512 * 1024 * 1024
JOBS = {}
LOCK = Lock()
ACTIVE = None
STOPPING = False
BONES = [[i, p] for i, p in enumerate([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]) if p >= 0]


@asynccontextmanager
async def lifespan(app):
    global STOPPING
    STOPPING = False
    yield
    with LOCK:
        STOPPING = True
        for job in JOBS.values():
            process = job.get("process")
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()


app = FastAPI(title="KPS Extractor", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")


def job_for(job_id):
    with LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在（重启服务后请重新上传）")
    return job


def files_for(job):
    return {p.name: p for p in job["folder"].glob("*_kps.npz")}


def parse_progress(line):
    """Parse explicit worker records, without guessing progress from tqdm text."""
    if not line.startswith("KPS_PROGRESS:"):
        return None
    try:
        data = json.loads(line.split(":", 1)[1])
        if data["stage"] not in ("initializing", "init_2d", "2d", "save_2d", "init_3d", "3d", "save_3d", "done"):
            return None
        current, total = data.get("current"), data.get("total")
        if current is not None and (type(current) is not int or current < 0):
            return None
        if total is not None and (type(total) is not int or total < 0):
            return None
        return dict(stage=data["stage"], current=current, total=total,
                    unit=str(data.get("unit", "")), message=str(data.get("message", "")))
    except (ValueError, KeyError, TypeError):
        return None


def run_job(job_id):
    global ACTIVE
    job = job_for(job_id)
    try:
        env = os.environ.copy()
        env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", KPS_PROGRESS="1")
        with LOCK:
            if STOPPING:
                raise RuntimeError("服务正在关闭")
            process = subprocess.Popen(
                [sys.executable, "-u", str(ROOT / "main.py"), "-i", str(job["video"]),
                 "-o", str(job["folder"]), "--device", job["device"]],
                cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
            job["process"] = process
            job["status"] = "running"
            job["revision"] += 1
        with (job["folder"] / "extract.log").open("w", encoding="utf-8") as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                progress = parse_progress(line)
                with LOCK:
                    if progress:
                        job["progress"] = progress
                        job["stage"] = progress["stage"]
                    else:
                        job["logs"] = (job["logs"] + line)[-16000:]
                    if line.startswith("STAGE:"):
                        job["stage"] = line.strip().split(":", 1)[1]
                    job["revision"] += 1
        process.stdout.close()
        code = process.wait()
        with LOCK:
            job["status"] = "done" if code == 0 else "error"
            if code:
                job["error"] = "提取失败，退出码 {}，详见下方日志。".format(code)
            job["finished_at"] = time.time()
            job["revision"] += 1
    except Exception as exc:
        with LOCK:
            job["status"], job["error"] = "error", str(exc)
            job["finished_at"] = time.time()
            job["revision"] += 1
    finally:
        with LOCK:
            job.pop("process", None)
            ACTIVE = None


@app.get("/")
def index():
    return FileResponse(ROOT / "web/index.html")


@app.get("/api/health")
def health():
    return {"weight_ready": WEIGHT.is_file(), "max_upload_mb": MAX_UPLOAD // (1024 * 1024)}


@app.post("/api/jobs", status_code=202)
async def upload(file: UploadFile = File(...), device: str = Form("auto")):
    global ACTIVE
    extension = Path(file.filename or "").suffix.lower()
    if extension not in (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"):
        raise HTTPException(400, "支持 MP4 / MOV / AVI / MKV / WebM / M4V")
    if device not in ("auto", "cpu", "cuda:0"):
        raise HTTPException(400, "无效的设备")
    if not WEIGHT.is_file():
        raise HTTPException(503, "缺少 MHFormer 权重: {}".format(WEIGHT))
    job_id = uuid.uuid4().hex
    with LOCK:
        if ACTIVE:
            raise HTTPException(409, "当前已有提取任务，请完成后再上传")
        ACTIVE = job_id
    folder = OUTPUTS / job_id
    video = folder / ("video" + extension)
    try:
        folder.mkdir(parents=True)
        size = 0
        with video.open("wb") as stream:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(413, "视频超过 512 MB")
                stream.write(chunk)
        if not size:
            raise HTTPException(400, "视频为空")
        job = dict(folder=folder, video=video, name=file.filename, device=device,
                   status="queued", stage="initializing", logs="", error=None,
                   progress=None, revision=0, created_at=time.time(), finished_at=None)
        with LOCK:
            JOBS[job_id] = job
        Thread(target=run_job, args=(job_id,), daemon=True).start()
        return {"id": job_id}
    except Exception:
        video.unlink(missing_ok=True)
        with LOCK:
            JOBS.pop(job_id, None)
            ACTIVE = None
        raise
    finally:
        await file.close()


@app.get("/api/jobs/{job_id}")
def status(job_id: str):
    job = job_for(job_id)
    with LOCK:
        result = {k: job[k] for k in ("name", "status", "stage", "logs", "error")}
        result["progress"] = job.get("progress")
        result["revision"] = job.get("revision", 0)
        result["elapsed"] = round((job.get("finished_at") or time.time()) - job.get("created_at", time.time()), 1)
    result["files"] = list(files_for(job))
    return result


@app.get("/api/jobs/{job_id}/events")
async def events(job_id: str, request: Request):
    job_for(job_id)  # Return a normal 404 before starting the stream.

    async def stream():
        revision = -1
        last_ping = time.monotonic()
        yield "retry: 1500\n\n"
        while not await request.is_disconnected():
            snapshot = status(job_id)
            finished = snapshot["status"] in ("done", "error")
            if snapshot["revision"] != revision or finished:
                revision = snapshot["revision"]
                yield "id: {}\nevent: progress\ndata: {}\n\n".format(
                    revision, json.dumps(snapshot, ensure_ascii=False))
                last_ping = time.monotonic()
            if finished:
                return
            if time.monotonic() - last_ping >= 15:
                yield ": keepalive\n\n"
                last_ping = time.monotonic()
            await asyncio.sleep(0.2)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/jobs/{job_id}/video")
def video(job_id: str):
    return FileResponse(job_for(job_id)["video"])


@app.get("/api/jobs/{job_id}/tracks/{slot}")
def track(job_id: str, slot: str):
    suffixes = {"2d": "2d_h36m", "3d": "3d"}
    if slot not in suffixes:
        raise HTTPException(400, "slot 必须是 2d 或 3d")
    job = job_for(job_id)
    file = job["folder"] / ("video_" + suffixes[slot] + "_kps.npz")
    if not file.is_file():
        raise HTTPException(404, "此骨架尚未生成")
    with np.load(file, allow_pickle=False) as data:
        frames = np.asarray(data["keypoints"], dtype=np.float64)
        if frames.ndim != 3 or frames.shape[1:] != (17, 3) or not len(frames):
            raise HTTPException(422, "骨架格式无效")
        if not np.isfinite(frames).all():
            raise HTTPException(422, "骨架包含非有限坐标")
        return {"frames": frames.tolist(), "fps": float(data["fps"]),
                "width": int(data["width"]), "height": int(data["height"]), "bones": BONES}


@app.get("/api/jobs/{job_id}/files/{name}")
def download(job_id: str, name: str):
    file = files_for(job_for(job_id)).get(name)
    if not file:
        raise HTTPException(404, "文件不存在")
    return FileResponse(file, filename=name)


@app.get("/api/jobs/{job_id}/archive")
def archive(job_id: str):
    job = job_for(job_id)
    if job["status"] not in ("done", "error"):
        raise HTTPException(409, "请等待提取结束")
    files = files_for(job)
    if not files:
        raise HTTPException(404, "没有可下载的骨架")
    target = job["folder"] / "skeletons.zip"
    with LOCK:
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as bundle:
            for name, path in files.items():
                bundle.write(path, name)
            # Aliases can be dropped directly into a kps_viewer package.
            for source, alias in (("video_2d_h36m_kps.npz", "kp2d.npz"), ("video_3d_kps.npz", "kp3d.npz")):
                if source in files:
                    bundle.write(files[source], alias)
    return FileResponse(target, filename="skeletons.zip")


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="KPS 提取器 Web 界面")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    print("KPS Extractor: http://{}:{}".format(args.host, args.port), flush=True)
    uvicorn.run(app, host=args.host, port=args.port)
