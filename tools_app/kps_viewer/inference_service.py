"""Model inference and per-viewer SSE sessions. No display coordinates are used."""
from __future__ import annotations

import asyncio
import json
import math
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

WINDOW_FRAMES = 64


def trailing_window(frames: np.ndarray, current: int) -> tuple[np.ndarray, dict]:
    current = min(max(current, 0), len(frames) - 1)
    start = max(0, current - WINDOW_FRAMES + 1)
    actual = frames[start:current + 1]
    padding = WINDOW_FRAMES - len(actual)
    padded = np.concatenate([actual, np.repeat(actual[-1:], padding, axis=0)], axis=0)
    return padded, {'start_frame': start, 'end_frame': current,
                    'observed_frames': len(actual), 'padding_frames': padding}


class AnalyzeRequest(BaseModel):
    model: str
    source: str
    path: str
    mode: Literal['full', 'window'] = 'full'
    threshold: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    time: float = Field(default=0, ge=0, allow_inf_nan=False)


class CursorRequest(BaseModel):
    time: float = Field(ge=0, allow_inf_nan=False)
    sequence: int = Field(ge=0)
    generation: int = Field(default=0, ge=0)


class Predictor:
    def __init__(self, model_dir: Path, locate_package, locate_file, read_array, read_fps):
        self.model_dir = model_dir.resolve()
        self.locate_package = locate_package
        self.locate_file = locate_file
        self.read_array = read_array; self.read_fps = read_fps
        self.lock = threading.Lock()
        self.cache = OrderedDict()

    def model_path(self, name: str) -> Path:
        path = (self.model_dir / name).resolve()
        if Path(name).name != name or path.parent != self.model_dir or path.suffix.lower() != '.pt':
            raise ValueError('模型必须是 out-models 下的 .pt 文件')
        if not path.is_file():
            raise ValueError('模型文件不存在')
        return path

    def load(self, params: AnalyzeRequest):
        # Loading and prediction are serialized to bound CPU/memory consumption.
        with self.lock:
            import torch
            from skeleton_demo.model import BinarySkeletonModel
            torch.set_num_threads(min(4, torch.get_num_threads()))
            path = self.model_path(params.model)
            stat = path.stat(); key = (str(path), stat.st_mtime_ns, stat.st_size)
            if key not in self.cache:
                checkpoint = torch.load(path, map_location='cpu', weights_only=True)
                if not isinstance(checkpoint, dict) or not {'model_state', 'config'} <= checkpoint.keys():
                    raise ValueError('不兼容的模型：需要 model_state 和 config')
                config = checkpoint['config']
                target_len = int(config['data'].get('target_len', 64))
                if not 1 <= target_len <= 2048:
                    raise ValueError('不支持的模型输入长度')
                model = BinarySkeletonModel(**config['model'])
                model.load_state_dict(checkpoint['model_state'], strict=True)
                model.eval()
                self.cache[key] = (model, target_len)
                while len(self.cache) > 3:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(key)
            model, target_len = self.cache[key]
            self.locate_package(params.path, params.source)
            tracks = {}
            for slot, dim in [('2d', 2), ('3d', 3)]:
                file = self.locate_file(f'{params.path}/kp{slot}.npz', params.source)
                if not file.is_file():
                    continue
                with np.load(file, allow_pickle=False) as data:
                    array, _ = self.read_array(data)
                    array = np.asarray(array, dtype=np.float32)
                    if array.ndim == 4 and len(array) == 1:
                        array = array[0]
                    if array.ndim != 3 or array.shape[1] != 17 or len(array) == 0:
                        raise ValueError(f'kp{slot}.npz 必须是 [T,17,D]')
                    if array.shape[2] != dim and not (slot == '2d' and array.shape[2] == 3):
                        raise ValueError(f'kp{slot}.npz 的坐标维数错误')
                    array = array[..., :dim].copy()
                    if not np.isfinite(array).all():
                        raise ValueError(f'kp{slot}.npz 有非有限坐标，拒绝推理')
                    tracks[slot] = {'frames': array, 'fps': self.read_fps(data)}
            if not tracks:
                raise ValueError('数据包中没有可用骨骼')
            if len(tracks) == 2 and (tracks['2d']['fps'] != tracks['3d']['fps'] or
                                     len(tracks['2d']['frames']) != len(tracks['3d']['frames'])):
                raise ValueError('推理需要 2D/3D 的 FPS 与帧数一致')
            return {'model': model, 'target_len': target_len, 'tracks': tracks}

    def predict(self, loaded, params: AnalyzeRequest, seconds: float, sequence: int):
        with self.lock:
            import torch
            from skeleton_demo.infer import prepare
            started = time.perf_counter()
            tracks = loaded['tracks']; primary = tracks.get('3d', tracks.get('2d'))
            fps = primary['fps']; count = len(primary['frames'])
            current = min(int(seconds * fps + 1e-7), count - 1)
            bounds = {'start_frame': 0, 'end_frame': count - 1,
                      'observed_frames': count, 'padding_frames': 0}
            tensors = []
            for slot, dim in [('2d', 2), ('3d', 3)]:
                array = tracks[slot]['frames'] if slot in tracks else None
                if array is not None and params.mode == 'window':
                    array, bounds = trailing_window(array, current)
                tensor = torch.from_numpy(array) if array is not None else None
                tensors.append(prepare(tensor, loaded['target_len'], dim).unsqueeze(0))
            with torch.inference_mode():
                logits = loaded['model'](*tensors, torch.tensor([float('2d' in tracks)]),
                                         torch.tensor([float('3d' in tracks)]))
                probability = float(torch.sigmoid(logits).item())
            if not math.isfinite(probability):
                raise ValueError('模型输出不是有限数值')
            return {'model': params.model, 'source': params.source, 'path': params.path,
                    'mode': params.mode, 'sequence': sequence, 'probability': probability,
                    'decision': int(probability >= params.threshold), 'threshold': params.threshold,
                    'requested_time': seconds, 'current_frame': current, 'fps': fps,
                    'model_input_frames': loaded['target_len'], 'window_frames': WINDOW_FRAMES,
                    'has2d': '2d' in tracks, 'has3d': '3d' in tracks,
                    'elapsed_ms': round((time.perf_counter() - started) * 1000, 1), **bounds}


class Session:
    def __init__(self, predictor, params):
        self.id = uuid.uuid4().hex
        self.params = params; self.predictor = predictor
        self.time = params.time; self.sequence = 0; self.generation = 0; self.version = 0
        self.queue = asyncio.Queue(maxsize=4); self.wake = asyncio.Event()
        self.last_seen = time.monotonic(); self.streaming = False; self.closed = False
        self.task = asyncio.create_task(self.run())

    def emit(self, event, data):
        if self.queue.full():
            self.queue.get_nowait()
        self.queue.put_nowait((event, data))

    async def run(self):
        try:
            self.emit('status', {'message': '正在加载模型和骨骼…'})
            loaded = await asyncio.to_thread(self.predictor.load, self.params)
            self.wake.set()
            while True:
                await self.wake.wait(); self.wake.clear()
                generation, seconds, sequence = self.generation, self.time, self.sequence
                result = await asyncio.to_thread(self.predictor.predict, loaded, self.params, seconds, sequence)
                result['generation'] = generation
                if generation == self.generation:
                    self.emit('result', result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            message = exc.detail if isinstance(exc, HTTPException) else str(exc)
            self.emit('analysis_error', {'message': message})

    def close(self):
        if not self.closed:
            self.closed = True; self.task.cancel()
            self.emit('closed', {'message': '分析结束'})


def install_analysis(app, model_dir, locate_package, locate_file, read_array, read_fps):
    predictor = Predictor(model_dir, locate_package, locate_file, read_array, read_fps)
    sessions = {}
    app.state.analysis_sessions = sessions

    @app.get('/api/models')
    def models():
        root = predictor.model_dir
        return {'root': str(root), 'models': [{'name': p.name, 'bytes': p.stat().st_size}
                for p in sorted(root.glob('*.pt')) if p.is_file() and not p.is_symlink()]}

    @app.post('/api/analysis')
    async def start(params: AnalyzeRequest):
        # Validate paths before allocating a session. Checkpoint compatibility is reported via SSE.
        try:
            predictor.model_path(params.model)
            locate_package(params.path, params.source)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        for key, session in list(sessions.items()):
            if time.monotonic() - session.last_seen > 90:
                session.close(); sessions.pop(key, None)
        if len(sessions) >= 16:
            raise HTTPException(429, '分析会话过多，请关闭闲置页面')
        session = Session(predictor, params); sessions[session.id] = session
        return {'id': session.id, 'events': f'/api/analysis/{session.id}/events'}

    @app.post('/api/analysis/{sid}/cursor')
    async def cursor(sid: str, params: CursorRequest):
        session = sessions.get(sid)
        if session is None:
            raise HTTPException(404, '分析会话已结束')
        session.last_seen = time.monotonic()
        if session.params.mode == 'window' and params.sequence > session.sequence:
            session.sequence = params.sequence; session.time = params.time; session.version += 1
            session.generation = params.generation
            if session.params.mode == 'window':
                session.wake.set()
        return {'sequence': session.sequence}

    @app.delete('/api/analysis/{sid}')
    async def stop(sid: str):
        session = sessions.pop(sid, None)
        if session: session.close()
        return {'stopped': True}

    @app.get('/api/analysis/{sid}/events')
    async def events(sid: str, request: Request):
        session = sessions.get(sid)
        if session is None or session.streaming:
            raise HTTPException(404, '分析会话不存在或已连接')
        session.streaming = True
        async def stream():
            try:
                while not session.closed and not await request.is_disconnected():
                    session.last_seen = time.monotonic()
                    try:
                        event, data = await asyncio.wait_for(session.queue.get(), timeout=15)
                        yield f'event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'
                        if event in ('analysis_error', 'closed'): break
                    except asyncio.TimeoutError:
                        yield ': heartbeat\n\n'
            finally:
                session.close(); sessions.pop(sid, None)
        return StreamingResponse(stream(), media_type='text/event-stream',
                                 headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    @app.on_event('shutdown')
    async def close_all():
        tasks = [session.task for session in sessions.values()]
        for session in sessions.values(): session.close()
        sessions.clear()
        if tasks: await asyncio.gather(*tasks, return_exceptions=True)
