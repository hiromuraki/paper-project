# KPS Extractor — Windows / Linux 本地骨架提取

视频 → RTMPose-l / RTMDet-m → COCO-17、H36M 2D → MHFormer 351 帧模型 → H36M 3D。

## 独立环境

本目录具有自己的 `pyproject.toml`、`uv.lock` 和 `.venv`，不使用主项目的 Python / Torch。
Windows x64 使用 Python 3.10、Torch 2.0.0 CUDA 11.8 和 MMCV 2.1.0 的 Windows 预编译 wheel；Linux x64 使用对应 Linux wheel。CUDA 不可用时自动退回 CPU（较慢）。不支持 macOS / ARM。

安装 uv 后，在项目根目录执行：

```powershell
# 创建或同步提取器专用环境（首次会下载较大的依赖）
uv sync --project .\tools_app\kps_extractor --locked
```

MMCV wheel 来自 OpenMMLab 官方 cu118 / torch2.0.0 目录，避免在 Windows 本地编译 CUDA 扩展。启动脚本显式固定 UV_PROJECT_ENVIRONMENT，主项目即使已激活虚拟环境也不会被使用或修改。

准备 MHFormer 权重：将 `model_4294.pth` 放到
`src/MHFormer/checkpoint/pretrained/351/model_4294.pth`（相对于本工具目录，约 120 MB）。
RTMPose / RTMDet 的权重由 MMPose 在首次推理时联网下载，默认缓存到本工具的 `.model-cache/`，避免依赖 Windows 用户目录的写权限。设置 `TORCH_HOME` 可覆盖该位置。

模型资料：[模型资料.md](./模型资料.md)。

## Web 界面

```powershell
# 从项目根目录启动
.\tools_app\kps_extractor\run.ps1
# 如果 PowerShell 执行策略限制：
powershell -ExecutionPolicy Bypass -File .\tools_app\kps_extractor\run.ps1
# 自定义端口
.\tools_app\kps_extractor\run.ps1 --port 8081
```

打开 http://127.0.0.1:8001 。Linux：`bash tools_app/kps_extractor/run.sh`。
也可以从任意工作目录使用启动脚本的绝对路径。

1. 选择视频（MP4 / MOV / AVI / MKV / WebM / M4V，最大 512 MB）。已选卡片高亮显示文件名、大小、视频尺寸和时长；视频立即通过本地 Blob URL 预览，无需上传或生成。确认后选择自动 / CPU / CUDA，点击「一键生成骨架」。更换视频会清除旧骨架和下载链接。
2. 上传显示实际字节进度；生成通过 SSE 推送当前阶段、2D 已处理帧数 / 总帧数、3D 已完成窗口 / 总窗口数、本阶段百分比及已用时。加载依赖 / 模型、下载权重和保存文件时总量未知，显示进行中，不伪造总进度。连接中断自动重连，重连收到最新完整状态；任务完成或失败后关闭推送。完整日志保存在任务目录的 `extract.log`。
3. 分别显示原视频、H36M 2D 和 H36M 3D。2D 完成后即可查看，3D 失败时保留 2D 结果。
4. 统一播放 / 暂停、逐帧、拖动时间轴、调速。空格播放，左右键逐帧。3D 可拖动旋转、滚轮缩放和重置视角。
5. 单独下载三个 NPZ 或全部 ZIP。ZIP 另附 `kp2d.npz`（H36M）和 `kp3d.npz` 别名；将其和原视频（命名 `video.mp4`）放入 `dataset-*/<分类>/<包>/`，即可使用 `kps_viewer`。

Three.js / OrbitControls 随工具提供，无 CDN 依赖。原视频预览需要浏览器支持视频编码；推荐 H.264 MP4。无法在浏览器播放的视频仍可提取、查看骨架。网页的 3D 显示逐帧减去骨盆位置，并将相机坐标的 y/z 方向翻转；骨盆固定在 (0, 0, 0)，参考网格经过原点。NPZ 保留相机方向的骨盆相对坐标。

SSE 接口：`GET /api/jobs/{id}/events`，事件名为 `progress`，包含任务状态、阶段、`progress.current / total / unit / message`、`elapsed`、日志和可下载文件；`id` 为任务修订号。无更新时发送心跳。`GET /api/jobs/{id}` 仍可读取当前状态。

一次只允许一个提取任务，重复上传返回忙碌提示。推理使用当前独立 Python 的子进程，不阻塞页面服务；关闭服务会终止推理。任务输出位于 `outputs/<任务ID>/`，不会修改原视频。重启后不恢复网页任务会话，但文件仍保留；需要时手动清理 `outputs/`。默认仅监听本机，显式 `--host 0.0.0.0` 可供局域网访问（服务没有身份认证）。

## 命令行

```powershell
.\tools_app\kps_extractor\extract3dkps.ps1 -i "D:\videos\运动.mp4" -o "D:\骨架输出"
.\tools_app\kps_extractor\extract3dkps.ps1 -i "D:\videos\运动.mp4" --device cpu
# 支持自定义 MHFormer 351 帧权重
.\tools_app\kps_extractor\extract3dkps.ps1 -i "D:\videos\运动.mp4" --weight "D:\models\model_4294.pth"
```

Linux：`bash tools_app/kps_extractor/extract3dkps.sh -i video.mp4 -o output`。
相对输入 / 输出路径以调用时的工作目录解析；权重以脚本目录解析，不依赖 `cd src`。

| 输出 | keypoints 形状 | 通道含义 |
| --- | --- | --- |
| `{视频名}_2d_coco17_kps.npz` | `(Persons, Frames, 17, 3)` | x、y、置信度 |
| `{视频名}_2d_h36m_kps.npz` | `(Frames, 17, 3)` | x、y、置信度 |
| `{视频名}_3d_kps.npz` | `(Frames, 17, 3)` | 相机坐标 x、y、z，骨盆为原点 |

每个 NPZ 还包含 `fps`（保留小数）、`width`、`height`；默认输出至视频同目录。Web 上传使用安全固定名 `video.*`，因此结果名以 `video_` 开头。

沿用原管线的单人策略：H36M / 3D 取每帧第一个人物，没有跨帧身份跟踪。多人、遮挡、漏检可能影响结果。未检测到人物时报告明确错误。短视频用末帧补齐 351 帧推理后裁回原长；短视频和长视频均保留骨盆相对坐标，不再按短视频单独调整 z。

## 验证

```powershell
uv run --project .\tools_app\kps_extractor --locked python -m unittest discover -s .\tools_app\kps_extractor -p "test_*.py"
```

测试使用合成数据和模拟模型，覆盖输出路径、FPS、任务成功 / 失败与部分结果、上传限制及下载边界；真实模型运行依赖本机驱动、模型权重和可解码视频。

本机验证（2026-10-13）：Python 3.10.20 / RTX 4050 / Torch 2.0.0+cu118 / MMCV 2.1.0；9 项回归测试通过（含 SSE 实时更新、重连、失败结束和长短视频骨盆归零）。命令行与网页上传均完成真实 8 帧、30 FPS 示例提取，生成三份有限坐标的 NPZ，3D 骨盆坐标为零；网页逐帧和 ZIP 下载通过。MPEG-4 Part 2 测试视频无法在当前浏览器播放时，页面显示编码提示，骨架仍可正常预览；建议使用 H.264 MP4。
