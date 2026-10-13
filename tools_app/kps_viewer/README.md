# KPS Viewer

自动发现指定根目录下的 `dataset/` 和全部 `dataset-*/` 目录并读取数据包，不需要手动上传文件。

```powershell
# Windows（项目根目录执行）
.\tools\kps_viewer\run.ps1
# 如执行策略限制脚本：
powershell -ExecutionPolicy Bypass -File .\tools\kps_viewer\run.ps1
```

```bash
# Linux / macOS（项目根目录执行）
bash tools/kps_viewer/run.sh
```

默认打开地址为 http://127.0.0.1:8000。两个脚本均支持 `--port 8080`、`--host 0.0.0.0`，也可从其他工作目录用脚本的绝对路径启动。

- 左侧按树形展示 `dataset → 分类 → 资源包`，例如 `dataset-mock/test/clip_bad_2001`。把 dataset 算作第一级，固定第三级目录为资源包；空包也列出，不扫描更深目录。分类可展开和折叠，目录树只显示到包级。
- 点击包后按槽位分别解析两份骨骼，自动连接同目录的 `video.mp4`。切包清空旧数据并从第 0 帧自动播放；视频默认静音以支持自动播放。缺少任一文件时仍可查看其他文件，损坏文件会显示原因。
- 搜索匹配相对路径；上一包 / 下一包沿当前搜索结果切换；新增或删除数据包后点击刷新。
- 底部统一播放、调速、拖动时间轴、逐帧前后移动或输入帧编号跳转。帧编号从 0 开始。空格播放 / 暂停，左右方向键逐帧（编辑输入框时不触发快捷键）。
- 两个视图右上角的设置菜单提供坐标模式、Y 翻转与 3D 显示选项。3D 支持拖动旋转、滚轮缩放、重置、选关节查看原始与视图坐标。
- 右侧信息面板的 Remark / Meta 标签显示 `remark.txt`（兼容拼写 `remakr.txt`）和 `meta.json` 原文，不格式化 JSON。缺少文本不会阻止播放。
- 各轨道使用自己的 NPZ FPS（缺少时为 30）按时间同步。帧数或 FPS 不一致时提示；较短骨骼序列停在末帧。视频需要浏览器支持其编码。

Three.js 和 OrbitControls 随工具放在 `vendor/`，无外网依赖。

后端检查：`.venv/Scripts/python.exe -m unittest discover -s tools/kps_viewer -p 'test_*.py'`

界面：左侧目录树，中间双视图与播放条，右侧概况 / Remark / Meta 信息面板。顶部“信息面板”按钮可收起或打开侧栏，窄窗口默认收起。

进度条支持点击任意位置、拖动和键盘方向键定位；所有手动定位都会暂停播放。顶部“亮色模式 / 暗色模式”切换界面与骨骼视图区配色，并记住选择。

左侧数据源下拉框动态列出 `dataset`、`dataset-mock`、`dataset-ui-prmd` 等目录，首次优先模拟数据，之后记住选择。所有数据源都采用“根目录 / 分类 / 资源包”结构。切换数据源会清空旧包并停止播放，选择新包后自动播放。

左右侧栏与主视口之间的竖向手柄可拖动调整宽度比例，刷新后保留。双击手柄或按 Home 恢复该侧默认宽度；聚焦手柄后左右方向键微调，Shift 加快。窄屏信息面板浮动或竖向布局时隐藏对应手柄。


`-dataset-root` 指定包含这些数据源目录的父目录（也接受 `--dataset-root`）。两个启动脚本默认使用自身位置的 `../../`，先解析为绝对路径，默认就是项目根目录；用户传入的参数可以覆盖默认值。

```powershell
.\tools\kps_viewer\run.ps1 -dataset-root "D:\motion-data" --port 8080
# 直接启动 Python：
.\.venv\Scripts\python.exe tools/kps_viewer/kps_viewer.py -dataset-root "D:\motion-data"
```

```bash
bash tools/kps_viewer/run.sh -dataset-root /path/to/motion-data --port 8080
```

点击刷新会重新发现新增或删除的数据源目录。名称必须是 `dataset` 或以 `dataset-` 开头的直接子目录；目录中的分类 / 资源包层级保持不变。`dataset-raw` 也会按名称规则列出，但其原始文件需先转换成查看器使用的 NPZ 数据包。


## 判别模型与 SSE 分析

将本项目训练得到的 checkpoint `.pt` 放入项目 `out-models/`。在右侧“概况 → 模型分析”中选择模型；新模型可点击“刷新模型”。只有包含 `model_state`、`config` 且与 BinarySkeletonModel 兼容的 checkpoint 可以加载，训练数据 PT 不可作为模型。

- 整段分析：一次读取整个资源包，沿用训练 / infer 的根对齐、尺度归一化、模型配置的 target_len 重采样和坐标 / 骨段 / 速度特征；播放位置不改变整段结果。
- 滑动窗口：截至当前帧取最近 64 个原始帧；不足 64 时重复当前帧补在末尾。如 `[frame0, frame1, current, current, ...]`，不读取未来帧。然后使用相同的模型预处理。窗口按帧计：30 FPS 下约 2.1 秒，100 FPS 下约 0.63 秒；模型配置若并非 64 帧，输入还会按其 target_len 重采样。
- 窗口位置每 250 ms 检查一次，播放、暂停定位和逐帧操作均可更新。后端合并待处理位置，仅计算最新位置；推理慢时显示最新完成结果，界面标出其帧范围。定位、回放循环、切包或切模型后，旧位置 / 旧会话的结果不会覆盖新结果。
- 判别按正类概率与可调阈值比较（初始 0.5），显示正类(1) / 负类(0)、概率、使用的帧范围、尾部 padding 数、输入模态与耗时。类别含义以模型训练标签为准；当前 best.pt 的 1 是深蹲良好、0 是非最优，不是具体错误诊断。概率未做校准。
- 原始 NPZ 输入用于推理，视图旋转、翻转和相机操作不影响判别。单一 2D 或 3D 模态可用；同时存在时要求帧数 / FPS 一致。坏坐标和不兼容模型明确提示错误。

“完整 trial”是一次采集记录中的连续多个重复动作，整段使用一个标签；当前模型用这种样本训练。短窗口（尤其大量尾部 padding）与该训练范围不同，滑动窗口判别用于探索，不等同于经过验证的逐帧质量标签。

推理在 CPU 上运行，兼容模型按文件修改时间缓存（最多 3 个）。SSE 仅负责推送状态、结果和错误；位置由 HTTP POST 上报。每个页面独立会话，关闭分析 / 切包 / 页面关闭会释放会话。

接口：`GET /api/models`；`POST /api/analysis` 创建会话；`GET /api/analysis/{id}/events` 接收 SSE（status/result/analysis_error）；`POST /api/analysis/{id}/cursor` 更新窗口位置；`DELETE /api/analysis/{id}` 停止会话。
