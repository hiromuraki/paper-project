#!/usr/bin/env bash
# 启动骨骼动画查看器。
#
#   tools/kps_viewer/run.sh                   # 默认监听 127.0.0.1:8000，仅本机可访问
#   tools/kps_viewer/run.sh --port 8080       # 换端口
#   tools/kps_viewer/run.sh --host 0.0.0.0    # 局域网可访问
#
# 浏览器里打开打印出的地址，选择 dataset 数据包即可查看动画。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$(dirname "$HERE")")"
PY="$ROOT/.venv/bin/python"

cd "$ROOT"

if [[ ! -x "$PY" ]]; then
    echo "未找到 $PY，回退到 uv run（首次会同步依赖）" >&2
    exec uv run python "$HERE/kps_viewer.py" -dataset-root "$ROOT" "$@"
fi

# 检查查看器依赖（发行包名，非 import 名）
if ! "$PY" - <<'PYCHECK'
import importlib.metadata as metadata

for name in ("fastapi", "uvicorn", "python-multipart"):
    metadata.version(name)
PYCHECK
then
    echo "缺少 fastapi / uvicorn / python-multipart，执行 uv sync 安装…" >&2
    uv sync
fi

exec "$PY" "$HERE/kps_viewer.py" -dataset-root "$ROOT" "$@"
