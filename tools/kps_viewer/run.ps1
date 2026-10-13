#!/usr/bin/env pwsh
# 启动骨骼动画查看器（run.sh 的 Windows/PowerShell 版本）。
#
#   tools/kps_viewer/run.ps1                   # 默认监听 127.0.0.1:8000，仅本机可访问
#   tools/kps_viewer/run.ps1 --port 8080       # 换端口
#   tools/kps_viewer/run.ps1 --host 0.0.0.0    # 局域网可访问
#
# 若提示禁止运行脚本，改用：
#   pwsh -ExecutionPolicy Bypass -File tools/kps_viewer/run.ps1
#
# 浏览器里打开打印出的地址，选择 dataset 数据包即可查看动画。

$ErrorActionPreference = 'Stop'

$Here = $PSScriptRoot                          # tools/kps_viewer/
$Root = Split-Path -Parent (Split-Path -Parent $Here)               # 仓库根
$Py = Join-Path $Root '.venv/Scripts/python.exe'   # Windows 布局是 Scripts 而非 bin
$Viewer = Join-Path $Here 'kps_viewer.py'

Set-Location $Root

if (-not (Test-Path -LiteralPath $Py -PathType Leaf)) {
    Write-Warning "未找到 $Py，回退到 uv run（首次会同步依赖）"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Error '未找到 uv，无法回退安装依赖，请先安装 https://docs.astral.sh/uv/' -ErrorAction Continue
        exit 127
    }
    & uv run python $Viewer @args
    exit $LASTEXITCODE
}

# 检查查看器依赖（发行包名，非 import 名）。缺包时 python 会把 traceback 写到 stderr，
# 这里临时关掉 $ErrorActionPreference 与原生命令的 stderr 语义，只取退出码判断。
$Check = @'
import importlib.metadata as metadata

for name in ("fastapi", "uvicorn", "python-multipart"):
    metadata.version(name)
'@

$Eap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$Check | & $Py - 2>$null
$DepsOk = $LASTEXITCODE -eq 0
$ErrorActionPreference = $Eap

if (-not $DepsOk) {
    Write-Warning '缺少 fastapi / uvicorn / python-multipart，执行 uv sync 安装…'
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Error "依赖缺失且未找到 uv，请手动执行：$Py -m pip install fastapi uvicorn python-multipart" -ErrorAction Continue
        exit 127
    }
    & uv sync
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

& $Py $Viewer @args
exit $LASTEXITCODE
