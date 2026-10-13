$ErrorActionPreference = 'Stop'
$previousEnv = $env:UV_PROJECT_ENVIRONMENT
$previousEncoding = $env:PYTHONUTF8
try {
    $env:UV_PROJECT_ENVIRONMENT = Join-Path $PSScriptRoot '.venv'
    $env:PYTHONUTF8 = '1'
    & uv run --project $PSScriptRoot --locked (Join-Path $PSScriptRoot 'web_app.py') @args
    $result = $LASTEXITCODE
} finally {
    $env:UV_PROJECT_ENVIRONMENT = $previousEnv
    $env:PYTHONUTF8 = $previousEncoding
}
exit $result
