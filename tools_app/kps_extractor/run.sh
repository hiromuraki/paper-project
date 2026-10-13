#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export UV_PROJECT_ENVIRONMENT="$ROOT/.venv"
export PYTHONUTF8=1
exec uv run --project "$ROOT" --locked "$ROOT/web_app.py" "$@"
