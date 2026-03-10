#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON_BIN:-python}"
HOST="${SAM2_API_HOST:-0.0.0.0}"
PORT="${SAM2_API_PORT:-8000}"
RESTART_INTERVAL_SECONDS="${SAM2_API_RESTART_INTERVAL_SECONDS:-43200}"
READY_TIMEOUT_SECONDS="${SAM2_API_RESTART_READY_TIMEOUT_SECONDS:-180}"
OLD_PROCESS_GRACEFUL_SECONDS="${SAM2_API_OLD_PROCESS_GRACEFUL_SECONDS:-3600}"
TIMEOUT_GRACEFUL_SHUTDOWN="${SAM2_API_TIMEOUT_GRACEFUL_SHUTDOWN:-3600}"

# 每 12 小时滚动重启一次 uvicorn 子进程：
# - 监听 socket 由 supervisor 持有，重启期间连接不会被拒绝（最多在 backlog 中排队）
# - 新进程 ready 后再优雅退出旧进程，尽量不中断请求
exec "${PYTHON}" "${ROOT}/sam2_segment_api_supervisor.py" \
  --python "${PYTHON}" \
  --app "sam2_segment_api:app" \
  --app-dir "${ROOT}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --restart-interval-seconds "${RESTART_INTERVAL_SECONDS}" \
  --restart-ready-timeout-seconds "${READY_TIMEOUT_SECONDS}" \
  --old-process-graceful-seconds "${OLD_PROCESS_GRACEFUL_SECONDS}" \
  --timeout-graceful-shutdown "${TIMEOUT_GRACEFUL_SHUTDOWN}"
