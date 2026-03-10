#!/usr/bin/env bash
set -euo pipefail

# 兼容用户可能使用的历史脚本名：sam2_segment_apiu.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${ROOT}/sam2_segment_api.sh"
