#!/usr/bin/env bash
# ============================================================================
# 算力云部署 · 停止全部后台服务
# 用法：bash deploy/cloud/stop_all.sh
# ============================================================================
set -uo pipefail

LOG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logs"
PID_FILE="$LOG_DIR/pids.txt"

if [ ! -f "$PID_FILE" ]; then
  echo "[stop] 未找到 $PID_FILE，可能没有通过 start_all.sh 启动的服务"
  exit 0
fi

while read -r pid; do
  [ -z "$pid" ] && continue
  if kill -0 "$pid" >/dev/null 2>&1; then
    echo "[stop] 结束进程 $pid"
    kill "$pid" >/dev/null 2>&1 || true
  fi
done < "$PID_FILE"

sleep 2
# 兜底：强制杀掉仍在的进程
while read -r pid; do
  [ -z "$pid" ] && continue
  kill -0 "$pid" >/dev/null 2>&1 && kill -9 "$pid" >/dev/null 2>&1 || true
done < "$PID_FILE"

: > "$PID_FILE"
echo "[stop] 完成。"
