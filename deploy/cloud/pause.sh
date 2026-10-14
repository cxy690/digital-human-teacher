#!/usr/bin/env bash
# ============================================================================
# 算力云部署 · 一键停止（停服务 + 停公网隧道），之后可放心"关机/停止"
# 用法：bash deploy/cloud/pause.sh
# ============================================================================
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "[pause] 停止公网隧道 ..."
pkill -f 'tunnel --url http://127.0.0.1:8001' >/dev/null 2>&1 && echo "[pause] 隧道已停" || echo "[pause] 隧道未在运行"

echo "[pause] 停止全栈服务（stop_all.sh）..."
bash deploy/cloud/stop_all.sh

# 兜底：停掉不是由 start_all.sh 启的实例（如手动重启过的主应用）
pkill -f 'serv[e]r.app' >/dev/null 2>&1 && echo "[pause] 主应用已停" || true

echo "[pause] 完成。现在可以去控制台点『关机/停止』（切勿点『释放/删除』）。"
