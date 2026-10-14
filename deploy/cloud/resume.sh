#!/usr/bin/env bash
# ============================================================================
# 算力云部署 · 一键恢复（幂等）
# 用途：关机/重启后，一条命令把「全栈服务 + 公网隧道」恢复，并输出新链接。
# 用法：bash deploy/cloud/resume.sh
# 说明：
#   * 已在运行则自动跳过（幂等，不会重复起）；
#   * 公网链接每次会变，本脚本会把最新链接写入 deploy/cloud/logs/public_url.txt 并打印。
# ============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"     # 项目根
cd "$ROOT"
LOG_DIR="$ROOT/deploy/cloud/logs"
mkdir -p "$LOG_DIR"

echo "[resume] 项目根：$ROOT"

# 1) 全栈服务（主应用已在监听则跳过）
if curl -s -m3 -o /dev/null http://127.0.0.1:8001/health; then
  echo "[resume] 主应用已在运行，跳过 start_all.sh"
else
  echo "[resume] 启动全栈服务（start_all.sh）..."
  bash deploy/cloud/start_all.sh
fi

# 2) 公网隧道（cloudflared 快速隧道）
CFD="$(command -v cfd 2>/dev/null || true)"
[ -z "$CFD" ] && [ -x /usr/local/bin/cfd ] && CFD=/usr/local/bin/cfd
if [ -n "$CFD" ]; then
  if pgrep -f 'tunnel --url http://127.0.0.1:8001' >/dev/null 2>&1; then
    echo "[resume] 隧道已在运行"
  else
    echo "[resume] 启动公网隧道 ..."
    setsid nohup "$CFD" tunnel --url http://127.0.0.1:8001 --no-autoupdate > "$LOG_DIR/cf.log" 2>&1 &
  fi
  sleep 8
  URL="$(grep -aoE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG_DIR/cf.log" | tail -1)"
  printf '%s\n' "$URL" > "$LOG_DIR/public_url.txt"
  echo "=========================================================="
  echo "  公网链接：${URL:-（未取到，请查看 $LOG_DIR/cf.log）}"
  echo "  已写入：$LOG_DIR/public_url.txt"
  echo "=========================================================="
else
  echo "[resume] 未找到 cloudflared(cfd)，跳过隧道；如需公网访问请安装后重试。"
fi
