#!/bin/bash
# 一键切换"答疑大脑"：online（DeepSeek API，质量优先）/ offline（本地 Qwen2.5-7B-AWQ，断网可用）
#
# 用法：
#   bash deploy/cloud/switch_mode.sh online            # 切在线并重启主应用
#   bash deploy/cloud/switch_mode.sh offline           # 切离线（自动拉起本地 LLM）并重启
#   bash deploy/cloud/switch_mode.sh online --no-restart
#
# 说明：语音识别（耳朵）已永久本地化（FunASR/SenseVoiceSmall），本脚本只切"大脑"。
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENVF="$R/.env"
LOG_DIR="$R/deploy/cloud/logs"
MODE="${1:-}"
NO_RESTART="${2:-}"
LOCAL_LLM_NAME_ENV="Qwen2.5-7B-AWQ"

set_env() {  # $1=key $2=value
  local key="$1" val="$2"
  if grep -q "^${key}=" "$ENVF" 2>/dev/null; then
    sed -i "s|^${key}=.*|${key}=${val}|" "$ENVF"
  else
    printf '%s=%s\n' "$key" "$val" >> "$ENVF"
  fi
}

apply_fragment() {  # 把片段文件里的 KEY=VALUE 逐行并入 .env
  local f="$1"
  [ -f "$f" ] || {
    echo "[mode] 缺少片段文件：$f"
    echo "[mode] 含密钥的片段文件不入库，请先 cp ${f}.example $f 并填入自己的密钥"
    exit 1
  }
  while IFS='=' read -r k v; do
    case "$k" in ''|'#'*) continue ;; esac
    set_env "$k" "$v"
  done < "$f"
}

restart_app() {
  pkill -f 'server\.app' 2>/dev/null
  sleep 2
  cd "$R" || exit 1
  nohup "$R/.venvs/app/bin/python" -m server.app >> "$LOG_DIR/app.log" 2>&1 &
  echo "[mode] 主应用已重启（约 10s 后就绪；日志 logs/app.log）"
}

ensure_local_llm() {
  if pgrep -f llm_engine_local.py >/dev/null 2>&1; then
    echo "[mode] 本地 LLM 引擎已在运行"
    return 0
  fi
  echo "[mode] 启动本地 LLM 引擎（首次加载约 1~2 分钟）..."
  cd "$R" || exit 1
  local llm_dir="${LOCAL_LLM_DIR:-}"
  if [ -z "$llm_dir" ]; then
    llm_dir="$(find "$R/models_zoo" -maxdepth 4 -type d -name '*Qwen2.5-7B-AWQ*' 2>/dev/null | head -1)"
  fi
  [ -z "$llm_dir" ] && { echo "[mode] 未找到本地模型目录，请检查 models_zoo/"; return 1; }
  echo "[mode] 模型目录：$llm_dir"
  LOCAL_LLM_DIR="$llm_dir" LOCAL_LLM_NAME="$LOCAL_LLM_NAME_ENV" ENGINE_PORT=8100 \
    setsid nohup "$R/.venvs/llm/bin/python" scripts/llm_engine_local.py \
    >> "$LOG_DIR/local_llm.log" 2>&1 < /dev/null &
  local i
  for i in $(seq 1 48); do
    sleep 5
    if curl -sf -m 3 http://127.0.0.1:8100/health >/dev/null 2>&1; then
      echo "[mode] 本地 LLM 就绪（约 $((i * 5))s）"
      return 0
    fi
  done
  echo "[mode] 本地 LLM 未就绪，请查看 logs/local_llm.log"
  return 1
}

case "$MODE" in
  online)
    apply_fragment "$SCRIPT_DIR/mode_online.env"
    echo "[mode] 已切到【在线模式】：答疑 = DeepSeek API（需要外网）"
    [ "$NO_RESTART" = "--no-restart" ] || restart_app
    ;;
  offline)
    ensure_local_llm
    apply_fragment "$SCRIPT_DIR/mode_offline.env"
    echo "[mode] 已切到【离线模式】：答疑 = 本地 Qwen2.5-7B-AWQ（断网可用）"
    [ "$NO_RESTART" = "--no-restart" ] || restart_app
    ;;
  *)
    echo "用法: bash deploy/cloud/switch_mode.sh online|offline [--no-restart]"
    echo "  online  —— 答疑走 DeepSeek API（质量优先，需外网）"
    echo "  offline —— 答疑走本地 Qwen2.5-7B-AWQ（断网可用，质量降一档）"
    exit 1
    ;;
esac

echo "[mode] 当前配置："
grep -E '^(LLM_PROVIDER|LLM_API_BASE|LLM_MODEL|ASR_PROVIDER|ASR_API_BASE|FUNASR_MODEL)=' "$ENVF" | sed 's/^/  /'