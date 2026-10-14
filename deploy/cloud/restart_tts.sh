#!/usr/bin/env bash
# 重启 CosyVoice 克隆 TTS 引擎（2/3 通用，fp16 关闭）。
#
# 与 restart_app.sh 同样的教训：镜像里没有 fuser，按端口查也不可靠，
# 必须扫 /proc/*/cmdline 精确匹配，且排除调用方 shell 自身，
# 否则「旧进程占着端口、新进程绑定失败退出」会导致改动不生效。
set -u
ROOT=/root/autodl-tmp/education
# 仓库位置：优先 COSYVOICE_DIR，其次项目自带的 models_zoo/CosyVoice，最后才看老的 /root/CosyVoice。
# （老脚本写死 /root/CosyVoice，而本机仓库其实在 models_zoo 下，直接跑会「找不到仓库」。）
COSY="${COSYVOICE_DIR:-$ROOT/models_zoo/CosyVoice}"
[ -d "$COSY" ] || COSY=/root/autodl-tmp/CosyVoice
# 解释器：必须 conda cv310（torch 2.3.1，与官方 CosyVoice 匹配）；不可用才回退 .venvs/cosy。
TTS_PYTHON="${TTS_PYTHON:-/root/miniconda3/envs/cv310/bin/python}"
[ -x "$TTS_PYTHON" ] || TTS_PYTHON="$ROOT/.venvs/cosy/bin/python"
# 模型：环境变量 > .env 的 COSYVOICE_MODEL > 引擎按目录名自动挑（优先 CosyVoice3）。
# 与 start_all.sh 保持一致——否则手动跑本脚本会忽略 .env 里钉住的模型版本。
[ -n "${COSYVOICE_MODEL:-}" ] || COSYVOICE_MODEL="$(grep -E '^COSYVOICE_MODEL=' "$ROOT/.env" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '[:space:]' || true)"
LOG="$ROOT/deploy/cloud/logs/cosyvoice.log"
cd "$ROOT" || exit 1
mkdir -p "$(dirname "$LOG")"
PATTERN="tts_engine_cosyvoice.py"

find_pids() {
  local p cmd
  for p in /proc/[0-9]*; do
    p="${p#/proc/}"
    [ "$p" = "$$" ] && continue
    [ "$p" = "${PPID:-0}" ] && continue
    cmd="$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || true)"
    case "$cmd" in
      *"$PATTERN"*) echo "$p" ;;
    esac
  done
}

OLD="$(find_pids | sort -u | tr '\n' ' ')"
if [ -n "${OLD// /}" ]; then
  echo "[restart-tts] 停止旧引擎：$OLD"
  # shellcheck disable=SC2086
  kill -9 $OLD 2>/dev/null || true
fi
for _ in $(seq 1 15); do
  [ -z "$(find_pids)" ] && break
  sleep 1
done

# COSYVOICE_MODEL 留空 = 引擎自动挑 pretrained_models 里的版本（优先 CosyVoice3），
# 想钉死某个版本就在 .env 里写 COSYVOICE_MODEL=<绝对路径>。
setsid nohup env COSYVOICE_DIR="$COSY" \
  COSYVOICE_MODEL="${COSYVOICE_MODEL:-}" \
  ENGINE_PORT=8090 ENGINE_HOST=127.0.0.1 COSYVOICE_FP16="${COSYVOICE_FP16:-0}" \
  "$TTS_PYTHON" scripts/tts_engine_cosyvoice.py >"$LOG" 2>&1 </dev/null &
sleep 2
echo "[restart-tts] 新引擎：$(find_pids | sort -u | tr '\n' ' ')"

for _ in $(seq 1 40); do
  curl -sf -m 4 http://127.0.0.1:8090/health >/dev/null 2>&1 && break
  sleep 3
done
echo "[restart-tts] health: $(curl -s -m 6 http://127.0.0.1:8090/health)"
