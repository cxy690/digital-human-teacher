#!/usr/bin/env bash
# ============================================================================
# 算力云部署 · 第二步：克隆模型仓库 + 下载权重
# 用法：按需开关（显存/磁盘不够就关掉用不到的）
#   ENABLE_QWEN3=1 ENABLE_AWQ=0 ENABLE_TTS=1 ENABLE_MUSETALK=1 ENABLE_WAV2LIP=0 \
#   bash deploy/cloud/download_models.sh
#   * ENABLE_AWQ=1 额外下载本地「离线答疑大脑」Qwen2.5-7B-AWQ（switch_mode.sh offline 依赖）
#
# 下载源：国内默认走 HF 镜像 hf-mirror.com + ModelScope（可用 USE_HF=1 切官方）
# 详细清单（大小/显存/来源）见 算力云部署.md
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
MODEL_ROOT="${MODEL_ROOT:-$ROOT/models_zoo}"     # 仓库与权重统一放这里
mkdir -p "$MODEL_ROOT"

ENABLE_QWEN3="${ENABLE_QWEN3:-1}"
ENABLE_AWQ="${ENABLE_AWQ:-0}"          # 本地「离线答疑大脑」AWQ 权重（switch_mode.sh offline 依赖）
ENABLE_TTS="${ENABLE_TTS:-1}"
ENABLE_MUSETALK="${ENABLE_MUSETALK:-1}"
ENABLE_WAV2LIP="${ENABLE_WAV2LIP:-0}"

USE_HF="${USE_HF:-0}"
if [ "$USE_HF" != "1" ]; then
  export HF_ENDPOINT="https://hf-mirror.com"       # 国内镜像
  echo "[download] 使用 HF 镜像: $HF_ENDPOINT"
fi

# 优先用 huggingface-cli；没有则尝试 modelscope
hf_download() {  # $1=repo_id  $2=local_dir
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$1" --local-dir "$2" --local-dir-use-symlinks False
  else
    python3 -m huggingface_hub.commands.huggingface_cli download "$1" --local-dir "$2" || \
      echo "[download] huggingface-cli 不可用，请 pip install -U huggingface_hub 或改用 modelscope"
  fi
}
ms_download() {  # $1=modelscope_id  $2=local_dir
  python3 -c "from modelscope import snapshot_download; snapshot_download('$1', local_dir='$2')"
}

# ---------------------------------------------------------------- Qwen3 (LLM)
if [ "$ENABLE_QWEN3" = "1" ]; then
  echo "[download] === Qwen3 LLM 权重 ==="
  QWEN_ID="${QWEN_ID:-Qwen/Qwen3-8B}"              # 显存小可改 Qwen/Qwen3-4B；INT4 用 Qwen3-8B-AWQ
  QWEN_DIR="$MODEL_ROOT/Qwen3"
  ms_download "$QWEN_ID" "$QWEN_DIR" || hf_download "$QWEN_ID" "$QWEN_DIR"
  echo "[download] Qwen3 -> $QWEN_DIR （vLLM 服务见 start_all.sh）"
fi

# ------------------------------------------------- 本地 AWQ 答疑大脑（可选，离线模式用）
# start_all.sh / switch_mode.sh offline 会从 models_zoo 自动查找 *Qwen2.5-7B-AWQ* 目录。
if [ "$ENABLE_AWQ" = "1" ]; then
  echo "[download] === 本地 AWQ 答疑大脑权重（离线模式）==="
  AWQ_ID="${AWQ_ID:-Qwen/Qwen2.5-7B-Instruct-AWQ}"   # 亦可 Qwen/Qwen3-8B-AWQ
  AWQ_DIR="$MODEL_ROOT/Qwen2.5-7B-AWQ"
  ms_download "$AWQ_ID" "$AWQ_DIR" || hf_download "$AWQ_ID" "$AWQ_DIR"
  echo "[download] AWQ -> $AWQ_DIR （离线答疑服务见 scripts/llm_engine_local.py）"
fi

# ---------------------------------------------------------------- CosyVoice (TTS，优先 3)
if [ "$ENABLE_TTS" = "1" ]; then
  echo "[download] === CosyVoice 仓库 + 权重（优先 CosyVoice3，失败回落 CosyVoice2）==="
  CV_DIR="$MODEL_ROOT/CosyVoice"
  [ -d "$CV_DIR" ] || git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git "$CV_DIR"
  # CosyVoice3：官方权重 FunAudioLLM/Fun-CosyVoice3-0.5B-2512（约 4-5GB，ModelScope 国内直连很快）。
  # 国内镜像优先，不需要代理/加速器；引擎按 pretrained_models 下的目录名自动挑版本。
  CV3_ID="${CV3_ID:-FunAudioLLM/Fun-CosyVoice3-0.5B-2512}"
  CV3_DIR="$CV_DIR/pretrained_models/Fun-CosyVoice3-0.5B"
  if [ ! -d "$CV3_DIR" ]; then
    ms_download "$CV3_ID" "$CV3_DIR" || \
      echo "[download] CosyVoice3 权重下载失败，请手动执行：modelscope download --model $CV3_ID --local_dir $CV3_DIR"
  fi
  if [ -d "$CV3_DIR" ]; then
    echo "[download] CosyVoice3 -> $CV3_DIR"
  else
    ms_download "iic/CosyVoice2-0.5B" "$CV_DIR/pretrained_models/CosyVoice2-0.5B" || \
      echo "[download] CosyVoice2 权重下载失败，请手动从 ModelScope iic/CosyVoice2-0.5B 获取"
    echo "[download] CosyVoice2 -> $CV_DIR"
  fi
fi

# ---------------------------------------------------------------- MuseTalk (L3 实时数字人)
if [ "$ENABLE_MUSETALK" = "1" ]; then
  echo "[download] === MuseTalk 仓库 + 权重 ==="
  MT_DIR="$MODEL_ROOT/MuseTalk"
  [ -d "$MT_DIR" ] || git clone https://github.com/TMElyralab/MuseTalk.git "$MT_DIR"
  # MuseTalk 权重（musetalk/pytorch_model.bin、sd-vae-ft-mse、whisper、dwpose、face-parse-bisnet 等）
  hf_download "TMElyralab/MuseTalk" "$MT_DIR/models/musetalk" || \
    echo "[download] MuseTalk 权重请按仓库 README 手动放置到 $MT_DIR/models"
  echo "[download] MuseTalk -> $MT_DIR"
fi

# ---------------------------------------------------------------- Wav2Lip (兼容兜底, 可选)
if [ "$ENABLE_WAV2LIP" = "1" ]; then
  echo "[download] === Wav2Lip 仓库 + 权重（兜底，可选）==="
  W2L_DIR="$MODEL_ROOT/Wav2Lip"
  [ -d "$W2L_DIR" ] || git clone https://github.com/Rudrabha/Wav2Lip.git "$W2L_DIR"
  hf_download "camenduru/Wav2Lip" "$W2L_DIR/checkpoints" || \
    echo "[download] Wav2Lip 权重(wav2lip_gan.pth / s3fd.pth)请手动放置到 $W2L_DIR/checkpoints"
  echo "[download] Wav2Lip -> $W2L_DIR"
fi

echo ""
echo "[download] 完成。模型仓库根目录：$MODEL_ROOT"
echo "[download] 请把以下路径写进启动环境（start_all.sh 已用同名变量，可在此覆盖）："
echo "  QWEN_DIR=$MODEL_ROOT/Qwen3"
echo "  AWQ_DIR=$MODEL_ROOT/Qwen2.5-7B-AWQ   （ENABLE_AWQ=1 时下载，离线答疑用）"
echo "  COSYVOICE_DIR=$MODEL_ROOT/CosyVoice"
echo "  MUSETALK_DIR=$MODEL_ROOT/MuseTalk"
echo "  磁盘占用参考见 算力云部署.md 下载清单。"
