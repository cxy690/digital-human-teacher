#!/usr/bin/env bash
# ============================================================
# 数字人模型权重下载脚本（在 AutoDL 服务器上跑；无卡模式即可，不需要 GPU）
# 把 MuseTalk / sd-vae / whisper 权重按项目目录结构下到 digital_human/models/
#
# 用法（在项目根或任意目录）：
#   bash digital_human/fetch_models.sh                  # 只下推理必需（unet/vae/whisper，约 3.8GB）
#   bash digital_human/fetch_models.sh --with-faceparse # 额外下 face-parse（仅用 genavatar 生成新素材时才需要）
#
# 说明：AutoDL 上国内访问 HuggingFace 慢/不通，脚本默认走 hf-mirror 镜像，
#       并尝试 source /etc/network_turbo（AutoDL 学术加速）。可用 HF_ENDPOINT 覆盖。
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 模型目录：默认取脚本所在目录下的 models/。
# 警告：若把脚本拷到别处（如 /tmp）运行，务必用 MODELS_DIR 显式指定目标，
# 否则会下到脚本所在目录（曾因此误下到 /tmp/models）！
MODELS_DIR="${MODELS_DIR:-$SCRIPT_DIR/models}"

# AutoDL 学术加速（若存在则启用）；默认走 hf-mirror 镜像
[ -f /etc/network_turbo ] && source /etc/network_turbo || true
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# hf-mirror 不支持 HF 新的 Xet 大文件存储后端（会报 cas-server.xethub 401），
# 禁用 xet 退回传统 HTTP 下载；如仍失败可改 HF_ENDPOINT=https://huggingface.co 走官方
export HF_HUB_DISABLE_XET=1

# huggingface CLI：新版叫 hf，旧版叫 huggingface-cli。
# 注意：新版 huggingface_hub 已移除 huggingface-cli 入口（只剩打印弃用警告的空壳），
# 所以“安装后必须重新探测”，且优先 hf，不能写死 huggingface-cli。
detect_hf() {
    if command -v hf >/dev/null 2>&1; then HF="hf download"
    elif command -v huggingface-cli >/dev/null 2>&1; then HF="huggingface-cli download"
    else return 1; fi
}
if ! detect_hf; then
    echo "==> 未找到 huggingface CLI，安装中..."
    pip install -U huggingface_hub >/dev/null
    detect_hf || { echo "安装后仍无 hf 命令，请手动: pip install -U huggingface_hub"; exit 1; }
fi

mkdir -p "$MODELS_DIR"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "==> [1/3] MuseTalk unet（TMElyralab/MuseTalk，约 3.3GB）..."
mkdir -p "$MODELS_DIR/musetalkV15"
$HF TMElyralab/MuseTalk musetalk/pytorch_model.bin musetalk/musetalk.json --local-dir "$TMP/musetalk"
cp -f "$TMP/musetalk/musetalk/pytorch_model.bin" "$MODELS_DIR/musetalkV15/unet.pth"
cp -f "$TMP/musetalk/musetalk/musetalk.json"     "$MODELS_DIR/musetalkV15/musetalk.json"

echo "==> [2/3] sd-vae（stabilityai/sd-vae-ft-mse，约 327MB）..."
mkdir -p "$MODELS_DIR/sd-vae"
$HF stabilityai/sd-vae-ft-mse diffusion_pytorch_model.bin config.json --local-dir "$TMP/sdvae"
cp -f "$TMP/sdvae/diffusion_pytorch_model.bin" "$MODELS_DIR/sd-vae/diffusion_pytorch_model.bin"
cp -f "$TMP/sdvae/config.json"                 "$MODELS_DIR/sd-vae/config.json"

echo "==> [3/3] whisper-tiny（openai/whisper-tiny，约 147MB）..."
mkdir -p "$MODELS_DIR/whisper"
$HF openai/whisper-tiny pytorch_model.bin config.json preprocessor_config.json --local-dir "$TMP/whisper"
cp -f "$TMP/whisper/pytorch_model.bin"        "$MODELS_DIR/whisper/pytorch_model.bin"
cp -f "$TMP/whisper/config.json"              "$MODELS_DIR/whisper/config.json"
cp -f "$TMP/whisper/preprocessor_config.json" "$MODELS_DIR/whisper/preprocessor_config.json"

if [ "${1:-}" = "--with-faceparse" ]; then
    echo "==> [可选] face-parse-bisent（仅用 genavatar 生成新素材时需要；跑推理不需要）..."
    mkdir -p "$MODELS_DIR/face-parse-bisent"
    # resnet18 backbone：PyTorch 官方直链，稳定可下
    wget -q --show-progress -O "$MODELS_DIR/face-parse-bisent/resnet18-5c106cde.pth" \
        https://download.pytorch.org/models/resnet18-5c106cde.pth
    # 79999_iter.pth：原仓库托管在 Google Drive，无稳定直链，需手动获取后放入：
    #   digital_human/models/face-parse-bisent/79999_iter.pth
    # 来源见 zllrunning/face-parsing.PyTorch 的 README（或你自己的备份）。
    echo "  [提示] 79999_iter.pth 需手动下载放入 $MODELS_DIR/face-parse-bisent/（Google Drive，无稳定直链）"
fi

echo ""
echo "完成。已下载的模型文件："
if find "$MODELS_DIR" -maxdepth 2 -type f -printf '.' >/dev/null 2>&1; then
    find "$MODELS_DIR" -maxdepth 2 -type f -printf '  %10s  %p\n' | sort -k2
else
    ls -lR "$MODELS_DIR"
fi
