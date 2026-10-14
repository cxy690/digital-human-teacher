#!/usr/bin/env bash
# ============================================================================
# 算力云部署 · 第一步：基础环境安装
# 适用：Linux GPU 云主机（AutoDL / 阿里云 / 腾讯云等，Ubuntu 20.04+，已带 NVIDIA 驱动）
# 用法：bash deploy/cloud/setup.sh
#
# 做的事：
#   1) 系统依赖（ffmpeg / git / libgl 等）
#   2) 主应用 venv（.venvs/app）+ 本项目依赖
#   3) FunASR venv（.venvs/asr）—— pip 可装、权重运行时自动下载
#   其余仓库型引擎（Qwen3 / CosyVoice / MuseTalk / Hallo2）的环境在
#   download_models.sh 克隆仓库后，用各自 requirements 安装。
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
VENV_DIR="$ROOT/.venvs"
PY="${PYTHON:-python3}"
# 平台镜像（优云智算/AutoDL 等）通常已预装 torch+CUDA：venv 用 --system-site-packages
# 继承系统 torch，避免重复下载与 CUDA 版本不匹配。设 INHERIT_SYSTEM=0 可改为独立安装。
INHERIT_SYSTEM="${INHERIT_SYSTEM:-1}"
VENV_FLAGS=""
[ "$INHERIT_SYSTEM" = "1" ] && VENV_FLAGS="--system-site-packages"

echo "[setup] 项目根目录: $ROOT"
mkdir -p "$VENV_DIR"

# ---- 0) GPU 自检（确认镜像 torch 认得这张卡 / 架构 kernel 可用）----
bash "$SCRIPT_DIR/check_gpu.sh" || \
  { echo "[setup] GPU 自检未通过，请先换镜像或确认驱动后重试"; exit 1; }

# ---- 1) 系统依赖 ----
if command -v apt-get >/dev/null 2>&1; then
  echo "[setup] 安装系统依赖 (ffmpeg/git/libgl)..."
  (apt-get update -y && apt-get install -y --no-install-recommends \
      ffmpeg git git-lfs libgl1 libglib2.0-0 wget curl) || \
    echo "[setup] 无 root 权限或已安装，跳过 apt（如缺 ffmpeg 请让云商镜像预装）"
  git lfs install || true
else
  echo "[setup] 非 apt 系统，请自行确保已安装 ffmpeg / git / git-lfs"
fi

# ---- 2) 主应用 venv ----
echo "[setup] 创建主应用环境 .venvs/app ..."
"$PY" -m venv $VENV_FLAGS "$VENV_DIR/app"
# shellcheck disable=SC1091
source "$VENV_DIR/app/bin/activate"
pip install --upgrade pip
# 主应用 + 编排 + 分层管线（不含 torch/GPU 依赖，纯 CPU 即可跑）
pip install -e ".[dev]"
# 常用可选：langgraph 编排 / 视素口型
pip install "langgraph>=0.2" "pypinyin>=0.5" || true
deactivate
echo "[setup] 主应用环境就绪"

# ---- 3) FunASR venv（ASR，pip 可装；torch 继承平台镜像，不重复下载）----
echo "[setup] 创建 FunASR 环境 .venvs/asr ..."
"$PY" -m venv $VENV_FLAGS "$VENV_DIR/asr"
# shellcheck disable=SC1091
source "$VENV_DIR/asr/bin/activate"
pip install --upgrade pip
# torch/torchaudio 由 --system-site-packages 继承平台镜像；这里只装 funasr 与服务依赖
pip install "funasr" "fastapi" "uvicorn" "python-multipart" "modelscope" || \
  echo "[setup] FunASR 依赖安装出现问题，请检查 torch 继承与 CUDA 版本匹配"
deactivate
echo "[setup] FunASR 环境就绪"

echo ""
echo "[setup] 完成。下一步："
echo "  1) bash deploy/cloud/download_models.sh   # 克隆仓库 + 下载权重（见 算力云部署.md 下载清单）"
echo "  2) cp deploy/cloud/env.cloud.example .env # 生成云端配置"
echo "  3) bash deploy/cloud/start_all.sh         # 一键启动全部服务"
echo "  提示：V100(sm_70) 等老架构，start_all.sh 会自动为 vLLM 切 XFORMERS attention 后端。"
