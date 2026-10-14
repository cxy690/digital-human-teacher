#!/usr/bin/env bash
# ============================================================================
# 算力云部署 · 第三步：一键启动全部服务
# 用法：
#   cp deploy/cloud/env.cloud.example .env          # 先准备配置
#   bash deploy/cloud/start_all.sh                  # 全开
#
# 端口约定（与 env.cloud.example 一致）：
#   Qwen3(vLLM) :8100 | FunASR :10095 | CosyVoice 3 :8090
#   MuseTalk :8012 | 主应用 :8001
# 日志：deploy/cloud/logs/*.log ；停止：bash deploy/cloud/stop_all.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
VENV_DIR="$ROOT/.venvs"
LOG_DIR="$SCRIPT_DIR/logs"
MODEL_ROOT="${MODEL_ROOT:-$ROOT/models_zoo}"
mkdir -p "$LOG_DIR" "$VENV_DIR"

# ---------------------------------------------------------------- 解释器兜底
# AutoDL 等镜像把 python3 装在 /root/miniconda3/bin 却**不在默认 PATH 里**，
# 直接用 `python3 -m venv` 建环境会报 "python3: command not found"，
# 于是整个脚本在第一步就中断 —— 现象是「只起来主应用，引擎全没起」。
# 这里自动把常见位置补进 PATH。
if ! command -v python3 >/dev/null 2>&1; then
  for _d in /root/miniconda3/bin /usr/local/miniconda3/bin /usr/local/bin /usr/bin; do
    if [ -x "$_d/python3" ]; then
      export PATH="$_d:$PATH"
      echo "[start] 补充 PATH：python3 -> $_d/python3"
      break
    fi
  done
fi

# TTS / MuseTalk 优先用 conda 环境（cv310 / mt310，已验证可跑），避免 .venvs/tts|/musetalk 在 py3.12 上重装失败。
# 如果 conda 环境不在（例如换镜像了），会自动回退到 .venvs/<name> 并跑 ensure_venv 重建。
TTS_PYTHON="${TTS_PYTHON:-/root/miniconda3/envs/cv310/bin/python}"
MUSETALK_PYTHON="${MUSETALK_PYTHON:-/usr/local/miniconda3/envs/mt310/bin/python}"
[ -x "$TTS_PYTHON" ]      || TTS_PYTHON="$VENV_DIR/cosy/bin/python"
[ -x "$MUSETALK_PYTHON" ] || MUSETALK_PYTHON="$VENV_DIR/musetalk/bin/python"
# ⚠ CosyVoice 必须跑在 conda cv310（torch 2.3.1，与官方匹配）。
# 不要改成继承系统 torch 2.8 的 .venvs/*：实测会出现 dtype / conv 不兼容
# （"mat1 and mat2 must have the same dtype" 或 "Kernel size can't be greater than
# actual input size"），且请求永久挂起 —— 表现就是「老师不出声、白板先行后无语音」。

QWEN_DIR="${QWEN_DIR:-$MODEL_ROOT/Qwen3}"
COSYVOICE_DIR="${COSYVOICE_DIR:-$MODEL_ROOT/CosyVoice}"
MUSETALK_DIR="${MUSETALK_DIR:-$MODEL_ROOT/MuseTalk}"
# CosyVoice 仓库/权重装在数据盘 /root/autodl-tmp/CosyVoice 时自动识别（盘大、不占系统盘）
[ -d "$COSYVOICE_DIR" ] || { [ -d /root/autodl-tmp/CosyVoice ] && COSYVOICE_DIR=/root/autodl-tmp/CosyVoice; }

# 本地答疑模型已改为 transformers+AWQ 轻量方案（见 scripts/llm_engine_local.py），
# 不再需要 vLLM（本机 pip 装 vLLM 会长时间卡在依赖下载）。默认关闭。
ENABLE_QWEN3="${ENABLE_QWEN3:-0}"
ENABLE_ASR="${ENABLE_ASR:-1}"
ENABLE_TTS="${ENABLE_TTS:-1}"
ENABLE_MUSETALK="${ENABLE_MUSETALK:-1}"
PY="${PYTHON:-python3}"
# 平台镜像已预装 torch+CUDA：venv 继承系统 torch，避免重复下载/版本不匹配
INHERIT_SYSTEM="${INHERIT_SYSTEM:-1}"
VENV_FLAGS=""
[ "$INHERIT_SYSTEM" = "1" ] && VENV_FLAGS="--system-site-packages"

# GPU 主版本号（V100=7，A100=8，4090=8，H20=9）；取不到默认 8
gpu_major() { python3 -c "import torch;print(torch.cuda.get_device_capability()[0])" 2>/dev/null || echo 8; }

# 老包（grpcio 等）源码构建时，新版 setuptools 已移除 pkg_resources -> 约束构建期 setuptools<81
CONSTRAINTS="$SCRIPT_DIR/constraints.txt"
[ -f "$CONSTRAINTS" ] || echo 'setuptools<81' > "$CONSTRAINTS"
export PIP_CONSTRAINT="$CONSTRAINTS"

# 确保某个 venv 存在；$2 给出 requirements 文件则安装（过滤 torch 行，不覆盖平台 torch）
ensure_venv() {  # $1=name  $2=requirements(optional)
  local name="$1" req="${2:-}"
  if [ ! -x "$VENV_DIR/$name/bin/python" ]; then
    echo "[start] 创建环境 .venvs/$name ..."
    "$PY" -m venv $VENV_FLAGS "$VENV_DIR/$name"
    "$VENV_DIR/$name/bin/pip" install -U pip >/dev/null 2>&1 || true
    # 构建期需要 pkg_resources：先装 setuptools<81 + wheel
    "$VENV_DIR/$name/bin/pip" install -U "setuptools<81" wheel >/dev/null 2>&1 || true
    if [ -n "$req" ] && [ -f "$req" ]; then
      # 过滤平台已带的 torch 系列，以及需按 py3.12 适配而会编译失败的 numpy/grpcio 固定版本
      grep -viE '^[[:space:]]*(torch|torchaudio|torchvision|cuda|nvidia|numpy|grpcio)' "$req" \
        | "$VENV_DIR/$name/bin/pip" install -r /dev/stdin || true
      # 用与 Python 3.12 兼容的版本补齐
      "$VENV_DIR/$name/bin/pip" install "numpy>=1.26,<2" "grpcio>=1.60" >/dev/null 2>&1 || true
    fi
  fi
}

# 修正 torchaudio 与平台 torch 的 CUDA 不匹配（如 vllm 拉入 cu130 而平台 torch 是 cu132）
fix_torchaudio() {  # $1=venv name
  local name="$1" ver cuda
  ver="$(python3 -c 'import torch;print(torch.__version__.split("+")[0])' 2>/dev/null || true)"
  cuda="$(python3 -c 'import torch;print("cu"+torch.version.cuda.replace(".",""))' 2>/dev/null || true)"
  if [ -n "$ver" ] && [ -n "$cuda" ]; then
    "$VENV_DIR/$name/bin/pip" install --force-reinstall --no-deps "torchaudio==$ver" \
      --index-url "https://download.pytorch.org/whl/$cuda" >/dev/null 2>&1 || true
  fi
}

launch() {  # $1=logfile  其余=命令
  local logf="$1"; shift
  echo "[start] $* -> 日志 $logf"
  nohup "$@" >"$logf" 2>&1 &
  echo $! >> "$LOG_DIR/pids.txt"
}

: > "$LOG_DIR/pids.txt"

# ---------------------------------------------------------------- 僵尸进程清理
# vLLM EngineCore 在异常退出时可能留下子进程占着 GPU 显存不放（本次部署实际碰到：30GB
# 显存被 PID 2864 VLLM::EngineCore 占住，导致新 vLLM 启动时报 free memory 不够）。
# 启动前先清一道，只杀 vllm/EngineCore，不动其他服务。
if command -v nvidia-smi >/dev/null 2>&1; then
  pkill -9 -f 'VLLM::EngineCore' 2>/dev/null && echo "[start] 清理僵尸 VLLM::EngineCore" || true
  pkill -9 -f 'vllm.entrypoints'  2>/dev/null && echo "[start] 清理僵尸 vllm.entrypoints"  || true
  sleep 2
fi

# ---------------------------------------------------------------- Qwen3 (vLLM)
if [ "$ENABLE_QWEN3" = "1" ]; then
  ensure_venv llm
  [ -x "$VENV_DIR/llm/bin/vllm" ] || "$VENV_DIR/llm/bin/pip" install vllm
  fix_torchaudio llm
  # flashinfer 采样器需 ninja 现场编译；装上 ninja、把 venv/bin 入 PATH，并优先用原生采样器
  "$VENV_DIR/llm/bin/pip" install ninja >/dev/null 2>&1 || true
  export VLLM_USE_FLASHINFER_SAMPLER=0
  export PATH="$VENV_DIR/llm/bin:$PATH"
  # V100(sm_70) 等老架构 FlashAttention 不可用 -> 回退 XFORMERS
  GM="$(gpu_major)"
  if [ "$GM" -lt 8 ]; then
    "$VENV_DIR/llm/bin/pip" install xformers >/dev/null 2>&1 || true
    export VLLM_ATTENTION_BACKEND=XFORMERS
    echo "[start] 检测到 sm_${GM}0 老架构（如 V100），vLLM 使用 XFORMERS attention 后端"
  fi
  # 显存与其余服务（FunASR/CosyVoice/MuseTalk 约 13G）共存：默认 0.6，避免 vLLM 抢 92% 导致 "free memory < gpu memory utilization" 启动失败
  launch "$LOG_DIR/qwen3.log" "$VENV_DIR/llm/bin/vllm" serve "$QWEN_DIR" \
      --served-model-name Qwen3-8B --host 127.0.0.1 --port 8100 \
      --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL:-0.6}"
fi

# ---------------------------------------------------------------- FunASR (ASR)
# 永久本地"耳朵"：SenseVoiceSmall（离线可用、约 0.3s/句）
if [ "$ENABLE_ASR" = "1" ]; then
  if [ ! -x "$VENV_DIR/asr/bin/python" ]; then
    ensure_venv asr
    "$VENV_DIR/asr/bin/pip" install -q funasr modelscope soundfile \
      fastapi uvicorn python-multipart kaldi-native-fbank >/dev/null 2>&1 || true
  fi
  launch "$LOG_DIR/funasr.log" "$VENV_DIR/asr/bin/python" scripts/asr_engine_funasr.py
fi

# ---------------------------------------------------------------- 本地 LLM（离线"大脑"）
# 仅当 .env 的 LLM_API_BASE 指向本机 8100（即离线模式）时才启动，在线模式不占显存。
if [ -x "$VENV_DIR/llm/bin/python" ] \
   && grep -q '^LLM_API_BASE=http://127.0.0.1:8100' "$ROOT/.env" 2>/dev/null; then
  LOCAL_LLM_DIR="${LOCAL_LLM_DIR:-$(find "$ROOT/models_zoo" -maxdepth 4 -type d \
    -name '*Qwen2.5-7B-AWQ*' 2>/dev/null | head -1)}"
  if [ -n "$LOCAL_LLM_DIR" ]; then
    echo "[start] 本地 LLM（离线答疑）：$LOCAL_LLM_DIR"
    LOCAL_LLM_DIR="$LOCAL_LLM_DIR" LOCAL_LLM_NAME=Qwen2.5-7B-AWQ ENGINE_PORT=8100 \
      launch "$LOG_DIR/local_llm.log" "$VENV_DIR/llm/bin/python" scripts/llm_engine_local.py
  else
    echo "[start] 未找到本地 LLM 模型目录，离线答疑将不可用"
  fi
fi

# ---------------------------------------------------------------- CosyVoice (TTS，2/3 通用)
if [ "$ENABLE_TTS" = "1" ]; then
  # 优先用 conda cv310（已含 hyperpyyaml/torch/torchaudio）；否则回退 .venvs/tts 并现场装依赖
  if [ "$TTS_PYTHON" = "$VENV_DIR/tts/bin/python" ]; then
    ensure_venv tts "$COSYVOICE_DIR/requirements.txt"
    fix_torchaudio tts
    "$VENV_DIR/tts/bin/pip" install fastapi uvicorn soundfile >/dev/null 2>&1 || true
  fi
  # 模型目录：优先 .env 的 COSYVOICE_MODEL；留空则**引擎按目录名自动挑**
  # （优先 Fun-CosyVoice3-0.5B，没有就回落 CosyVoice2-0.5B）——换模型不用改脚本。
  if [ -z "${COSYVOICE_MODEL:-}" ] && [ -f "$ROOT/.env" ]; then
    _cv_model_from_env="$(grep -E '^COSYVOICE_MODEL=' "$ROOT/.env" | tail -1 | cut -d= -f2- | tr -d '[:space:]' || true)"
    [ -n "$_cv_model_from_env" ] && COSYVOICE_MODEL="$_cv_model_from_env"
  fi
  echo "[start] CosyVoice 使用解释器：$TTS_PYTHON  模型：${COSYVOICE_MODEL:-（自动挑 pretrained_models 里的版本）}"
  # fp16 关闭：本机实测半精度会让输出退化成「定长噪声」
  COSYVOICE_DIR="$COSYVOICE_DIR" COSYVOICE_MODEL="${COSYVOICE_MODEL:-}" \
    ENGINE_PORT=8090 COSYVOICE_FP16="${COSYVOICE_FP16:-0}" \
    launch "$LOG_DIR/cosyvoice.log" "$TTS_PYTHON" scripts/tts_engine_cosyvoice.py
fi

# ---------------------------------------------------------------- MuseTalk (L3)
if [ "$ENABLE_MUSETALK" = "1" ]; then
  # 优先用 conda mt310（已含 cv2/torch/mmcv/musetalk 依赖）；否则回退 .venvs/musetalk
  if [ "$MUSETALK_PYTHON" = "$VENV_DIR/musetalk/bin/python" ]; then
    ensure_venv musetalk "$MUSETALK_DIR/requirements.txt"
    fix_torchaudio musetalk
  fi
  # 引擎侧读 MUSETALK_BBOX_SHIFT（人脸框上下平移）；若未显式指定，从 .env 的 AVATAR_BBOX_SHIFT 拉取。
  # 两侧保持一致，避免“主应用下发 bbox_shift=X、引擎默认 Y”的歧义。
  if [ -z "${MUSETALK_BBOX_SHIFT:-}" ] && [ -f "$ROOT/.env" ]; then
    _shift_from_env="$(grep -E '^AVATAR_BBOX_SHIFT=' "$ROOT/.env" | tail -1 | cut -d= -f2 | tr -d '[:space:]' || true)"
    [ -n "$_shift_from_env" ] && MUSETALK_BBOX_SHIFT="$_shift_from_env"
  fi
  MUSETALK_BBOX_SHIFT="${MUSETALK_BBOX_SHIFT:-0}"
  # 动作基元库：主底片 talk.mp4 + 手势 nod/shake/shrug/point.mp4（可选 idle.mp4）。
  # 优先读 .env 的 AI_TEACHER_CLIP_DIR（相对项目根），否则默认项目内 data/ai_teacher。
  if [ -z "${AVATAR_CLIP_DIR:-}" ] && [ -f "$ROOT/.env" ]; then
    _clip_from_env="$(grep -E '^AI_TEACHER_CLIP_DIR=' "$ROOT/.env" | tail -1 | cut -d= -f2 | tr -d '[:space:]' || true)"
    [ -n "$_clip_from_env" ] && AVATAR_CLIP_DIR="$_clip_from_env"
  fi
  AVATAR_CLIP_DIR="${AVATAR_CLIP_DIR:-$ROOT/data/teachers}"
  # 引擎启动后会 os.chdir(MUSETALK_DIR)：相对路径会失效（读不到底片 -> “底片无有效帧”）。
  # 这里统一按项目根解析成绝对路径，保留 .env 里写相对路径的习惯。
  case "$AVATAR_CLIP_DIR" in
    /*) ;;
    *) AVATAR_CLIP_DIR="$ROOT/$AVATAR_CLIP_DIR" ;;
  esac
  # 主底片路径：只认 .env 的 AVATAR_FACE_VIDEO；**不再回退到任何 底片.mp4**
  # （多老师模式下底片由 teacher_id 决定：$AVATAR_TEACHERS_DIR/<key>/talk.mp4）
  if [ -z "${AVATAR_FACE_VIDEO:-}" ] && [ -f "$ROOT/.env" ]; then
    _face_from_env="$(grep -E '^AVATAR_FACE_VIDEO=' "$ROOT/.env" | tail -1 | cut -d= -f2- | tr -d '[:space:]' || true)"
    [ -n "$_face_from_env" ] && AVATAR_FACE_VIDEO="$_face_from_env"
  fi
  case "${AVATAR_FACE_VIDEO:-}" in
    /*) ;;
    '') ;;                                    # 留空：引擎跳过旧单底片，完全按 teacher_id 加载
    *) AVATAR_FACE_VIDEO="$ROOT/$AVATAR_FACE_VIDEO" ;;
  esac
  # 人脸预处理缓存目录（按 video_hash 分子目录）；相对路径由引擎按项目根自解析。
  AVATAR_CACHE_DIR="${AVATAR_CACHE_DIR:-$ROOT/data/cache}"
  MUSETALK_CROSSFADE="${MUSETALK_CROSSFADE:-6}"
  echo "[start] MuseTalk 使用解释器：$MUSETALK_PYTHON  bbox_shift=$MUSETALK_BBOX_SHIFT（正下负上，官方推荐 [-10,10]）"
  echo "[start] MuseTalk 动作基元目录：$AVATAR_CLIP_DIR  crossfade=$MUSETALK_CROSSFADE（手势切换淡化帧数）"
  echo "[start] MuseTalk 人脸缓存目录：$AVATAR_CACHE_DIR"
  # 新版 API 引擎 v15（mmcv-free；适配 vae/unet/pe + AudioProcessor/WhisperModel）；内部会 os.chdir(MUSETALK_DIR)
  # 主底片 talk.mp4 + 问候底片 idle.mp4（gesture=greet 时用 idle.mp4 打招呼并对口型）
  # 多老师：底片按学科放在 $AVATAR_TEACHERS_DIR/<key>/talk.mp4，由 /render 的 teacher_id 按需加载。
  # 关掉「旧单底片预加载」可省显存与启动时间（AVATAR_LEGACY_PRELOAD=0）。
  AVATAR_TEACHERS_DIR="${AVATAR_TEACHERS_DIR:-$ROOT/data/teachers}"
  case "$AVATAR_TEACHERS_DIR" in
    /*) ;;
    *) AVATAR_TEACHERS_DIR="$ROOT/$AVATAR_TEACHERS_DIR" ;;
  esac
  AVATAR_LEGACY_PRELOAD="${AVATAR_LEGACY_PRELOAD:-0}"
  echo "[start] MuseTalk 教师底片目录：$AVATAR_TEACHERS_DIR（legacy_preload=$AVATAR_LEGACY_PRELOAD）"
  MUSETALK_DIR="$MUSETALK_DIR" ENGINE_PORT=8012 \
    AVATAR_CLIP_DIR="$AVATAR_CLIP_DIR" \
    AVATAR_TEACHERS_DIR="$AVATAR_TEACHERS_DIR" \
    AVATAR_LEGACY_PRELOAD="$AVATAR_LEGACY_PRELOAD" \
    AVATAR_FACE_VIDEO="${AVATAR_FACE_VIDEO:-}" \
    AVATAR_GREET_CLIP="${AVATAR_GREET_CLIP:-}" \
    MUSETALK_FP16="${MUSETALK_FP16:-1}" \
    MUSETALK_VIDEO_QUALITY="${MUSETALK_VIDEO_QUALITY:-4}" \
    launch "$LOG_DIR/musetalk.log" "$MUSETALK_PYTHON" scripts/avatar_engine_musetalk_v15.py
fi

# ---------------------------------------------------------------- 主应用
echo "[start] 等待各引擎初始化（首次加载模型较慢）..."
sleep "${WAIT_BEFORE_APP:-20}"
ensure_venv app
launch "$LOG_DIR/app.log" "$VENV_DIR/app/bin/python" -m server.app

echo ""
echo "[start] 全部服务已在后台启动。健康检查："
sleep 5
for url in \
  "http://127.0.0.1:8100/v1/models|Qwen3(vLLM)" \
  "http://127.0.0.1:10095/health|FunASR" \
  "http://127.0.0.1:8090/health|CosyVoice 3" \
  "http://127.0.0.1:8012/health|MuseTalk" \
  "http://127.0.0.1:8001/health|主应用" ; do
  u="${url%%|*}"; name="${url##*|}"
  if curl -sf "$u" >/dev/null 2>&1; then echo "  [OK]   $name  $u"; else echo "  [WAIT] $name  $u（仍在加载或未启用，看 logs/）"; fi
done

# ------------------------------------------------- 公网隧道（cpolar，国内直连）
# 把本机 8001 暴露成 HTTPS 公网地址，供本机以外的人访问（HTTPS 才能让浏览器放开麦克风）。
# 前提：已 `cpolar authtoken <token>` 写入配置；否则跳过，不影响本地服务。
#
# 注意两点（实测踩坑）：
# 1) 配置文件路径随安装方式不同：官方脚本装在 /usr/local/etc/cpolar/cpolar.yml，
#    手动解压常见 /root/.cpolar/cpolar.yml —— 两个都探一下，别写死。
# 2) 用 `cpolar http 8001`（临时隧道）而不是 `start-all`：后者只启动**后台已创建**的
#    命名隧道，全新账号的配置里没有任何隧道，start-all 会静默什么也不做。
if command -v cpolar >/dev/null 2>&1; then
  CPOLAR_CFG=""
  for c in "${CPOLAR_CFG:-}" /usr/local/etc/cpolar/cpolar.yml /root/.cpolar/cpolar.yml; do
    [ -n "$c" ] && [ -f "$c" ] && CPOLAR_CFG="$c" && break
  done
  if [ -z "$CPOLAR_CFG" ]; then
    echo "  [SKIP] 公网隧道(cpolar)  未配置 authtoken（先执行 cpolar authtoken <token>）"
  elif pgrep -f 'cpolar http' >/dev/null 2>&1; then
    echo "  [OK]   公网隧道(cpolar)  已在运行"
  else
    echo "[start] 启动 cpolar 公网隧道（-> 127.0.0.1:8001）..."
    nohup cpolar http -log=stdout "${APP_PORT:-8001}" > "$LOG_DIR/cpolar.log" 2>&1 &
    sleep 8
  fi
  # 免费版域名随机，重启后会变 —— 一律以日志里的新地址为准
  _pub=$(grep -hoE 'https://[A-Za-z0-9.-]+\.cpolar\.[a-z]+' "$LOG_DIR/cpolar.log" 2>/dev/null | tail -1)
  if [ -n "$_pub" ]; then
    echo "  [OK]   公网隧道(cpolar)  $_pub/app/subject_select/"
  elif [ -n "$CPOLAR_CFG" ]; then
    echo "  [WAIT] 公网隧道(cpolar)  仍在建立，见 logs/cpolar.log"
  fi
fi

echo ""
echo "[start] 学生端(本机): http://127.0.0.1:8001/app/subject_select/"
echo "[start] 公网地址: 见上方 cpolar 那一行（免费版域名每次重启会变）"
echo "[start] 停止: bash deploy/cloud/stop_all.sh"
