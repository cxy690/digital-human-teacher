#!/usr/bin/env bash
# ============================================================================
# 算力云部署 · 开机 GPU 自检（优云智算 / 任意 GPU 云通用）
# 用法：bash deploy/cloud/check_gpu.sh
#
# 验证三件事：
#   1) 平台镜像的 torch 能否识别到 GPU（nvidia-smi + torch.cuda）
#   2) GPU 计算能力（compute capability）——V100=sm_70，决定 vLLM/FlashAttention 能否用
#   3) 一次真实 CUDA 矩阵乘，确认 kernel 可用（老架构 torch 可能不含对应 kernel）
# ============================================================================
set -uo pipefail

echo "===== 1) nvidia-smi ====="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
else
  echo "[WARN] 未找到 nvidia-smi：可能无 GPU 或驱动未装"
fi

echo ""
echo "===== 2) torch 识别 GPU + compute capability + 矩阵乘 ====="
python3 - <<'PY'
import sys
try:
    import torch
except Exception as e:  # noqa: BLE001
    print("[FAIL] 无法 import torch:", e); sys.exit(1)
print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("[FAIL] torch 看不到 GPU：检查驱动/镜像是否带 CUDA；V100 需 torch 含 sm_70 kernel")
    sys.exit(1)
cap = torch.cuda.get_device_capability()
name = torch.cuda.get_device_name()
print(f"GPU: {name} | compute capability: sm_{cap[0]}{cap[1]}")
try:
    a = torch.rand(64, 64, device="cuda")
    b = torch.rand(64, 64, device="cuda")
    _ = (a @ b).sum().item()
    print("[OK] CUDA 矩阵乘正常")
except Exception as e:  # noqa: BLE001
    print("[FAIL] CUDA 运算报错（镜像 torch 可能不含该架构 kernel，换镜像）:", e); sys.exit(1)

if cap[0] < 8:
    print(f"[NOTE] sm_{cap[0]}{cap[1]} 为老架构（V100=sm_70）：vLLM 的 FlashAttention 不可用，")
    print("       start_all.sh 会自动改用 XFORMERS attention 后端；MuseTalk/CosyVoice/Hallo2 仍可运行。")
else:
    print("[NOTE] sm_80+：可用 FlashAttention，性能最佳。")
PY
rc=$?
echo ""
if [ $rc -eq 0 ]; then echo "[check_gpu] 通过，可继续 setup/download/start。"; else echo "[check_gpu] 未通过：先换带对应架构 kernel 的镜像再继续。"; fi
exit $rc
