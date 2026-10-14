"""本地下载教学AI模型权重（Windows / Linux 通用，无需 bash）。

把文档要求的模型权重下载到本地 ``models_zoo/``，供本地调试/演示使用。
国内默认走 ModelScope + HF 镜像（hf-mirror.com），可用 --source hf 切官方。

注意：本地 RTX 5070 Laptop 仅 8GB 显存，**下载全部没问题（≈25-30GB）**，
但**同时全栈运行会 OOM**；本地推荐组合见 算力云部署.md /  README（LLM 走云 API +
TTS 走 edge-tts/CPU + 口型走本地 MuseTalk，或 Qwen3-4B-INT4 本地 + 照片木偶）。

用法：
    python -m scripts.download_models_local                 # 下载全部（qwen+cosyvoice+musetalk+funasr）
    python -m scripts.download_models_local --only musetalk # 只下某一个
    python -m scripts.download_models_local --qwen Qwen/Qwen3-4B   # 换更小的 Qwen
组件可选：qwen / cosyvoice / musetalk / funasr / all
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _default_dest() -> Path:
    """模型仓库根目录：优先环境变量 MODEL_ROOT；Windows 默认 D:/models_zoo（大件放 D 盘），否则项目内。"""
    env = os.environ.get("MODEL_ROOT")
    if env:
        return Path(env)
    if os.name == "nt" and Path("D:/").exists():
        return Path("D:/models_zoo")
    return ROOT / "models_zoo"


DEST = _default_dest()

# 国内镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

COMPONENTS = ["qwen", "cosyvoice", "musetalk", "funasr"]


def _pip_install(pkgs: list[str]) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)


def _ensure_libs() -> None:
    try:
        import modelscope  # noqa: F401
    except Exception:
        print("[dl] 安装 modelscope ...")
        _pip_install(["modelscope"])
    try:
        import huggingface_hub  # noqa: F401
    except Exception:
        print("[dl] 安装 huggingface_hub ...")
        _pip_install(["huggingface_hub"])


def _ms_download(model_id: str, local_dir: Path) -> None:
    from modelscope import snapshot_download

    print(f"[dl] ModelScope 下载 {model_id} -> {local_dir}")
    snapshot_download(model_id, local_dir=str(local_dir))


def _hf_download(repo_id: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download as hf_snap

    print(f"[dl] HuggingFace 下载 {repo_id} -> {local_dir}（镜像 {os.environ.get('HF_ENDPOINT')}）")
    hf_snap(repo_id, local_dir=str(local_dir))


def _git_clone(url: str, local_dir: Path) -> None:
    if local_dir.exists():
        print(f"[dl] 已存在，跳过克隆：{local_dir}")
        return
    print(f"[dl] git clone {url} -> {local_dir}")
    subprocess.run(["git", "clone", "--depth", "1", url, str(local_dir)], check=False)


def dl_qwen(model_id: str) -> None:
    _ms_download(model_id, DEST / "Qwen3")


def dl_cosyvoice() -> None:
    _git_clone("https://github.com/FunAudioLLM/CosyVoice.git", DEST / "CosyVoice")
    _ms_download("iic/CosyVoice2-0.5B", DEST / "CosyVoice" / "pretrained_models" / "CosyVoice2-0.5B")


def dl_musetalk() -> None:
    _git_clone("https://github.com/TMElyralab/MuseTalk.git", DEST / "MuseTalk")
    _hf_download("TMElyralab/MuseTalk", DEST / "MuseTalk" / "models" / "musetalk")


def dl_funasr() -> None:
    print("[dl] FunASR 通过 pip 安装；权重(paraformer-zh/vad/punc)首次运行时自动从 ModelScope 拉取")
    _pip_install(["funasr", "torchaudio"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all", help="逗号分隔：qwen,cosyvoice,musetalk,funasr 或 all")
    ap.add_argument("--qwen", default="Qwen/Qwen3-8B-AWQ", help="Qwen3 ModelScope id（本地8G显存建议 4B 或 8B-AWQ）")
    args = ap.parse_args()

    want = [c.strip() for c in args.only.split(",")] if args.only != "all" else COMPONENTS
    DEST.mkdir(parents=True, exist_ok=True)
    _ensure_libs()

    for comp in want:
        try:
            if comp == "qwen":
                dl_qwen(args.qwen)
            elif comp == "cosyvoice":
                dl_cosyvoice()
            elif comp == "musetalk":
                dl_musetalk()
            elif comp == "funasr":
                dl_funasr()
            else:
                print(f"[dl] 未知组件：{comp}")
        except Exception as e:  # noqa: BLE001
            print(f"[dl] {comp} 下载失败：{e}（可稍后单独重试 --only {comp}）")

    print("\n[dl] 完成。模型根目录：", DEST)
    print("[dl] 提醒：本地 8GB 显存无法同时跑 Qwen3+CosyVoice+MuseTalk；参考 README/CLOUD_DEPLOY 的本地组合建议。")


if __name__ == "__main__":
    main()
