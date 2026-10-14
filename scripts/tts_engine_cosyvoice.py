"""CosyVoice 语音合成服务（TTS 引擎，克隆音色）—— 支持 CosyVoice3 / CosyVoice2。

对应《立项方案》3.3：TTS 首选 **CosyVoice**（流式推理、克隆授课老师音色，首包 200-300ms）。
本项目客户端 ``models/tts.py::CosyVoiceTTS`` 约定 ``POST {base}/api/v1/tts``，
本引擎把该契约包装到 CosyVoice 官方仓库（``FunAudioLLM/CosyVoice``）之上。

在**独立** conda 环境（torch + CosyVoice 依赖）中运行：

    export COSYVOICE_DIR=/root/autodl-tmp/education/models_zoo/CosyVoice
    export COSYVOICE_MODEL=$COSYVOICE_DIR/pretrained_models/Fun-CosyVoice3-0.5B   # 可选，默认自动挑
    python scripts/tts_engine_cosyvoice.py

监听 127.0.0.1:8090，契约：
    请求  POST /api/v1/tts {"text":"...", "stream":true, "speaker":"", "ref_audio":"/x.wav", "ref_text":"..."}
    响应  audio/wav 裸字节（或 {"audio":"<base64>"}）

版本选择（**默认自动**，也可用环境变量强制）：
* 目录里有 ``cosyvoice3.yaml`` -> CosyVoice3（官方权重 ``FunAudioLLM/Fun-CosyVoice3-0.5B-2512``）；
* 有 ``cosyvoice2.yaml``      -> CosyVoice2（``iic/CosyVoice2-0.5B``）；
* 有 ``cosyvoice.yaml``       -> CosyVoice 1（``CosyVoice-300M`` 系列）。
环境变量 ``COSYVOICE_VERSION=3`` 可强制指定；``COSYVOICE_TEXT_FRONTEND=1`` 打开文本规范化前端。

依赖：需先克隆 CosyVoice 仓库（COSYVOICE_DIR）并下载对应权重（COSYVOICE_MODEL）。
zero-shot 克隆：传 ref_audio + ref_text 即用参考干声克隆音色；不传则用仓库自带样本。
"""
from __future__ import annotations

import io
import os
import sys
import threading
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

COSYVOICE_DIR = os.getenv("COSYVOICE_DIR", r"/root/CosyVoice")


def _pick_model_dir() -> str:
    """模型目录：``COSYVOICE_MODEL`` 优先；否则在 pretrained_models 下自动挑（优先 3）。

    「自动挑」是为了换模型时**不用改任何启动脚本**：把 CosyVoice3 权重放进去、
    重启引擎就生效；没有 3 时自然回落到 2/1。
    """
    explicit = os.getenv("COSYVOICE_MODEL", "").strip()
    if explicit:
        return explicit
    base = Path(COSYVOICE_DIR) / "pretrained_models"
    for name in (
        "Fun-CosyVoice3-0.5B", "Fun-CosyVoice3-0.5B-2512", "CosyVoice3-0.5B",
        "CosyVoice2-0.5B",
    ):
        candidate = base / name
        if candidate.is_dir():
            return str(candidate)
    return str(base / "CosyVoice2-0.5B")


MODEL_DIR = _pick_model_dir()


def _detect_version(model_dir: str) -> int:
    """按模型目录里的 yaml 判断版本（与官方 ``AutoModel`` 的判据一致）。"""
    path = Path(model_dir)
    if (path / "cosyvoice3.yaml").is_file():
        return 3
    if (path / "cosyvoice2.yaml").is_file():
        return 2
    if (path / "cosyvoice.yaml").is_file():
        return 1
    return 2


VERSION = int(os.getenv("COSYVOICE_VERSION", "0") or 0) or _detect_version(MODEL_DIR)

# CosyVoice3 的 prompt_text（参考音频的逐字文本）必须带这段指令前缀，官方 example.py 同款；
# 少了它，音色克隆/韵律会明显异常（模型不知道该把 prompt 当指令还是当参考文本）。
V3_PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"

# 零样本参考：未传 ref 时用仓库自带样本（CosyVoice2/3 无内置音色，须给参考音频）
DEFAULT_REF_WAV = os.getenv("TTS_DEFAULT_REF_WAV", os.path.join(COSYVOICE_DIR, "asset/zero_shot_prompt.wav"))
DEFAULT_REF_TEXT = os.getenv("TTS_DEFAULT_REF_TEXT", "希望你以后能够做的比我还好喲。")
PORT = int(os.getenv("ENGINE_PORT", "8090"))
HOST = os.getenv("ENGINE_HOST", "127.0.0.1")
SAMPLE_RATE = 24000                                    # CosyVoice2/3 输出采样率（模型 config 为准）
# 文本规范化前端：需从 ModelScope 拉 wetext/ttsfrd 资源，离线时会卡在下载，
# 故默认关闭；中文讲解不做规范化也够用，需要时置 1 打开。
TEXT_FRONTEND = os.getenv("COSYVOICE_TEXT_FRONTEND", "0") == "1"

sys.path.insert(0, COSYVOICE_DIR)
sys.path.insert(0, os.path.join(COSYVOICE_DIR, "third_party/Matcha-TTS"))

_lock = threading.Lock()
_model = None


def _load_model() -> None:
    """加载 CosyVoice（一次性）。构造参数按版本对齐（3 与 2 的签名不同）。

    ``COSYVOICE_FP16=0`` 可关闭半精度：部分环境下 fp16 会让输出变成
    「定长乱码」（听感像外语/胡话），关掉即可恢复清晰中文。
    """
    global _model
    fp16 = os.getenv("COSYVOICE_FP16", "1") == "1" and torch.cuda.is_available()
    if VERSION >= 3:
        # CosyVoice3(model_dir, load_trt=False, load_vllm=False, fp16=False, trt_concurrent=1)
        # 注意：3 没有 load_jit 参数，传了会 TypeError。
        from cosyvoice.cli.cosyvoice import CosyVoice3  # noqa: PLC0415

        _model = CosyVoice3(MODEL_DIR, load_trt=False, fp16=fp16)
    elif VERSION == 2:
        from cosyvoice.cli.cosyvoice import CosyVoice2  # noqa: PLC0415

        _model = CosyVoice2(MODEL_DIR, load_jit=False, load_trt=False, fp16=fp16)
    else:
        from cosyvoice.cli.cosyvoice import CosyVoice  # noqa: PLC0415

        _model = CosyVoice(MODEL_DIR, load_jit=False, load_trt=False, fp16=fp16)

    if not fp16:
        # 关键修复：fp16=False 时权重并不会自动变成 float32 —— 权重文件本身是 bf16
        # （4090 上实测），推理会抛 "mat1 and mat2 must have the same dtype,
        # but got Float and BFloat16"，且请求永久挂起（表现为 TTS 超时、老师不出声）。
        # 这里显式把各子模型统一转成 float32，与输入 dtype 对齐。
        cast = 0
        for attr in ("model", "flow", "hift", "llm", "text_encoder"):
            sub = getattr(_model, attr, None)
            if hasattr(sub, "float"):
                try:
                    sub.float()
                    cast += 1
                except Exception as err:  # noqa: BLE001 - 单个子模型失败不影响其它
                    print(f"[cosyvoice] cast {attr} failed: {err}", flush=True)
        if hasattr(_model, "float"):
            try:
                _model.float()
                cast += 1
            except Exception:  # noqa: BLE001
                pass
        print(f"[cosyvoice] 已统一转为 float32（{cast} 个子模型）", flush=True)
    print(
        f"[cosyvoice] loaded: {MODEL_DIR} version={VERSION} fp16={fp16} "
        f"sr={_sample_rate()} frontend={TEXT_FRONTEND}",
        flush=True,
    )


def _sample_rate() -> int:
    """输出采样率：以模型 config 为准（CosyVoice3 可能与 2 不同）。"""
    try:
        return int(getattr(_model, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
    except (TypeError, ValueError):
        return SAMPLE_RATE


def _tensor_to_wav_bytes(audio: torch.Tensor) -> bytes:
    """把模型输出张量编码为 wav 字节（返回给客户端）。"""
    import soundfile as sf  # noqa: PLC0415

    buf = io.BytesIO()
    data = audio.squeeze().cpu().numpy()
    sf.write(buf, data, _sample_rate(), format="WAV")
    return buf.getvalue()


def _prompt_text(raw: str) -> str:
    """CosyVoice3 的参考文本要加指令前缀（官方约定），已经带过就不再重复加。"""
    text = (raw or "").strip()
    if VERSION >= 3 and not text.startswith(V3_PROMPT_PREFIX):
        return V3_PROMPT_PREFIX + text
    return text


def _pick_speaker(model) -> str:
    """取一个可用的内置音色名（兼容 list_available_spks 为方法/属性/缺失三种情况）。"""
    fn = getattr(model, "list_available_spks", None)
    try:
        spks = fn() if callable(fn) else fn
    except Exception:  # noqa: BLE001 - 任何异常都回退默认音色
        spks = None
    return spks[0] if spks else "中文女"


class TTSRequest(BaseModel):
    text: str = ""
    stream: bool = False
    speaker: str = ""
    ref_audio: str = ""
    ref_text: str = ""


app = FastAPI(title="CosyVoice TTS 服务", version="0.2.0")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "engine": f"cosyvoice{VERSION}",
        "version": VERSION,
        "model": MODEL_DIR,
        "sample_rate": _sample_rate(),
        "text_frontend": TEXT_FRONTEND,
        "model_loaded": _model is not None,
    }


@app.post("/api/v1/tts")
def tts(req: TTSRequest) -> Response:
    if not req.text:
        raise HTTPException(status_code=400, detail="text 不能为空")
    if _model is None:
        raise HTTPException(status_code=503, detail="CosyVoice 未加载（检查 COSYVOICE_DIR/COSYVOICE_MODEL）")

    with _lock:                                          # 模型非线程安全，串行化
        chunks = []
        ref_audio = req.ref_audio or DEFAULT_REF_WAV
        ref_text = _prompt_text(req.ref_text or DEFAULT_REF_TEXT)
        # text_frontend 默认 False：跳过 wetext/ttsfrd 文本规范化前端。
        # 该前端需联网从 ModelScope 拉模型，离线/无 token 时会卡在下载（"no frontend
        # is avaliable" → 一直重试），导致整个 TTS 不可用。
        if ref_audio and ref_text and os.path.isfile(ref_audio):   # zero-shot 音色克隆
            # 第三参 ``prompt_wav`` 传**音频文件路径**（官方 example.py 同样传路径，
            # frontend 内部自行 load/resample 到模型采样率）；传张量会报 Invalid file。
            gen = _model.inference_zero_shot(req.text, ref_text, ref_audio,
                                             stream=req.stream, text_frontend=TEXT_FRONTEND)
        else:                                            # 内置音色（CosyVoice2/3 通常无内置 spk）
            speaker = req.speaker or _pick_speaker(_model)
            gen = _model.inference_sft(req.text, speaker, stream=req.stream,
                                       text_frontend=TEXT_FRONTEND)
        for out in gen:
            chunks.append(out["tts_speech"])

    if not chunks:
        raise HTTPException(status_code=500, detail="合成失败：无音频输出")
    audio = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
    wav = _tensor_to_wav_bytes(audio)
    return Response(content=wav, media_type="audio/wav")


def main() -> None:
    import uvicorn

    if not Path(COSYVOICE_DIR).is_dir():
        raise SystemExit(f"未找到 CosyVoice 仓库：{COSYVOICE_DIR}（请设 COSYVOICE_DIR）")
    if not Path(MODEL_DIR).is_dir():
        raise SystemExit(f"未找到模型权重：{MODEL_DIR}（请设 COSYVOICE_MODEL）")
    _load_model()
    print(f"[cosyvoice] http://{HOST}:{PORT}  (POST /api/v1/tts)", flush=True)
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
