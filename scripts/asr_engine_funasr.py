"""FunASR 语音识别服务（ASR 引擎）。

对应《立项方案》3.3：ASR 用 **FunASR**（中文课堂场景识别优于通用模型，支持流式识别与 VAD，
学生提问的打断检测依赖其实现）。本项目客户端 ``models/asr.py::FunASRClient`` 约定
``POST {base}/recognize``（multipart 音频 + 表单 model/sample_rate/hotwords），本引擎包装之。

在**独立** conda 环境（torch + funasr）中运行：

    python scripts/asr_engine_funasr.py

监听 127.0.0.1:10095，契约：
    请求  POST /recognize  file=<audio.wav>  data={model, sample_rate, hotwords}
    响应  {"text": "识别结果"}

依赖：``pip install funasr torch torchaudio``；模型权重（paraformer-zh / fsmn-vad / ct-punc）
首次运行自动从 ModelScope 下载，无需手动放置。GPU 不可用时自动回落 CPU。
"""
from __future__ import annotations

import os
import re
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

DEVICE = os.getenv("FUNASR_DEVICE", "cuda:0")          # 无 GPU 时置 cpu
MODEL = os.getenv("FUNASR_MODEL", "iic/SenseVoiceSmall")
VAD_MODEL = os.getenv("FUNASR_VAD_MODEL", "fsmn-vad")
PUNC_MODEL = os.getenv("FUNASR_PUNC_MODEL", "ct-punc")
# SenseVoiceSmall：自带 VAD 与标点、输出含 <|zh|><|NEUTRAL|><|Speech|> 等标记，
# 走另一套调用参数并在返回前清洗；paraformer-zh 则需另挂 vad/punc 子模型。
IS_SENSEVOICE = "sensevoice" in MODEL.lower()
PORT = int(os.getenv("ENGINE_PORT", "10095"))
HOST = os.getenv("ENGINE_HOST", "127.0.0.1")

_lock = threading.Lock()
_model = None


def _load_model() -> None:
    """加载 FunASR AutoModel（一次性）。权重缺失时自动从 ModelScope 拉取。"""
    global _model
    from funasr import AutoModel  # noqa: PLC0415

    def _build(dev: str):
        """SenseVoice 自带 VAD/标点，不再挂子模型；paraformer 需要 vad+punc。"""
        if IS_SENSEVOICE:
            return AutoModel(model=MODEL, device=dev, disable_update=True)
        return AutoModel(model=MODEL, vad_model=VAD_MODEL, punc_model=PUNC_MODEL, device=dev)

    try:
        _model = _build(DEVICE)
    except Exception as err:  # noqa: BLE001 - GPU 不可用则回落 CPU
        print(f"[funasr] device {DEVICE} failed ({err}), fallback to cpu", flush=True)
        _model = _build("cpu")
    print(f"[funasr] loaded: {MODEL}", flush=True)


def _clean_text(text: str) -> str:
    """剥离 SenseVoice 输出的特殊标记（``<|zh|>`` ``<|NEUTRAL|>`` ``<|Speech|>`` 等）。"""
    return re.sub(r"<\|[^|]*\|>", "", text or "").strip()


app = FastAPI(title="FunASR 语音识别服务", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "engine": "funasr", "model": MODEL,
            "device": DEVICE, "model_loaded": _model is not None}


@app.post("/recognize")
async def recognize(
    file: UploadFile = File(...),
    model: str = Form(MODEL),
    sample_rate: int = Form(16000),
    hotwords: str = Form(""),
) -> dict:
    if _model is None:
        raise HTTPException(status_code=503, detail="FunASR 未加载")
    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="音频为空")

    tmp = tempfile.mkdtemp(prefix="funasr_")
    wav_path = os.path.join(tmp, "in.wav")
    Path(wav_path).write_bytes(audio)

    with _lock:                                          # 串行化推理
        if IS_SENSEVOICE:
            kwargs: dict = {"input": wav_path, "cache": {}, "language": "auto",
                            "use_itn": True, "batch_size_s": 60}
        else:
            kwargs = {"input": wav_path, "batch_size_s": 300}
        if hotwords:
            kwargs["hotword"] = hotwords
        res = _model.generate(**kwargs)
    text = ""
    if res:
        text = res[0].get("text", "") if isinstance(res[0], dict) else str(res[0])
    return {"text": _clean_text(text), "model": model, "sample_rate": sample_rate}


def main() -> None:
    import uvicorn

    _load_model()
    print(f"[funasr] http://{HOST}:{PORT}  (POST /recognize)", flush=True)
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
