"""本地 LLM 推理服务（Qwen3 AWQ，OpenAI 兼容端点）。

离线模式下的"答疑大脑"，替代云端 API：对外提供 ``POST /v1/chat/completions``
（含 SSE 流式）与 ``GET /v1/models``，与 ``models/llm.py::OpenAICompatLLM/Qwen3LLM``
的请求契约完全一致。

为什么不用 vLLM：本机 pip 装 vLLM 反复卡在依赖下载（30 分钟无进展）。
本项目单学生会话、无并发压力，transformers + AWQ 足够（首 token 约 0.5s）。

启动：
    LOCAL_LLM_DIR=<模型目录> ENGINE_PORT=8100 .venvs/llm/bin/python scripts/llm_engine_local.py
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Any

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_DIR = os.getenv("LOCAL_LLM_DIR", "/root/autodl-tmp/education/models_zoo/Qwen3-8B-AWQ")
MODEL_NAME = os.getenv("LOCAL_LLM_NAME", "Qwen3-8B-AWQ")
PORT = int(os.getenv("ENGINE_PORT", "8100"))
HOST = os.getenv("ENGINE_HOST", "127.0.0.1")
DEVICE = os.getenv("LOCAL_LLM_DEVICE", "cuda:0")
MAX_NEW_TOKENS = int(os.getenv("LOCAL_LLM_MAX_NEW", "1024"))

app = FastAPI(title="本地 LLM 服务（Qwen3 AWQ）")

_lock = threading.Lock()
_tok = None
_model = None


def _load() -> None:
    """加载模型（AWQ Int4，约 7~8G 显存）。"""
    global _tok, _model
    _tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, device_map=DEVICE, trust_remote_code=True, torch_dtype=torch.float16
    )
    _model.eval()
    print(f"[local-llm] loaded {MODEL_DIR} on {DEVICE}", flush=True)


def _build_prompt(messages: list[dict[str, Any]]) -> str:
    """套用 Qwen3 对话模板；关闭思考模式以降首响，模板不支持时逐级退化。"""
    try:
        return _tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        pass
    try:
        return _tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return "\n".join(f"{m.get('role')}: {m.get('content')}" for m in messages) + "\nassistant:"


class ChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[dict[str, Any]] = []
    temperature: float = 0.3
    max_tokens: int = 0
    stream: bool = False


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "engine": "local-llm", "model": MODEL_NAME, "loaded": _model is not None}


@app.get("/v1/models")
def models() -> dict:
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}


def _gen_kwargs(req: ChatRequest) -> tuple[dict[str, Any], int]:
    """构造 generate 参数；返回 (kwargs, prompt 长度)。"""
    prompt = _build_prompt(req.messages)
    inputs = _tok([prompt], return_tensors="pt").to(_model.device)
    prompt_len = int(inputs["input_ids"].shape[1])
    kwargs: dict[str, Any] = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs.get("attention_mask"),
        "max_new_tokens": req.max_tokens or MAX_NEW_TOKENS,
        "do_sample": req.temperature > 0.01,
        "repetition_penalty": 1.05,
        "pad_token_id": _tok.eos_token_id,
    }
    if kwargs["do_sample"]:
        kwargs["temperature"] = max(req.temperature, 0.01)
    return kwargs, prompt_len


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    kwargs, prompt_len = _gen_kwargs(req)

    if not req.stream:
        with _lock, torch.no_grad():
            out = _model.generate(**kwargs)
        text = _tok.decode(out[0][prompt_len:], skip_special_tokens=True)
        return {
            "id": cid, "object": "chat.completion", "created": created, "model": MODEL_NAME,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_len, "completion_tokens": len(out[0]) - prompt_len,
                      "total_tokens": len(out[0])},
        }

    def _sse():
        streamer = TextIteratorStreamer(_tok, skip_prompt=True, skip_special_tokens=True)
        run_kwargs = dict(kwargs)
        run_kwargs["streamer"] = streamer

        def _run() -> None:
            with _lock, torch.no_grad():
                _model.generate(**run_kwargs)

        threading.Thread(target=_run, daemon=True).start()
        head = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_NAME,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
        yield f"data: {json.dumps(head, ensure_ascii=False)}\n\n"
        for piece in streamer:
            chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                     "model": MODEL_NAME,
                     "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        tail = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_NAME,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(tail, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_sse(), media_type="text/event-stream")


def main() -> None:
    import uvicorn

    _load()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()