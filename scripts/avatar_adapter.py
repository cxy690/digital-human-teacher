"""数字人渲染适配器（参考实现）：把本项目约定的 ``POST /render`` 协议，
转发到本地 MuseTalk / metahuman-stream 类渲染服务。

为什么需要它：
* 各开源数字人项目的 API 形态不一（有的收音频、有的收音频流、有的走 WebRTC 房间），
  本项目只约定一种最小契约，差异都在这个适配器里消化；
* 没有 GPU 时可用 ``AVATAR_STUB_VIDEO`` 指定一个示例 mp4，
  让整条「AI 说话 -> ai_avatar 事件 -> 前端播放视频」链路先跑通再换真引擎。

契约：
    请求  POST /render  {"audio": "<base64 音频>", "text": "这一句的话术"}
    响应  {"video": "<base64 mp4>", "mime": "video/mp4"}  或  {"stream_url": "..."}

启动：
    python scripts/avatar_adapter.py             # 默认监听 127.0.0.1:8010
    AVATAR_STUB_VIDEO=data/raw/avatar_sample.mp4 python scripts/avatar_adapter.py
    AVATAR_UPSTREAM=http://127.0.0.1:8011/render python scripts/avatar_adapter.py
"""
from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

APP_PORT = int(os.getenv("ADAPTER_PORT", "8010"))
UPSTREAM = os.getenv("AVATAR_UPSTREAM", "").strip()
STUB_VIDEO = os.getenv("AVATAR_STUB_VIDEO", "").strip()
UPSTREAM_TIMEOUT = float(os.getenv("ADAPTER_TIMEOUT", "60"))

app = FastAPI(title="AI 教师数字人渲染适配器", version="0.1.0")


class RenderRequest(BaseModel):
    audio: str = ""
    text: str = ""


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "mode": "upstream" if UPSTREAM else ("stub" if STUB_VIDEO else "unconfigured"),
        "upstream": UPSTREAM or None,
        "stub_video": STUB_VIDEO or None,
    }


@app.post("/render")
async def render(req: RenderRequest) -> dict:
    """渲染一句音频，返回视频（base64）或流地址。"""
    started = time.perf_counter()

    if UPSTREAM:
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
            resp = await client.post(
                UPSTREAM, json={"audio": req.audio, "text": req.text}
            )
            resp.raise_for_status()
            data = resp.json()
        result = _normalize(data)
        result["latency_ms"] = int((time.perf_counter() - started) * 1000)
        return result

    if STUB_VIDEO:
        path = Path(STUB_VIDEO)
        if not path.exists():
            raise HTTPException(status_code=503, detail=f"stub video not found: {path}")
        return {
            "mode": "video",
            "video": base64.b64encode(path.read_bytes()).decode("ascii"),
            "mime": "video/mp4",
            "stub": True,
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }

    raise HTTPException(
        status_code=503,
        detail="未配置渲染引擎：请设置 AVATAR_UPSTREAM（真实引擎）或 AVATAR_STUB_VIDEO（示例视频）",
    )


# 上游返回字段的常见别名（不同引擎/云平台命名不一，统一在这里归一化）
_STREAM_KEYS = ("stream_url", "streamUrl", "video_url", "videoUrl", "url")
_VIDEO_KEYS = ("video", "video_base64", "videoBase64", "data")


def _normalize(data: dict) -> dict:
    """把上游各种返回格式归一化成本项目契约。

    * ``stream_url``（含 video_url / url 等别名）透传给前端 <video> 直接播放；
    * base64 视频（video / video_base64 / data）归一化并剥离 data URI 前缀；
    * 兼容 ``{"code":0,"data":{...}}`` 这类信封结构：内层为 dict 时下沉一层。
    """
    if not isinstance(data, dict):
        return {"mode": "video", "video": "", "mime": "video/mp4"}
    for key in ("data", "result"):                     # 信封结构下沉一层
        inner = data.get(key)
        if isinstance(inner, dict):
            data = {**data, **inner}
            break
    for key in _STREAM_KEYS:
        url = data.get(key)
        if isinstance(url, str) and url.strip():
            return {"mode": "video", "stream_url": url.strip(), "text": ""}
    video = ""
    for key in _VIDEO_KEYS:
        value = data.get(key)
        if isinstance(value, str) and value:
            video = value
            break
    if video.startswith("data:"):                      # data URI -> 去掉前缀
        head, _, payload = video.partition(",")
        mime = head.split(";")[0].removeprefix("data:") or "video/mp4"
        return {"mode": "video", "video": payload, "mime": mime}
    return {"mode": "video", "video": video, "mime": data.get("mime", "video/mp4")}


def main() -> None:
    import uvicorn

    print("=" * 66, flush=True)
    print(f"  数字人渲染适配器: http://127.0.0.1:{APP_PORT}   (POST /render)", flush=True)
    print(f"  模式: {'上游转发 ' + UPSTREAM if UPSTREAM else ('示例视频 ' + STUB_VIDEO if STUB_VIDEO else '未配置')}", flush=True)
    print("=" * 66, flush=True)
    uvicorn.run(app, host="127.0.0.1", port=APP_PORT)


if __name__ == "__main__":
    main()
