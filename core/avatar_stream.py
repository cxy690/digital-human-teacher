"""数字人视频落盘与 URL 暴露。

单句渲染视频通常 1~3MB。若以 base64 经 WebSocket 下发，前端一次 ``JSON.parse`` +
``base64`` 解码会造成明显掉帧（本机实测单次可达 90ms+），且大字符串内存难以回收。
改为「后端落盘 + ``/media/avatar_stream/<file>.mp4`` 由浏览器流式加载」后，
前端只需把 URL 赋给 ``<video>``，卡顿与内存问题一并消失。

* 目录：``data/avatar_stream``（与 ``server/app.py`` 的静态挂载一致）；
* 清理：目录内超过 ``MAX_FILES`` 个视频时，按修改时间删除最旧的；
* 失败：任何异常都返回 None，调用方退回 base64 下发，不影响播报。
"""
from __future__ import annotations

import base64
import time
from pathlib import Path

from config.logging import get_logger

log = get_logger("core.avatar_stream")

STREAM_DIR = Path("data/avatar_stream")
STREAM_URL_PREFIX = "/media/avatar_stream"
MAX_FILES = 160          # 目录内最多保留的视频数，超出按修改时间清理最旧的


def _cleanup_old() -> None:
    """目录内只保留最近 ``MAX_FILES`` 个视频，避免磁盘无限增长。

    ``idle*``（静音待机视频）是长期复用资源，不参与滚动清理。
    """
    try:
        files = sorted(
            (p for p in STREAM_DIR.glob("*.mp4") if not p.name.startswith("idle")),
            key=lambda p: p.stat().st_mtime,
        )
        if len(files) > MAX_FILES:
            for p in files[: len(files) - MAX_FILES]:
                try:
                    p.unlink()
                except OSError:
                    pass
    except Exception:  # noqa: BLE001 - 清理失败不影响主流程
        pass


def save_avatar_video(video_b64: str, tag: str = "") -> str | None:
    """把 base64 视频写盘并返回访问 URL（相对路径）；失败返回 None。"""
    if not video_b64:
        return None
    try:
        STREAM_DIR.mkdir(parents=True, exist_ok=True)
        safe_tag = "".join(ch for ch in (tag or "v") if ch.isalnum() or ch in ("-", "_")) or "v"
        name = f"{int(time.time() * 1000)}_{safe_tag}.mp4"
        (STREAM_DIR / name).write_bytes(base64.b64decode(video_b64))
        _cleanup_old()
        return f"{STREAM_URL_PREFIX}/{name}"
    except Exception as err:  # noqa: BLE001 - 落盘失败则调用方退回 base64
        log.warning("save avatar video failed: {}", err)
        return None
