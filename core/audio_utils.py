"""音频后处理：给 TTS 音频尾部追加纯静音。

用途（数字人平滑收尾）：
* MuseTalk / LatentSync 等口型引擎在处理**静音段**时，嘴部会趋于闭合或稳定；
* 因此给每句 TTS 音频尾部补一小段静音后送引擎，生成的视频就会以「闭嘴稳定帧」收尾；
* 前端在讲完后取该末帧（即静音段最后一帧）与待机底片第一帧做短交叉淡化，即可无缝回待机。

ffmpeg 不可用 / 处理失败时**原样返回**，绝不影响正常播报。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from config.logging import get_logger

log = get_logger("core.audio_utils")


def _resolve_ffmpeg() -> str | None:
    """定位 ffmpeg：FFMPEG_PATH -> imageio_ffmpeg -> PATH 中的 ffmpeg。"""
    env = os.getenv("FFMPEG_PATH", "").strip()
    if env and Path(env).is_file():
        return env
    try:
        import imageio_ffmpeg  # noqa: PLC0415 - 可选依赖

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - 可选依赖缺失
        pass
    return shutil.which("ffmpeg")


def append_tail_silence(audio: bytes, seconds: float = 0.6) -> bytes:
    """在音频末尾追加 ``seconds`` 秒纯静音，返回 mp3 字节（失败则原样返回）。

    实现：``anullsrc`` 生成静音轨，与原音频统一重采样后 ``concat``。
    统一到 44.1kHz / 单声道 / s16，避免原始 mp3 采样率不一致导致拼接失败。
    """
    if not audio or seconds <= 0:
        return audio
    ffmpeg = _resolve_ffmpeg()
    if not ffmpeg:
        return audio
    tmp = tempfile.mkdtemp(prefix="tts_pad_")
    try:
        src = os.path.join(tmp, "in.mp3")
        out = os.path.join(tmp, "out.mp3")
        Path(src).write_bytes(audio)
        filt = (
            "[0:a]aresample=44100,aformat=sample_fmts=s16:channel_layouts=mono[a0];"
            "[1:a]aresample=44100,aformat=sample_fmts=s16:channel_layouts=mono[a1];"
            "[a0][a1]concat=n=2:v=0:a=1[out]"
        )
        subprocess.run(
            [
                ffmpeg, "-y",
                "-i", src,
                "-f", "lavfi", "-t", str(seconds), "-i", "anullsrc=r=44100:cl=mono",
                "-filter_complex", filt, "-map", "[out]",
                "-c:a", "libmp3lame", "-q:a", "4",
                out,
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        data = Path(out).read_bytes()
        return data or audio
    except Exception as err:  # noqa: BLE001 - 补静音失败不影响播报
        log.warning("append tail silence failed, use original audio: {}", err)
        return audio
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
