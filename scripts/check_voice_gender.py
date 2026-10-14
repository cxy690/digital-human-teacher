"""核对「老师形象性别」与「实际发声性别」是否一致。

为什么需要它
------------
本项目有两条发声链路，音色性别都可能出错，而且**看配置看不出来**：

* ``voice_engine = "clone"``（CosyVoice 零样本克隆）：音色性别完全由
  ``voice_ref.wav`` 决定。历史上出现过「变体老师的 voice_ref.wav 直接抄了
  基础老师的（md5 一致）」——于是女形象用了男声干声，克隆出来就是男声。
* ``voice_engine = "edge"``（edge-tts 预置音色）：由 ``voice`` 音色名决定
  （微软命名约定：``Yun*`` 男声、``Xiao*`` 女声），这一路本是确定的。

做法：不看配置，直接对**真正会播给学生听的那条音频**做基频（F0）分析，
再和档案里的 ``gender`` 对比。

F0 判据（普通话成人）：男 ~85-155Hz，女 ~165-255Hz；155~165 为重叠区，
判为「不确定」而不是不匹配，避免误报。

跑在哪
------
需要同时能读到 ``data/teachers/*/voice_ref.wav`` 与访问 CosyVoice 服务，
因此**在云端跑**（本地没有 :8090）：

    python scripts/check_voice_gender.py                 # 全部老师
    python scripts/check_voice_gender.py physics math     # 只看指定老师

可用环境变量覆盖：``EDU_ROOT``（项目根）、``COSYVOICE_API``（默认
``http://127.0.0.1:8090``）。
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import pathlib
import subprocess
import sys
import urllib.request
import wave

import numpy as np

SR = 16000
TEXT = "同学们好，我们开始上课，今天讲一个知识点。"
API = os.getenv("COSYVOICE_API", "http://127.0.0.1:8090")
ROOT = pathlib.Path(os.getenv("EDU_ROOT") or pathlib.Path(__file__).resolve().parents[1])


def _to_pcm(b: bytes) -> bytes:
    """任意音频字节 -> 16k 单声道 WAV（edge-tts 返回 MP3，需先转码）。"""
    if b[:4] == b"RIFF":
        return b
    import imageio_ffmpeg

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    return subprocess.run(
        [ff, "-v", "error", "-i", "pipe:0", "-vn", "-ac", "1", "-ar", str(SR), "-f", "wav", "pipe:1"],
        input=b, capture_output=True, check=False,
    ).stdout


def read_pcm(b: bytes) -> np.ndarray:
    b = _to_pcm(b)
    with wave.open(io.BytesIO(b)) as w:
        n, ch, sr = w.getnframes(), w.getnchannels(), w.getframerate()
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != SR and len(x) > 1:
        idx = np.linspace(0, len(x) - 1, int(len(x) * SR / sr))
        x = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
    return x


def median_f0(x: np.ndarray) -> float:
    """自相关法估基频，取浊音帧中位数（对音色性别判定足够稳）。"""
    if len(x) < SR // 10:
        return 0.0
    frame, hop = int(SR * 0.04), int(SR * 0.02)
    lo, hi = int(SR / 400), int(SR / 60)
    vals: list[float] = []
    for i in range(0, len(x) - frame, hop):
        seg = x[i:i + frame]
        seg = seg - seg.mean()
        if float(np.sqrt(np.mean(seg ** 2))) < 0.01:
            continue
        ac = np.correlate(seg, seg, "full")[frame - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        win = ac[lo:hi]
        if not len(win):
            continue
        k = int(np.argmax(win)) + lo
        if ac[k] < 0.3:                     # 周期性不足 -> 不当浊音
            continue
        vals.append(SR / k)
    return float(np.median(vals)) if vals else 0.0


def f0_gender(f0: float) -> str:
    if f0 <= 0:
        return "?"
    if f0 < 155:
        return "male"
    if f0 > 165:
        return "female"
    return "?"                              # 男女重叠区


def edge_name_gender(name: str) -> str:
    """微软预制音色的性别：Yun*=男声，Xiao*=女声。"""
    n = (name or "").lower()
    if "yun" in n:
        return "male"
    if "xiao" in n:
        return "female"
    return "?"


def synth_clone(ref: pathlib.Path, ref_text: str) -> bytes:
    body = json.dumps({"text": TEXT, "stream": False,
                       "ref_audio": str(ref), "ref_text": ref_text}).encode()
    req = urllib.request.Request(f"{API}/api/v1/tts", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def synth_edge(voice: str) -> bytes:
    import edge_tts

    async def go() -> bytes:
        c = edge_tts.Communicate(TEXT, voice or "zh-CN-XiaoxiaoNeural")
        buf = bytearray()
        async for ch in c.stream():
            if ch.get("type") == "audio":
                buf += ch["data"]
        return bytes(buf)

    return asyncio.run(go())


def main() -> int:
    wanted = {a.strip() for a in sys.argv[1:] if a.strip()}
    rows, bad = [], []
    for cfg in sorted((ROOT / "data" / "teachers").glob("*/teacher.json")):
        key = cfg.parent.name
        if wanted and key not in wanted:
            continue
        d = json.loads(cfg.read_text(encoding="utf-8"))
        declared = str(d.get("gender") or "").strip().lower()
        eng = str(d.get("voice_engine") or "")
        voice = str(d.get("voice") or "")
        ref = str(d.get("voice_ref_audio") or "")
        refp = pathlib.Path(ref) if ref.startswith("/") else (ROOT / ref)

        ref_f0 = 0.0
        if refp.is_file():
            try:
                ref_f0 = median_f0(read_pcm(refp.read_bytes()))
            except Exception:  # noqa: BLE001
                ref_f0 = 0.0

        out_f0, note = 0.0, ""
        try:
            b = synth_clone(refp, str(d.get("voice_ref_text") or "")) if eng == "clone" \
                else synth_edge(voice)
            out_f0 = median_f0(read_pcm(b))
            note = f"{len(b) // 1024}KB"
        except Exception as err:  # noqa: BLE001
            note = f"合成失败 {type(err).__name__}"

        og = f0_gender(out_f0)
        heed = edge_name_gender(voice) if eng == "edge" else "-"
        verdict = "?"
        if out_f0 > 0 and declared:
            verdict = "OK" if og == declared else ("UNCERTAIN" if og == "?" else "MISMATCH")
        if verdict == "MISMATCH":
            bad.append(key)
        rows.append((key, declared, eng, voice, ref_f0, out_f0, og, heed, verdict, note))

    print(f"{'key':<17}{'档案':<7}{'引擎':<7}{'干声F0':<8}{'输出F0':<8}{'输出性别':<9}"
          f"{'edge名':<8}{'结论':<11}备注")
    print("-" * 116)
    for r in rows:
        key, declared, eng, voice, ref_f0, out_f0, og, heed, verdict, note = r
        warn = "  <- edge 音色名与档案性别不符" if (heed not in ("-", "?") and heed != declared) else ""
        print(f"{key:<17}{declared or '-':<7}{eng:<7}{ref_f0:<8.1f}{out_f0:<8.1f}{og:<9}"
              f"{heed:<8}{verdict:<11}{note}{warn}")

    print()
    print("MISMATCH（性别不符）:", bad or "无")
    print("缺 gender 字段:", [r[0] for r in rows if not r[1]] or "无")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
