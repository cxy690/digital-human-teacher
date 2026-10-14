"""校准 CosyVoice 克隆音色：修正 ``ref_text``，并体检克隆质量。

为什么需要
----------
CosyVoice zero-shot 克隆要求 ``ref_text`` 与干声内容**逐字一致**。此前这些文本
多为手工猜写（如「九地理，选国商低心理。」），与录音对不上时克隆会退化：
轻则发音/口音跑偏（听起来不像普通话），重则**输出被截断**
（实测同一句话：官方样例干声 134KB，错误 ref_text 的教师干声只有 3.8KB ≈ 0.08s）。

做什么（幂等，可反复执行）
--------------------------
1. 用云端 FunASR（``/recognize``）把每位老师的干声**重新转写**，写回 ``voice_ref_text``；
2. 对每位克隆老师做合成体检（短句 + 中句，测音频时长）：
   时长明显不足 → 判定该干声不可靠，``voice_engine`` 自动退回 ``edge`` 标准普通话，
   保证「宁要清楚的普通话，不要含糊的克隆」。

用法（云端）::

    .venvs/app/bin/python scripts/calibrate_clone_voices.py
    # 可选环境变量：ASR_API_BASE / TTS_API_BASE / DRY_RUN=1
"""
from __future__ import annotations

import json
import os
import sys
import wave
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
TEACHERS = ROOT / "data" / "teachers"

ASR_BASE = os.getenv("ASR_API_BASE", "http://127.0.0.1:10095").rstrip("/")
TTS_BASE = os.getenv("TTS_API_BASE", "http://127.0.0.1:8090").rstrip("/")
DRY_RUN = os.getenv("DRY_RUN") == "1"

# 体检用的测试句：短句最容易被「坏干声」带崩，所以短句、中句各测一条
PROBE_SHORT = "同学们好，我们开始上课。"
PROBE_MED = "下面我们来看这道例题的解题思路和关键步骤。"

SAMPLE_RATE = 24000        # 与 tts_engine_cosyvoice.py 的输出采样率一致
SPEECH_CPS = 4.0           # 正常朗读约 4 字/秒
MIN_RATIO = 0.4            # 实际时长低于「预期 x 该比例」判为截断


def _asr(path: Path) -> str:
    """把干声交给 FunASR 转写，返回带标点的识别文本。"""
    with httpx.Client(timeout=180) as client:
        resp = client.post(
            f"{ASR_BASE}/recognize",
            files={"file": (path.name, path.read_bytes(), "audio/wav")},
        )
        resp.raise_for_status()
        return str(resp.json().get("text") or "").strip()


def _tts_seconds(text: str, ref_audio: str, ref_text: str) -> float:
    """克隆合成一句并返回音频时长（秒）；失败返回 0。"""
    payload = {"text": text, "stream": False, "ref_audio": ref_audio, "ref_text": ref_text}
    try:
        with httpx.Client(timeout=180) as client:
            resp = client.post(f"{TTS_BASE}/api/v1/tts", json=payload)
            resp.raise_for_status()
        tmp = ROOT / "data" / "cache" / "_clone_probe.wav"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(resp.content)
        with wave.open(str(tmp)) as w:
            return w.getnframes() / float(w.getframerate() or SAMPLE_RATE)
    except Exception as err:  # noqa: BLE001 - 单个老师失败不影响其它
        print(f"    合成失败：{type(err).__name__}: {err}")
        return 0.0


def _expected(text: str) -> float:
    """这句话正常应该读多久（秒）。"""
    return max(1.0, len(text) / SPEECH_CPS)


def main() -> None:
    profiles: list[tuple[Path, dict]] = []
    for path in sorted(TEACHERS.glob("*/teacher.json")):
        profiles.append((path, json.loads(path.read_text(encoding="utf-8"))))

    # ---------- 1. 修正 ref_text ----------
    print("== 第 1 步：FunASR 重转写干声，修正 voice_ref_text ==")
    for path, data in profiles:
        ref = str(data.get("voice_ref_audio") or "")
        p = Path(ref) if Path(ref).is_absolute() else (ROOT / ref)
        if not ref or not p.is_file():
            continue
        try:
            text = _asr(p)
        except Exception as err:  # noqa: BLE001
            print(f"  {path.parent.name:20s} ASR 失败：{err}")
            continue
        old = str(data.get("voice_ref_text") or "")
        if not text or text == old:
            print(f"  {path.parent.name:20s} ref_text 无变化")
            continue
        print(f"  {path.parent.name:20s} {old!r}")
        print(f"  {'':20s}   -> {text!r}")
        if not DRY_RUN:
            data["voice_ref_text"] = text
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---------- 2. 克隆质量体检 ----------
    print("\n== 第 2 步：克隆合成体检（时长不足 -> 退回 edge 标准普通话）==")
    for path, data in profiles:
        if str(data.get("voice_engine") or "") != "clone":
            continue
        key = path.parent.name
        ref = str(data.get("voice_ref_audio") or "")
        ref_text = str(data.get("voice_ref_text") or "")
        p = Path(ref) if Path(ref).is_absolute() else (ROOT / ref)
        if not ref or not p.is_file() or not ref_text:
            continue
        secs = [
            _tts_seconds(PROBE_SHORT, ref, ref_text),
            _tts_seconds(PROBE_MED, ref, ref_text),
        ]
        exp = [_expected(PROBE_SHORT), _expected(PROBE_MED)]
        ok = all(s >= e * MIN_RATIO for s, e in zip(secs, exp))
        mark = "ok" if ok else "异常"
        print(f"  {key:20s} 短句={secs[0]:.2f}s/{exp[0]:.2f}s  中句={secs[1]:.2f}s/{exp[1]:.2f}s  [{mark}]")
        if ok or DRY_RUN:
            continue
        data["voice_engine"] = "edge"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"  {key:20s} 干声不可靠 -> voice_engine=edge")

    print("\n完成。改动过档案后需重启主应用（python scripts/cloud_ctl.py restart_app）。")


if __name__ == "__main__":
    if os.name == "nt":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
