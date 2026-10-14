"""为每位老师判定发声方式并写回档案。

用法（云端 tts venv，在 data/teachers 下执行）：
    python _set_voice_engine.py            # 按素材语音占比自动判定
    python _set_voice_engine.py --all-edge # 全部改用 edge 中文音色（克隆不可用时）

判定：素材有效语音占比 >= ``MIN_SPEECH_RATIO`` 才用 CosyVoice 零样本克隆，
否则退回 edge-tts 自带中文音色。
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import soundfile as sf

MIN_SPEECH_RATIO = 0.30

# 每位老师的 edge 中文音色（克隆不可用时的可靠兜底）
EDGE_VOICES = {
    "chinese": "zh-CN-XiaoxiaoNeural",     # 温婉女声
    "math": "zh-CN-YunxiNeural",           # 沉稳男声
    "english": "zh-CN-XiaoyiNeural",       # 明快女声
    "physics": "zh-CN-YunyangNeural",      # 严谨男声
    "chemistry": "zh-CN-XiaoyiNeural",
    "biology": "zh-CN-XiaoxiaoNeural",
    "politics": "zh-CN-YunxiaNeural",
    "morality": "zh-CN-YunjianNeural",
    "geography": "zh-CN-YunxiNeural",
    "history": "zh-CN-XiaoxiaoNeural",
}


def speech_ratio(path: str) -> float:
    try:
        data, sr = sf.read(path, dtype="float32")
    except Exception:
        return 0.0
    if data.ndim > 1:
        data = data.mean(axis=1)
    win = int(sr * 0.03)
    n = len(data) // win
    if n == 0:
        return 0.0
    rms = np.array([float(np.sqrt(np.mean(data[i * win:(i + 1) * win] ** 2))) for i in range(n)])
    return float(np.mean(rms > max(0.008, rms.max() * 0.08)))


def main() -> None:
    force_edge = "--all-edge" in sys.argv
    for p in sorted(glob.glob("*/teacher.json")):
        key = os.path.dirname(p)
        d = json.load(open(p, encoding="utf-8"))
        ratio = speech_ratio(os.path.join(key, "voice_ref.wav"))
        engine = "edge" if (force_edge or ratio < MIN_SPEECH_RATIO) else "clone"
        if engine == "edge":
            d["voice"] = EDGE_VOICES.get(key, d.get("voice") or "zh-CN-XiaoxiaoNeural")
        d["voice_engine"] = engine
        d["voice_speech_ratio"] = round(ratio, 3)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(d, fh, ensure_ascii=False, indent=2)
        print(f"{key:<10} 语音占比={ratio:<6.2f} 发声={engine:<7} 音色={d.get('voice') or '(学科默认)'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
