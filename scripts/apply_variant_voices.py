# -*- coding: utf-8 -*-
"""给「第二类（变体）老师」接上克隆音色，去掉 edge-tts 的机械感。

背景
----
同科多老师里，第一类（math/physics/...）建档案时从素材视频抽了干声，走 CosyVoice 克隆
零样本克隆；而**第二类变体（math_strict/physics_lab/...）的底片视频几乎没有音轨**
（实测峰值 0.004~0.20，有声帧 0.0%~6.7%），抽不出可用人声，只能设成 ``edge``
（edge-tts 自带音色）—— 这就是「听起来很机械」的根因。

策略（按干声质量分两档）
------------------------
1. 同学科主老师有可用干声 -> **复用主老师干声**，设 ``clone``。
   同科两位老师音色一致，符合「同一位老师的不同讲课风格」这一变体语义。
2. 同学科主老师干声也偏弱（geography/history/morality）或无主老师（science_inquiry）
   -> 设 ``clone`` 但**留空 voice_ref_audio**，回落到全局 ``TTS_REF_AUDIO``
   （算力云上的 data/teacher_voice/teacher_ref_8s.wav，真人教师音色）。
   真人克隆音色再弱也比 edge-tts 的机械音自然，且不会像「干声太差硬克隆」那样音色错乱。

用法
----
    python scripts/apply_variant_voices.py            # 应用（改 teacher.json）
    python scripts/apply_variant_voices.py --dry       # 只预览
    python scripts/apply_variant_voices.py --revert    # 回退到 edge
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEACHERS_DIR = ROOT / "data" / "teachers"

# 变体 -> 同学科主老师（主老师干声可用时复用其音色）
VARIANT_TO_BASE: dict[str, str] = {
    "math_strict": "math",
    "english_immersive": "english",
    "chinese_logic": "chinese",
    "biology_explore": "biology",
    "chemistry_micro": "chemistry",
    "physics_lab": "physics",
}

# 主老师干声也偏弱 / 没有主老师 -> 留空干声，走全局 TTS_REF_AUDIO
FALLBACK_CLONE: tuple[str, ...] = (
    "geography_map", "history_story", "morality_case", "science_inquiry",
)


def _load(key: str) -> dict | None:
    p = TEACHERS_DIR / key / "teacher.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _save(key: str, data: dict) -> None:
    p = TEACHERS_DIR / key / "teacher.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def apply_one(variant: str, base: str | None, dry: bool) -> str:
    data = _load(variant)
    if data is None:
        return f"{variant}: 缺 teacher.json，跳过"

    if base:
        base_data = _load(base)
        base_ref = TEACHERS_DIR / base / "voice_ref.wav"
        if base_data is None or not base_ref.is_file():
            return f"{variant}: 主老师 {base} 无可用干声，跳过"
        target = TEACHERS_DIR / variant / "voice_ref.wav"
        if not dry:
            shutil.copyfile(base_ref, target)
        data["voice_ref_audio"] = f"data/teachers/{variant}/voice_ref.wav"
        data["voice_ref_text"] = str(base_data.get("voice_ref_text") or "")
        data["voice_speech_ratio"] = base_data.get("voice_speech_ratio")
        note = f"复用 {base} 干声"
    else:
        data["voice_ref_audio"] = ""
        data["voice_ref_text"] = ""
        note = "留空干声 -> 全局 TTS_REF_AUDIO"

    data["voice_engine"] = "clone"
    data["voice_clone_source"] = base or "global:TTS_REF_AUDIO"
    if not dry:
        _save(variant, data)
    return f"{variant}: -> clone（{note}）"


def revert_one(variant: str, dry: bool) -> str:
    data = _load(variant)
    if data is None:
        return f"{variant}: 缺 teacher.json"
    data["voice_engine"] = "edge"
    data["voice_ref_audio"] = ""
    data["voice_ref_text"] = ""
    data.pop("voice_clone_source", None)
    if not dry:
        _save(variant, data)
    return f"{variant}: -> edge"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="只预览不落盘")
    ap.add_argument("--revert", action="store_true", help="回退成 edge")
    args = ap.parse_args()

    mode = "REVERT" if args.revert else ("DRY" if args.dry else "APPLY")
    print(f"[variant-voice] mode={mode}")

    for variant, base in VARIANT_TO_BASE.items():
        print("  " + (revert_one(variant, args.dry) if args.revert
                      else apply_one(variant, base, args.dry)))
    for variant in FALLBACK_CLONE:
        print("  " + (revert_one(variant, args.dry) if args.revert
                      else apply_one(variant, None, args.dry)))

    if not args.dry:
        print("\n[done] 已写入 teacher.json；同步后需重启主应用生效：")
        print("       python scripts/sync_cloud.py && python scripts/cloud_ctl.py restart_app")
    return 0


if __name__ == "__main__":
    sys.exit(main())
