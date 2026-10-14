"""把 20 位老师的 edge 音色校准到「算力云上真实存在」的中文音色。

背景
----
微软 Edge 朗读端点只暴露 8 个 zh-CN 音色（4 男 4 女，含 2 个方言口音）。
之前凭印象配的 ``zh-CN-XiaozhenNeural`` / ``YunfengNeural`` / ``XiaomengNeural``
等**并不存在于该端点**，调用必然 ``NoAudioReceived``；``TeacherTTS`` 随即回退到
全局 ``TTS_REF_AUDIO``（那是一份女声），于是男老师也发出女声。

分配原则
--------
1. 只用云端实测可用的音色（`scripts/diag_tts_probe.py` 全量验证过）；
2. 老师音色的性别必须与**形象性别**一致；
3. 同科两位老师（如 math / math_strict）尽量给不同音色，方便区分讲课风格；
4. **一律用普通话标准音色**：``zh-CN-liaoning-XiaobeiNeural``（东北）与
   ``zh-CN-shaanxi-XiaoniNeural``（陕西）两个方言音色**不再指派给任何老师**——
   讲课要让学生听懂，方言口音会影响听课（历史做法是把方言音色给变体老师，
   现在改成在「晓晓 / 晓伊」两个标准女声之间错开，既无方言也能区分风格）。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data" / "teachers"

# 本脚本可指派的音色（云端实测可用；方言音色已剔除，见模块 docstring 第 4 条）
MALE_VOICES = {
    "zh-CN-YunyangNeural",      # 云扬 沉稳男声
    "zh-CN-YunjianNeural",      # 云健 浑厚男声
    "zh-CN-YunxiNeural",        # 云希 清朗男声
    "zh-CN-YunxiaNeural",       # 云夏 少年男声
}
FEMALE_VOICES = {
    "zh-CN-XiaoxiaoNeural",               # 晓晓 标准女声
    "zh-CN-XiaoyiNeural",                 # 晓伊 标准女声
    # 方言音色（zh-CN-liaoning-XiaobeiNeural / zh-CN-shaanxi-XiaoniNeural）
    # 刻意不在白名单里：写字上去会被 main() 判为“与性别不匹配”而拦下。
}

VOICE: dict[str, str] = {
    # ---------------- 男老师（4 个男声轮转）
    "math": "zh-CN-YunyangNeural",
    "morality": "zh-CN-YunxiaNeural",
    "morality_case": "zh-CN-YunjianNeural",
    "history": "zh-CN-YunxiNeural",
    "history_story": "zh-CN-YunyangNeural",
    "politics": "zh-CN-YunjianNeural",
    # 物理 / 地理默认老师：素材形象是**男性**，一律配男声（物理原配晓伊女声、地理原配云夏
    # 少年音，听起来都像女生，2026-09-29 按形象改为男声）
    "physics": "zh-CN-YunyangNeural",
    "geography": "zh-CN-YunxiNeural",
    # ---------------- 女老师（只用 2 个普通话标准女声，同科错开）
    "chinese": "zh-CN-XiaoxiaoNeural",
    "chinese_logic": "zh-CN-XiaoyiNeural",
    "english": "zh-CN-XiaoxiaoNeural",
    "english_immersive": "zh-CN-XiaoyiNeural",
    "math_strict": "zh-CN-XiaoxiaoNeural",
    # 同科两位老师：默认老师与变体老师各占一个标准女声，便于区分风格（不再用方言口音）
    "physics_lab": "zh-CN-XiaoxiaoNeural",
    "chemistry": "zh-CN-XiaoxiaoNeural",
    "chemistry_micro": "zh-CN-XiaoyiNeural",
    "biology": "zh-CN-XiaoyiNeural",
    "biology_explore": "zh-CN-XiaoxiaoNeural",
    "geography_map": "zh-CN-XiaoyiNeural",
    "science_inquiry": "zh-CN-XiaoyiNeural",
}


def main() -> None:
    fixed = 0
    problems: list[str] = []
    for name, voice in sorted(VOICE.items()):
        path = ROOT / name / "teacher.json"
        if not path.is_file():
            problems.append(f"{name}: 档案不存在")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        gender = str(data.get("gender") or "")
        allowed = MALE_VOICES if gender == "male" else FEMALE_VOICES
        if voice not in allowed:
            problems.append(f"{name}: 音色 {voice} 与性别 {gender} 不匹配")
            continue
        old = data.get("voice")
        data["voice"] = voice
        data["voice_engine"] = "edge"
        # 干声为空的老师不要再留 clone 痕迹：否则哪天 voice_engine 被改回，
        # 又会落到全局 TTS_REF_AUDIO（女声）
        if not data.get("voice_ref_audio"):
            data.pop("voice_clone_source", None)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        fixed += 1
        flag = "" if old == voice else "   <- 修正"
        print(f"{name:20s} {gender:7s} {voice:32s}{flag}")

    print(f"\n共写入 {fixed} 份档案")
    if problems:
        print("异常：" + "; ".join(problems))


if __name__ == "__main__":
    main()
