"""统一教师音色策略：**CosyVoice3 克隆优先，edge 标准普通话兜底**。

背景
----
* 云端 CosyVoice3 推理服务（``scripts/tts_engine_cosyvoice.py``，:8090）已就绪，
  zero-shot 克隆比 v2 相似度更高、更稳定 —— 有干声的老师优先用它还原「本人嗓音」。
* 但克隆是**零样本**的：没有干声素材的老师无从克隆，必须退回 edge 的
  ``zh-CN-*`` 标准普通话音色（微软端点实测可用的 6 个，均无方言口音）。
* 历史教训：曾给变体老师配过**方言口音**音色（东北 / 陕西），已全部下线；
  音色只允许标准 ``zh-CN`` 普通话。

处理（幂等，可反复执行）
------------------------
* ``voice_ref_audio`` 指向的干声文件**真实存在** → ``voice_engine = "clone"``；
* 没有干声 / 干声文件缺失 → ``voice_engine = "edge"``，音色用标准普通话；
* ``voice`` 字段一律收敛到标准普通话音色（方言音色按形象性别替换）；
* 已合规的档案不动，避免无谓改写。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEACHERS = ROOT / "data" / "teachers"

# 标准 zh-CN 普通话音色（微软 Edge 端点实测可用，均无方言口音）
MALE = {
    "zh-CN-YunyangNeural",      # 云扬 沉稳男声
    "zh-CN-YunjianNeural",      # 云健 浑厚男声
    "zh-CN-YunxiNeural",        # 云希 清朗男声
    "zh-CN-YunxiaNeural",       # 云夏 少年男声
}
FEMALE = {
    "zh-CN-XiaoxiaoNeural",     # 晓晓 标准女声
    "zh-CN-XiaoyiNeural",       # 晓伊 标准女声
}
FEMALE_DEFAULT = "zh-CN-XiaoxiaoNeural"
MALE_DEFAULT = "zh-CN-YunxiNeural"


def _ref_ok(raw: str) -> bool:
    """干声路径非空且文件真实存在（相对路径按项目根解析）。"""
    if not raw:
        return False
    p = Path(raw)
    return (p if p.is_absolute() else (ROOT / p)).is_file()


def main() -> None:
    changed = 0
    for path in sorted(TEACHERS.glob("*/teacher.json")):
        key = path.parent.name
        data = json.loads(path.read_text(encoding="utf-8"))
        gender = str(data.get("gender") or "").strip().lower()
        allowed = MALE if gender == "male" else FEMALE

        voice = str(data.get("voice") or "").strip()
        new_voice = voice
        if not voice or voice not in allowed:
            # 方言音色 / 空音色 / 与形象性别不符：按形象性别给标准普通话音色
            new_voice = MALE_DEFAULT if gender == "male" else FEMALE_DEFAULT

        # 发声方式：有干声 -> CosyVoice3 克隆；没有 -> edge 标准普通话兜底
        engine = "clone" if _ref_ok(str(data.get("voice_ref_audio") or "")) else "edge"

        if new_voice == voice and engine == str(data.get("voice_engine") or ""):
            continue                                 # 已合规，不写盘
        old = f"{voice or '(空)'}/{data.get('voice_engine')}"
        data["voice"] = new_voice
        data["voice_engine"] = engine
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        changed += 1
        print(f"{key:20s} {old:34s} -> {new_voice} / {engine}")

    print(f"\n共改写 {changed} 份档案（其余已合规，未动）")


if __name__ == "__main__":
    if os.name == "nt":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
