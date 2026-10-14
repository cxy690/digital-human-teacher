"""方案A 本地效果演示：deepseek 回答 + edge-tts 声音 + Wav2Lip 对口型 → 成品 mp4。

链路（1对1 教学AI 方案A）：
    学生提问 → [LLM deepseek 云] 回答文本 → [TTS edge-tts CPU] 语音 → [数字人 Wav2Lip 本地:8011] 对口型视频
产出自带音轨的 mp4（声音+口型同步），可直接播放查看"会说话、对口型的 AI 老师"效果。

运行（需先启动 Wav2Lip 引擎 :8011）：
    python -m scripts.demo_plan_a
    python -m scripts.demo_plan_a --question "老师，什么是分数的基本性质？"
产出：
    clients/plan_a_demo.mp4   （访问 http://127.0.0.1:8001/app/plan_a_demo.mp4）
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from pathlib import Path

from config.logging import setup_logging
from models.avatar import get_avatar
from models.llm import get_llm
from models.tts import get_tts

ROOT = Path(__file__).resolve().parents[1]
OUT_MP4 = ROOT / "clients" / "plan_a_demo.mp4"

SYSTEM_PROMPT = (
    "你是一位亲切的中小学 AI 教师，正在一对一给学生答疑。"
    "请用口语化、简短的方式回答（控制在 2-3 句、60 字以内），便于语音播报与对口型渲染。"
)


async def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default="老师，什么是分数的基本性质？能举个例子吗？")
    args = ap.parse_args()

    llm, tts, avatar = get_llm(), get_tts(), get_avatar()
    print(f"[planA] LLM={type(llm).__name__} TTS={type(tts).__name__} Avatar={type(avatar).__name__}")

    # 1) LLM 出回答文本
    print(f"[planA] 提问：{args.question}")
    answer = await llm.chat(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": args.question}],
        temperature=0.4,
    )
    answer = (answer or "").strip()
    print(f"[planA] 回答：{answer}")

    # 2) TTS 出声音
    audio = await tts.synthesize(answer)
    if not audio:
        raise SystemExit("[planA] TTS 无音频（检查 TTS_PROVIDER=edge 是否可用）")
    print(f"[planA] 语音 {len(audio)//1024}KB 合成完成")

    # 3) 数字人渲染对口型视频（自带音轨）
    if not avatar.produces_video:
        raise SystemExit("[planA] 当前数字人不是视频引擎（检查 AVATAR_BACKEND/引擎是否启动 :8011）")
    result = await avatar.render(audio, text=answer)
    if result.get("mode") != "video" or not result.get("video"):
        raise SystemExit(f"[planA] 渲染失败：{result}")
    OUT_MP4.write_bytes(base64.b64decode(result["video"]))
    print(f"\n[planA] 成品视频（有声+对口型）：{OUT_MP4}")
    print(f"[planA] 浏览器播放：http://127.0.0.1:8001/app/plan_a_demo.mp4")


if __name__ == "__main__":
    asyncio.run(main())
