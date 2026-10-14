"""离线演示：不启动服务即可跑通「答疑 + 麦控」全链路。

运行：
    python -m scripts.run_demo
无需任何外部密钥（LLM/ASR/TTS 均自动降级为 Mock）。
"""
from __future__ import annotations

import asyncio
from typing import Any

from config.logging import setup_logging
from core.harness import EducationHarness
from core.state.models import StudentTurn


def _printer(payload: dict[str, Any]) -> None:
    """把服务端下行事件打印到控制台，直观看到链路输出。"""
    t = payload.get("type")
    data = payload.get("payload")
    if t == "ai_speech":
        print(f"  [AI 语音] {data['text']}")
    elif t == "board_ops":
        print(f"  [白板指令] {len(data)} 条（如：{data[0]['op']}）")
    elif t == "status":
        print(f"  [状态] {data}")
    elif t == "chat_message":
        print(f"  [{data['role']}] {data['text'][:40]}")


async def demo_qa() -> None:
    print("\n=== 场景一：学生文字提问 -> AI 分步讲解 + 白板输出 ===")
    harness = EducationHarness()
    state, supervisor, _ = harness.create_session("demo-student", emit=_printer)
    await supervisor.on_start()
    await supervisor.handle_turn(StudentTurn(text="老师，这道方程题我不会，能帮我讲讲吗？"))


async def demo_mic() -> None:
    print("\n=== 场景二：智能麦控（学生说话 AI 静默 -> 静默后 AI 答疑）===")
    harness = EducationHarness()
    state, supervisor, _ = harness.create_session("demo-mic", emit=_printer)
    await supervisor.on_start()

    base = 1_000_000
    for i in range(4):  # 学生持续说话
        supervisor.on_vad(True, base + i * 100)
    supervisor.on_vad(False, base + 1000)  # 静默 -> 触发答疑
    await supervisor.on_voice(b"demo-audio-bytes")


async def main() -> None:
    setup_logging()
    await demo_qa()
    await demo_mic()
    print("\n演示结束。启动实时服务： python -m server.app  然后访问 http://localhost:8080/app/")


if __name__ == "__main__":
    asyncio.run(main())
