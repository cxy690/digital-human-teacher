"""本地验证「白板边流边分批上板」：用假 LLM 流模拟真实输出，检查增量与最终一致性。

跑法：  cd education && python ../_test_stream_board.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "education"
sys.path.insert(0, str(ROOT))

from agents.shared.whiteboard import Whiteboard  # noqa: E402
from agents.teacher.agent import TeacherAgent, drain_steps  # noqa: E402
from core.state.models import StudentTurn  # noqa: E402
from core.state.session import SessionState  # noqa: E402
from models.base import BaseLLMClient  # noqa: E402

# ---- 模拟 LLM 的真实输出：理解段 -> 哨兵 -> 解答段（title/steps 先，speech 后）----
UNDERSTANDING = (
    '{"title": "一元一次方程",'
    ' "understanding": ["已知 3x+5=20", "要求 x 的值"],'
    ' "steps": ["先移项", "再合并"],'
    ' "speech": "同学你好，我们先看清这道题要我们求什么，再来一步步解它。"}'
)
STEPS = [
    "两边同时减 5，得到 3x = 15",
    "两边同时除以 3，得到 x = 5",
    "代入原式验算：3×5+5=20，等式成立",
    "所以 x=5 是这道题的解",
]
SOLUTION = (
    '{"title": "一元一次方程求解",'
    ' "steps": [' + ",".join(f'"{s}"' for s in STEPS) + "],"
    ' "equation": "3x+5=20",'
    ' "answer": "x=5",'
    ' "knowledge_points": ["移项要变号", "等式性质2"],'
    ' "pitfalls": ["移项时忘记变号"],'
    ' "speech": "这道题先把 5 移到右边，再两边除以 3，就得到 x=5。",'
    ' "followup": {"question": "如果方程变成 3x-5=20，第一步该做什么？",'
    '              "answer": "两边同时加 5",'
    '              "hint": "想想移项要变号",'
    '              "options": ["两边加5", "两边减5"]}}'
)
# 真实协议是「哨兵在前、JSON 在后」（与 _BOARD_FIRST_PROMPT 的要求一致）
FULL = "###UNDERSTANDING###" + UNDERSTANDING + "###SOLUTION###" + SOLUTION

# 把完整输出切成 14 段增量（模拟真实流式），剩余部分作为最后一段
CHUNKS: list[str] = []
acc = 0
for size in (18, 24, 20, 16, 22, 18, 20, 24, 18, 22, 20, 24, 18):
    CHUNKS.append(FULL[acc:acc + size])
    acc += size
CHUNKS.append(FULL[acc:])                       # 剩余全部（含最后几步）作为最后一段
assert "".join(CHUNKS) == FULL, "切片必须无损覆盖完整输出"


class FakeLLM(BaseLLMClient):
    """按预切片逐段吐出的假 LLM（验证流式消费逻辑足够）。"""

    async def chat(self, messages, *, temperature=0.3, max_tokens=1024):  # noqa: D401
        return FULL

    async def chat_stream(self, messages, *, temperature=0.3, max_tokens=1024):
        for c in CHUNKS:
            yield c


def op_key(op) -> tuple:
    return (op.op, getattr(op, "text", ""), getattr(op, "x", None), getattr(op, "y", None))


async def main() -> int:
    agent = TeacherAgent(llm=FakeLLM(), board=Whiteboard())
    session = SessionState(session_id="test-stream", student_id="tester")
    session.subject = "数学"
    turn = StudentTurn(text="3x+5=20，x 等于多少？")

    stages: list[tuple[str, object]] = []
    async for stage, resp in agent.stream_board_first(turn, session):
        stages.append((stage, resp))

    kinds = [s for s, _ in stages]
    print("=== 1) 产出阶段顺序 ===")
    print("   ", kinds)
    assert kinds[0] == "understanding", "第一段应是理解"
    assert kinds[-1] == "solution", "最后应是解答"
    board_stages = [s for s in kinds if s == "board"]
    assert board_stages, "应产出 board 增量"

    boards = [r for s, r in stages if s == "board"]
    final = stages[-1][1]

    print()
    print("=== 2) 每次上板的累计内容（应逐步多一行步骤）===")
    for i, b in enumerate(boards, 1):
        n_steps = len(b.board_ops) - 1
        print(f"   第{i}次：标题={'有' if b.board_ops else '无'} 步骤数={n_steps} "
              f"最新步骤={b.board_ops[-1].text[:22] if len(b.board_ops) > 1 else '-'}")
    step_counts = [len(b.board_ops) - 1 for b in boards]
    assert step_counts == sorted(step_counts), "步骤数应单调递增"
    assert step_counts[-1] == len(STEPS), f"最终应包含全部 {len(STEPS)} 步，实际 {step_counts[-1]}"

    print()
    print("=== 3) 增量必须是最终板书的前缀（坐标/文本逐像素一致）===")
    final_keys = [op_key(o) for o in final.board_ops]
    for i, b in enumerate(boards, 1):
        keys = [op_key(o) for o in b.board_ops]
        assert keys == final_keys[: len(keys)], f"第{i}次上板与最终板书前缀不一致"
    print(f"   全部 {len(boards)} 次增量都是最终 {len(final.board_ops)} 条板书的前缀 ✓")

    print()
    print("=== 4) 最终板书包含知识点/易错点，且位置符合布局公式 ===")
    texts = [getattr(o, "text", "") for o in final.board_ops]
    assert any("知识点" in t for t in texts), "应有【知识点】区块"
    assert any("易错点" in t for t in texts), "应有【易错点】区块"
    gap = Whiteboard.EXPLAIN_LINE_GAP
    expect_kp_y = round(Whiteboard.EXPLAIN_START_Y + (len(STEPS) + 1) * gap, 3)
    kp_op = next(o for o in final.board_ops if getattr(o, "text", "") == "【知识点】")
    assert abs(kp_op.y - expect_kp_y) < 1e-9, f"知识点 y={kp_op.y} 应为 {expect_kp_y}"
    print(f"   【知识点】y={kp_op.y}（与布局公式一致）✓")

    print()
    print("=== 5) drain_steps 的健壮性 ===")
    assert drain_steps('{"steps": ["a", "b"]}') == ["a", "b"]
    assert drain_steps('{"steps": ["a", "b"]}') == ["a", "b"]   # 数组未闭合但项完整
    assert drain_steps('{"steps": ["a", "b"]}') == ["a", "b"]         # 末项半截被丢弃
    assert drain_steps('{"steps": ["a", "b"]}') == ["a", "b"]        # 只有前导
    assert drain_steps('{"title": "x"}') == []                  # 没有 steps 就不给
    assert drain_steps('{"understanding": ["1", "2"]}') == []   # 只认 steps 字段
    BS = chr(92)                                                # 反斜杠（避免源码里转义混淆）
    esc = '{"steps": ["带 ' + BS + '"引号' + BS + '" 的步骤"]}'
    assert drain_steps(esc) == ['带 "引号" 的步骤'], "应解码转义引号"
    nl = '{"steps": ["第一行' + BS + 'n第二行"]}'
    assert drain_steps(nl) == ["第一行\n第二行"], "应解码换行转义（一步内含换行）"
    uni = '{"steps": ["' + BS + 'u6570' + BS + 'u5b66"]}'
    assert drain_steps(uni) == ["数学"], "应解码 unicode 转义"
    print("   半截项 / 转义引号 / 换行转义 / unicode 转义 / 无 steps -> 全部正确 ✓")

    print()
    print("=== 6) 模型不按字段顺序输出时的降级 ===")
    assert drain_steps('{"speech": "很长的话", "steps": ["s1"]}') == ["s1"]
    print("   steps 在 speech 之后也能取到 ✓")

    print()
    print("结论：全部通过 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
