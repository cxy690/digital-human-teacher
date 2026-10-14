"""白板排版回归测试。

历史 bug：讲解排版的 y 坐标是 ``0.12 + index * 0.08``，步骤/知识点/易错点一多
（≥12 行）y 就会超过 1.0。core/validator.py 判 board_text_y_out_of_range 之后，
降级逻辑把 **整块** board_ops 清空 —— 表现为「AI 照常讲解、白板一片空白」。
"""
from __future__ import annotations

import pytest

from config.constants import BoardOp
from agents.shared.whiteboard import Whiteboard
from core.state.models import AIResponse, WhiteboardOp
from core.validator import ResponseValidator, drop_invalid_board_ops


def _build(steps: int, kp: int, pit: int) -> list:
    return Whiteboard().build_explain(
        title="题目讲解",
        steps=[f"第{i}步的说明文字" for i in range(1, steps + 1)],
        knowledge_points=[f"知识点{i}" for i in range(1, kp + 1)] or None,
        pitfalls=[f"易错点{i}" for i in range(1, pit + 1)] or None,
    )


@pytest.mark.parametrize(
    ("steps", "kp", "pit"),
    [
        (4, 2, 2),    # 常规
        (6, 3, 3),    # 典型（线上触发 bug 的规模）
        (12, 0, 0),   # 步骤刚好多
        (10, 4, 4),   # 步骤 + 两个区块
        (20, 5, 5),   # 极端多
    ],
)
def test_all_text_ops_within_bounds(steps: int, kp: int, pit: int) -> None:
    """任何规模的内容，所有文字指令的坐标都必须落在 [0, 1]。"""
    ops = _build(steps, kp, pit)
    assert ops, "白板至少要有标题"
    for op in ops:
        assert op.x is not None and 0.0 <= op.x <= 1.0, op
        assert op.y is not None and 0.0 <= op.y <= 1.0, op


def test_content_grows_never_yields_empty_board() -> None:
    """内容再多也不能产出空白板（历史上会被整块清空）。"""
    for steps in (4, 8, 12, 20, 30):
        assert len(_build(steps, 3, 3)) >= 1


def test_write_text_clamps_out_of_range_coords() -> None:
    """兜底：任何越界坐标都会被夹回合法区间，不再触发校验失败。"""
    board = Whiteboard()
    assert board.write_text("x", -0.5, 1.8).y == 1.0
    assert board.write_text("x", -0.5, 1.8).x == 0.0


def test_validator_passes_for_large_board() -> None:
    """大内容白板必须能通过校验（不得出现 board_text_y_out_of_range）。"""
    response = AIResponse(speech="我们来一步步看这道题。", board_ops=_build(10, 4, 4))
    result = ResponseValidator().validate(response)
    assert result.ok, result.reasons


def test_drop_invalid_board_ops_keeps_the_rest() -> None:
    """校验降级只丢有问题的那几条，不能把整块板清空。

    直接构造越界 op（绕过 write_text 的夹取），验证“定点丢弃”本身。
    """
    board = Whiteboard()
    response = AIResponse(
        speech="讲解",
        board_ops=[
            board.write_text("第一行", 0.08, 0.12),
            WhiteboardOp(op=BoardOp.TEXT, text="越界行", x=0.08, y=1.5),  # 非法
            board.write_text("第三行", 0.08, 0.28),
        ],
    )
    dropped = drop_invalid_board_ops(response)
    assert dropped == 1
    assert len(response.board_ops) == 2
    assert [op.text for op in response.board_ops] == ["第一行", "第三行"]


def test_validator_flags_manually_out_of_range_op() -> None:
    """手动构造的越界 op 仍会被 validator 判出来（守卫没有失效）。"""
    response = AIResponse(
        speech="讲解",
        board_ops=[WhiteboardOp(op=BoardOp.TEXT, text="越界", x=0.08, y=1.5)],
    )
    assert "board_text_y_out_of_range" in ResponseValidator().validate(response).reasons
