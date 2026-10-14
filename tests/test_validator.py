"""回复校验单测。"""
from __future__ import annotations

from config.constants import BoardOp
from core.hint_homework import HintLevel
from core.state.models import AIResponse, WhiteboardOp
from core.validator import ResponseValidator


def test_empty_response_invalid() -> None:
    assert not ResponseValidator().validate(AIResponse()).ok


def test_answer_leak_blocked_in_hint_mode() -> None:
    response = AIResponse(speech="这道题的答案：4")
    result = ResponseValidator().validate(response, hint_level=HintLevel.METHOD)
    assert not result.ok
    assert "answer_leaked" in result.reasons


def test_answer_allowed_at_full_level() -> None:
    response = AIResponse(speech="这道题的答案：4")
    result = ResponseValidator().validate(response, hint_level=HintLevel.ANSWER)
    assert result.ok


def test_board_coordinate_out_of_range() -> None:
    response = AIResponse(
        speech="我们在白板上画一下辅助线。",
        board_ops=[WhiteboardOp(op=BoardOp.DRAW, points=[[1.5, 0.2]])],
    )
    result = ResponseValidator().validate(response)
    assert not result.ok
    assert "board_point_out_of_range" in result.reasons
