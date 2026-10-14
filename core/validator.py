"""回复校验：在 AI 产物播报/渲染前做安全与规范校验，保证输出质量与合规。

覆盖：话术长度与语言、作业模式下的答案泄露、白板指令坐标合法性、敏感内容。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from config.constants import BoardOp
from core.hint_homework import HintLevel
from core.state.models import AIResponse, WhiteboardOp

# 极简敏感词（真实环境应接入内容安全服务）
_SENSITIVE_PATTERNS = [r"去死", r"笨蛋", r"蠢", r"滚"]

_MAX_SPEECH_CHARS = 800
_ANSWER_PATTERN = re.compile(r"(答案|结果为|等于)\s*[:：]?\s*[A-D0-9．.]+")


@dataclass
class ValidationResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)

    def add(self, reason: str) -> None:
        self.ok = False
        self.reasons.append(reason)


def _board_op_ok(op: WhiteboardOp) -> bool:
    """单条白板指令坐标是否合法（与 :meth:`ResponseValidator.validate` 的判据一致）。"""
    if op.op in (BoardOp.DRAW, BoardOp.ANNOTATE, BoardOp.SHAPE):
        return all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in op.points)
    if op.op == BoardOp.TEXT:
        if op.x is not None and not (0.0 <= op.x <= 1.0):
            return False
        if op.y is not None and not (0.0 <= op.y <= 1.0):
            return False
    return True


def drop_invalid_board_ops(response: AIResponse) -> int:
    """只丢弃坐标非法的白板指令，保留其余；返回丢弃条数。

    历史 bug：校验不通过时直接 ``response.board_ops = []``，一处坐标越界就把整块
    白板清空 —— 表现为「AI 照常讲解、数字人照常播报，但白板一片空白」。
    改为定点丢弃后，最多少一两行，不会出现整块空白。
    """
    if not response.board_ops:
        return 0
    kept = [op for op in response.board_ops if _board_op_ok(op)]
    dropped = len(response.board_ops) - len(kept)
    response.board_ops = kept
    return dropped


class ResponseValidator:
    """AI 回复校验器。"""

    def validate(
        self,
        response: AIResponse,
        *,
        hint_level: HintLevel | None = None,
    ) -> ValidationResult:
        result = ValidationResult(ok=True)

        speech = (response.speech or "").strip()
        if not speech and not response.board_ops:
            result.add("empty_response")
        if len(speech) > _MAX_SPEECH_CHARS:
            result.add("speech_too_long")
        if speech and not re.search(r"[\u4e00-\u9fff]", speech):
            result.add("speech_not_chinese")
        for pat in _SENSITIVE_PATTERNS:
            if re.search(pat, speech):
                result.add(f"sensitive:{pat}")

        # 作业低层级提示时，禁止直接泄露答案
        if hint_level in (HintLevel.CONCEPT, HintLevel.METHOD, HintLevel.STEP):
            if _ANSWER_PATTERN.search(speech):
                result.add("answer_leaked")

        # 白板坐标合法性
        for op in response.board_ops:
            if op.op in (BoardOp.DRAW, BoardOp.ANNOTATE, BoardOp.SHAPE):
                for x, y in op.points:
                    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                        result.add("board_point_out_of_range")
                        break
            if op.op == BoardOp.TEXT:
                if op.x is not None and not (0.0 <= op.x <= 1.0):
                    result.add("board_text_x_out_of_range")
                if op.y is not None and not (0.0 <= op.y <= 1.0):
                    result.add("board_text_y_out_of_range")
        return result
