"""答疑产物收尾：回复校验 + 独立验算（全项目唯一实现）。

**为什么单独成一个模块**：这段逻辑原先在项目里有**三份**实现——
``Supervisor._finalize_streamed_response``、内置 ``TaskGraph`` 的 validate/solve 节点、
LangGraph 子图的 validate/degrade/solve 节点。三份同时存在时「改一处漏两处」是迟早的事；
更糟的是**流式链路会把它们跑两遍**（答疑管线一遍、收尾又一遍），
于是 sympy 的更正说明被重复追加到话术里。

现在收敛成这一个函数，统一由 ``TeachingAgent.solve`` 节点调用，
调用方保证**每回合恰好一次**。
"""
from __future__ import annotations

from typing import Any

from config.logging import get_logger
from core import math_solver
from core.hint_homework import HintLevel
from core.state.models import AIResponse
from core.validator import ResponseValidator, ValidationResult, drop_invalid_board_ops

log = get_logger("core.postprocess")


def postprocess_response(
    response: AIResponse,
    *,
    hint_level: HintLevel | None = None,
    board: Any | None = None,
    validator: ResponseValidator | None = None,
) -> ValidationResult:
    """对一段 AI 产物做播报前的收尾处理（**就地修改** ``response``）。

    两步，顺序不能换：

    1. **回复校验** —— 格式 / 语言 / 作业模式答案泄露 / 白板坐标。
       校验不通过时**只丢弃非法的白板指令**，不整块清空：历史 bug 是一处坐标越界
       就把白板清成空白，学生看到「讲了半天、板上一片白」。
    2. **独立验算** —— 用 sympy 复核模型给出的答案，错了就往 steps 与话术里追加更正。
       必须排在校验之后：验算会往产物里追内容，那些内容也该被校验口径覆盖。

    任何一步抛错都不能中断播报——「校验挂了」绝不能变成「老师不说话了」。
    """
    result = ValidationResult(ok=True)
    try:
        result = (validator or ResponseValidator()).validate(response, hint_level=hint_level)
        if not result.ok:
            log.warning("response validation issues: {}", result.reasons)
            dropped = drop_invalid_board_ops(response)
            if dropped:
                log.warning(
                    "dropped {} invalid board op(s), kept {}", dropped, len(response.board_ops)
                )
    except Exception as err:  # noqa: BLE001 - 校验异常不应中断播报
        log.warning("response validation failed: {}", err)
    try:
        math_solver.apply_verification(response, board, require_answer=True)
    except Exception as err:  # noqa: BLE001 - 验算异常同样不应中断播报
        log.warning("response verification failed: {}", err)
    return result


__all__ = ["postprocess_response"]
