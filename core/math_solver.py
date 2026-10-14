"""数学求解与校验：用 sympy 独立求解/验算，弥补 LLM 心算不可靠的问题。

依赖可选：未安装 sympy 时 ``available()`` 为 False，所有校验自动降级为「不做处理」。
"""
from __future__ import annotations

import re
from typing import Any

from config.logging import get_logger

log = get_logger("core.math_solver")

try:
    import sympy
    from sympy.parsing.sympy_parser import (
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    _HAS_SYMPY = True
except ImportError:  # pragma: no cover - 取决于环境
    _HAS_SYMPY = False

_TRANSFORMS = (
    standard_transformations + (implicit_multiplication_application,)
) if _HAS_SYMPY else None

# 「像一道具体题目」的判据：方程里含非零数字（通用公式如 ax+b=0 只有 0，视为公式）
_HAS_NONZERO_DIGIT = re.compile(r"[1-9]")

_REPLACES = {
    "×": "*", "÷": "/", "−": "-", "–": "-", "—": "-",
    "＝": "=", "＾": "^", "（": "(", "）": ")", "，": ",",
}


def available() -> bool:
    """sympy 是否可用。"""
    return _HAS_SYMPY


#: ``warn_if_unavailable`` 是否已经喊过一次（避免每次调用都刷日志）。
_WARNED = False


def warn_if_unavailable() -> bool:
    """启动时调用一次：sympy 缺失就打一条**醒目告警**。返回是否可用。

    为什么需要这个函数：sympy 缺失时**所有校验静默降级为「不做处理」**——
    不报错、不崩、不留痕迹，唯一的后果是 **AI 老师讲错答案时不再被自动纠正**
    （该解 ``x=5`` 说成 ``x=9``，线上会安静地把错答案讲完）。

    这个坑真的踩过：``requirements.txt`` 与 ``pyproject.toml`` 双双漏了 sympy，
    本地碰巧装了所以测试全绿，云端却长期静默失效而无人察觉。因此**降级必须喊一声**——
    宁可启动时刺一条 WARNING，也不能让关键能力无声消失。
    """
    global _WARNED
    if _HAS_SYMPY:
        return True
    if not _WARNED:
        _WARNED = True
        log.warning(
            "sympy 未安装：方程求解与独立验算已降级为「不做处理」——"
            "AI 老师讲错答案时不会被自动纠正（线上会静默讲错）。"
            "请执行：pip install 'sympy>=1.12'"
        )
    return False


def _clean(text: str) -> str:
    s = (text or "").strip()
    for a, b in _REPLACES.items():
        s = s.replace(a, b)
    return s


def _parse(expr: str) -> Any:
    return parse_expr(expr.replace("^", "**"), transformations=_TRANSFORMS, evaluate=True)


def check_equation(equation: str) -> bool | None:
    """校验纯算术等式是否成立（无未知数）。无法判定返回 None。"""
    if not _HAS_SYMPY:
        return None
    eq = _clean(equation)
    if "=" not in eq:
        return None
    try:
        lhs, rhs = eq.split("=", 1)
        diff = sympy.simplify(_parse(lhs) - _parse(rhs))
    except Exception:  # noqa: BLE001
        return None
    if not diff.free_symbols:
        return bool(diff == 0)
    return None


def solve_equation(equation: str) -> dict[str, str]:
    """求解含未知数的等式，返回 {符号: 解}。支持 3x+5=20、a+2b=8c。"""
    if not _HAS_SYMPY:
        return {}
    eq = _clean(equation)
    if not eq:
        return {}
    try:
        if "=" in eq:
            lhs, rhs = eq.split("=", 1)
            resid = _parse(lhs) - _parse(rhs)
        else:
            resid = _parse(eq)
    except Exception as err:  # noqa: BLE001
        log.warning("parse equation failed: {} ({})", equation, err)
        return {}
    symbols = sorted(resid.free_symbols, key=lambda s: s.name)
    if len(symbols) != 1:
        return {}  # 仅对单未知数给出确定解
    try:
        sols = sympy.solve(resid, symbols, dict=True)
    except Exception as err:  # noqa: BLE001
        log.warning("solve failed: {} ({})", equation, err)
        return {}
    if not sols:
        return {}
    sol = sols[0]
    s = symbols[0]
    if s not in sol:
        return {}
    return {str(s): str(sympy.simplify(sol[s]))}


def _parse_assignments(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in re.split(r"[,，;；]", _clean(text)):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()
        if re.fullmatch(r"[A-Za-z]\w*", k) and v:
            out[k] = v
    return out


def verify_answer(equation: str, answer: str) -> bool | None:
    """校验模型答案是否满足方程。True/False；无法判定返回 None。"""
    if not _HAS_SYMPY:
        return None
    correct = solve_equation(equation)
    if not correct:
        return None
    given = _parse_assignments(answer)
    if not given:
        return None
    for k, v in given.items():
        if k not in correct:
            return None
        try:
            if sympy.simplify(_parse(correct[k]) - _parse(v)) != 0:
                return False
        except Exception:  # noqa: BLE001
            return None
    return True


def _append_note(response: Any, meta: dict[str, Any], board: Any | None, note: str) -> None:
    """把更正/提示追加到白板步骤、白板与话术。"""
    steps = list(meta.get("steps") or [])
    steps.append(note)
    meta["steps"] = steps
    if board is not None:
        try:
            y = 0.12 + 0.08 * (len(steps) + 1)
            response.board_ops.append(
                board.write_text(note, 0.08, round(min(y, 0.92), 3), color="#d4380d")
            )
        except Exception:  # noqa: BLE001 - 白板不可用不影响文本
            pass
    if response.speech:
        response.speech = f"{note}{response.speech}"


def apply_verification(response: Any, board: Any | None = None, *, require_answer: bool = False) -> Any:
    """对 AI 产物做「独立求解 + 验算」。

    依据 ``response.meta['equation']` 与 ``response.meta['answer']``（教师模型给出）：
    * 单未知数等式：sympy 求解并比对答案，错误则追加更正；
    * 多未知数等式：判定为「解不唯一」，追加提示（避免硬给一个错答案）；
    * 纯算术等式：直接验算是否成立。

    ``require_answer=True`` 时，模型未给出具体答案（``meta['answer']`` 为空）就不做
    「多未知数 -> 解不唯一」提示：这种情况通常是**概念/通用公式**（如 ``ax+b=0``），
    对它提示「解不唯一」会污染讲解话术。
    """
    if not _HAS_SYMPY:
        return response
    meta = response.meta if response.meta is not None else {}
    equation = _clean(str(meta.get("equation") or ""))
    answer = str(meta.get("answer") or "").strip()
    if not equation or "=" not in equation:
        return response

    lhs, rhs = equation.split("=", 1)
    try:
        free = (_parse(lhs) - _parse(rhs)).free_symbols
    except Exception:  # noqa: BLE001
        return response
    meta["verified"] = None

    if len(free) > 1:
        # 「解不唯一」只在**像一道具体题目**时提示：方程里应含非零数字（如 4A+5B=100）。
        # 通用公式（ax+b=0、x+y=z 这类系数用字母表示）没有非零数字，对它提示「条件不足」
        # 属于误判——概念讲解时模型经常这样填 equation，提示会污染讲解话术。
        if not _HAS_NONZERO_DIGIT.search(equation):
            return response
        if require_answer and not answer:
            return response
        note = f"提示：{equation} 中未知数多于方程数，解不唯一，需再补充条件才能唯一确定。"
        _append_note(response, meta, board, note)
        return response

    if len(free) == 0:
        ok = check_equation(equation)
        meta["verified"] = ok
        if ok is False:
            _append_note(response, meta, board, f"更正：等式「{equation}」不成立，请重新核算。")
        return response

    # 单未知数：求解并与模型答案比对（模型未给答案则不判定）
    if not answer:
        return response
    status = verify_answer(equation, answer)
    meta["verified"] = status
    if status is False:
        correct = solve_equation(equation)
        if correct:
            pretty = "、".join(f"{k}={v}" for k, v in correct.items())
            meta["corrected_answer"] = pretty
            _append_note(response, meta, board, f"更正：{equation} 的正确解是 {pretty}。")
    return response
