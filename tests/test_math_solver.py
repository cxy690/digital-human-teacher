"""数学求解/校验单测（sympy 为可选依赖：缺失时整体跳过）。"""
from __future__ import annotations

import pytest

from core import math_solver
from core.state.models import AIResponse

pytestmark = pytest.mark.skipif(not math_solver.available(), reason="sympy not installed")


def test_solve_single_unknown() -> None:
    assert math_solver.solve_equation("3x+5=20") == {"x": "5"}


def test_solve_implicit_multiplication() -> None:
    assert math_solver.solve_equation("2b=10").get("b") == "5"


def test_solve_multi_unknown_returns_empty() -> None:
    assert math_solver.solve_equation("a+2b=8c") == {}


def test_verify_answer_correct_and_wrong() -> None:
    assert math_solver.verify_answer("3x+5=20", "x=5") is True
    assert math_solver.verify_answer("3x+5=20", "x=4") is False


def test_check_pure_arithmetic() -> None:
    assert math_solver.check_equation("49+51=100") is True
    assert math_solver.check_equation("47+51=100") is False


def test_apply_verification_corrects_wrong_answer() -> None:
    response = AIResponse(
        speech="同学你好。",
        meta={"equation": "3x+5=20", "answer": "x=4", "steps": ["解方程"]},
    )
    out = math_solver.apply_verification(response)
    assert out.meta.get("verified") is False
    assert out.meta.get("corrected_answer") == "x=5"
    assert "更正" in out.speech


def test_apply_verification_multi_unknown_hints() -> None:
    response = AIResponse(
        speech="同学你好。",
        meta={"equation": "4A+5B=100", "answer": "A=9,B=1", "steps": ["看个位"]},
    )
    out = math_solver.apply_verification(response)
    assert out.meta.get("verified") is None
    assert "解不唯一" in out.speech


def test_apply_verification_require_answer_skips_generic_formula() -> None:
    """概念讲解时模型常返回通用公式（如 ax+b=0）且不给具体答案。

    此时若提示「未知数多于方程数，解不唯一」，会把无关内容塞进讲解话术，
    故 require_answer=True 下必须跳过。
    """
    response = AIResponse(
        speech="一元一次方程的标准形式是 ax+b=0（a≠0）。",
        meta={"equation": "ax+b=0", "answer": "", "steps": ["认识方程"]},
    )
    out = math_solver.apply_verification(response, require_answer=True)
    assert out.speech == "一元一次方程的标准形式是 ax+b=0（a≠0）。"   # 未被污染
    assert "解不唯一" not in out.speech
    assert out.meta.get("verified") is None


def test_apply_verification_skips_formula_even_when_answer_present() -> None:
    """概念讲解时模型会返回 equation=ax+b=0 且 answer=x=-b/a（通用公式）。

    这是「公式」不是「条件不足的题目」，仍不得追加「解不唯一」，否则话术被污染。
    """
    response = AIResponse(
        speech="同学你好，一元一次方程的标准形式是 ax+b=0。",
        meta={"equation": "ax+b=0", "answer": "x=-b/a", "steps": ["认识方程"]},
    )
    out = math_solver.apply_verification(response, require_answer=True)
    assert "解不唯一" not in out.speech
    assert out.speech.startswith("同学你好")


def test_apply_verification_require_answer_keeps_hint_for_real_problem() -> None:
    """真题目（模型给了答案）仍应保留「解不唯一」提示。"""
    response = AIResponse(
        speech="同学你好。",
        meta={"equation": "4A+5B=100", "answer": "A=9,B=1", "steps": ["看个位"]},
    )
    out = math_solver.apply_verification(response, require_answer=True)
    assert "解不唯一" in out.speech


def test_apply_verification_ok_no_change() -> None:
    response = AIResponse(
        speech="同学你好。",
        meta={"equation": "3x+5=20", "answer": "x=5", "steps": ["解方程"]},
    )
    out = math_solver.apply_verification(response)
    assert out.meta.get("verified") is True
    assert out.speech == "同学你好。"
