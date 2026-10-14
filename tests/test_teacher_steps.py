"""教师步骤前缀剥离单测：避免白板/播报重复编号。"""
from __future__ import annotations

from agents.teacher.agent import TeacherAgent


def test_strip_single_prefix() -> None:
    assert TeacherAgent._strip_step_prefix("第一步：明确变化名称") == "明确变化名称"


def test_strip_compound_prefix() -> None:
    # 模型可能同时带“1.”和“第一步：”，必须一次剥净
    assert TeacherAgent._strip_step_prefix("1. 第一步：明确变化名称") == "明确变化名称"


def test_strip_paren_and_dot_prefix() -> None:
    assert TeacherAgent._strip_step_prefix("（1）定义") == "定义"
    assert TeacherAgent._strip_step_prefix("(2) abc") == "abc"
    assert TeacherAgent._strip_step_prefix("2. abc") == "abc"


def test_strip_spaced_prefix() -> None:
    assert TeacherAgent._strip_step_prefix("第 1 步：xxx") == "xxx"
    assert TeacherAgent._strip_step_prefix("步骤二：举例") == "举例"


def test_keep_leading_number_content() -> None:
    # 不带标点的“3个苹果”不是编号，不能误删
    assert TeacherAgent._strip_step_prefix("3个苹果") == "3个苹果"
