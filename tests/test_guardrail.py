"""教学护栏与内容安全（G5）单测。"""
from __future__ import annotations

from core.guardrail import TeachingGuardrail


def test_screen_question_out_of_scope():
    g = TeachingGuardrail(enable=True)
    verdict = g.screen_question("老师，明天的世界杯比分是多少？", topic_keywords=["分数", "约分"])
    assert verdict.in_scope is False
    assert verdict.redirect_speech


def test_screen_question_in_scope():
    g = TeachingGuardrail(enable=True)
    verdict = g.screen_question("分数的基本性质是什么？", topic_keywords=["分数", "约分"])
    assert verdict.in_scope is True
    assert verdict.redirect_speech == ""


def test_screen_disabled_never_blocks():
    g = TeachingGuardrail(enable=False)
    verdict = g.screen_question("完全无关的问题", topic_keywords=["分数"])
    assert verdict.in_scope is True


def test_filter_output_masks_unsafe():
    g = TeachingGuardrail(safety_enable=True)
    safe, verdict = g.filter_output("你怎么这么笨蛋，去死吧")
    assert verdict.allowed is False
    assert "笨蛋" not in safe and "去死" not in safe
    assert verdict.unsafe


def test_filter_output_keeps_safe_text():
    g = TeachingGuardrail(safety_enable=True)
    safe, verdict = g.filter_output("分数的分子和分母同时乘 2，大小不变。")
    assert verdict.allowed is True
    assert safe.startswith("分数")


def test_tag_ai_content_adds_watermark():
    g = TeachingGuardrail()
    tagged = g.tag_ai_content({"speech": "hello"})
    assert tagged["ai_generated"] is True
    assert tagged["watermark"]
    assert tagged["speech"] == "hello"
