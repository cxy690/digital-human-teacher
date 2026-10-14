"""回合级调用链日志（core/trace.py）测试。

覆盖：

1. 关闭开关时**完全无副作用**（不打日志、不记步骤、行为不变）；
2. 开启时按调用层级记录「哪个 py 的哪个函数」；
3. 输出完成后打一条汇总日志，含函数名；耗时只在 ``TRACE_SHOW_MS`` 打开时才带；
4. 异常时标记为失败但**照常抛出**（追踪不能吞异常）；
5. 嵌套的 ``root`` 不会重复输出汇总。
"""
from __future__ import annotations

import asyncio

import pytest

from config.logging import logger as app_logger
from config.settings import settings
from core.trace import Step, current_steps, traced


# ---------------------------------------------------------------- 被测场景
@traced
async def _leaf() -> str:
    await asyncio.sleep(0)
    return "leaf"


@traced
async def _branch() -> str:
    return await _leaf()


@traced(root=True)
async def _root_ok() -> str:
    return await _branch()


@traced(root=True)
async def _root_nested() -> str:
    """根里再套一个 root：内层应退化为普通步骤。"""
    return await _root_ok()


@traced(root=True)
async def _root_boom() -> None:
    raise RuntimeError("boom")


@pytest.fixture()
def captured() -> list[str]:
    """收集本次测试期间所有日志消息。"""
    lines: list[str] = []
    sink_id = app_logger.add(lambda message: lines.append(message), level="INFO", format="{message}")
    try:
        yield lines
    finally:
        app_logger.remove(sink_id)


@pytest.fixture()
def trace_on(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "trace_enable", True)


# ---------------------------------------------------------------- 用例
async def test_disabled_keeps_everything_untouched(monkeypatch, captured) -> None:
    """关掉开关：不打日志、不记步骤，函数照常返回。"""
    monkeypatch.setattr(settings, "trace_enable", False)
    assert await _root_ok() == "leaf"
    assert current_steps() == []
    assert not [line for line in captured if "[trace]" in line]


async def test_records_steps_with_nesting(trace_on) -> None:
    """开启后，步骤按调用层级记录，且层级用 depth 表达。"""
    seen: list[Step] = []

    @traced
    async def _child() -> None:
        return None

    @traced
    async def _probe() -> None:
        await _child()
        # 此刻 _probe 与 _child 都已在栈上（进入即入栈）
        seen.extend(current_steps())

    @traced(root=True)
    async def _run() -> None:
        await _probe()

    await _run()
    labels = [s.label for s in seen]
    assert len(seen) == 2, f"应记录父与子两步，实际 {labels}"
    assert labels[0].endswith("_probe") and labels[1].endswith("_child")
    assert seen[0].depth == 0 and seen[1].depth == 1, "子步骤的层级必须是父 + 1"
    assert seen[0].ok is True and seen[1].ms >= 0
    # 回合结束后必须清空，不能污染下一次
    assert current_steps() == []


async def test_summary_log_lists_functions(trace_on, captured) -> None:
    """输出完成后打一条汇总日志，里面能看出调用了哪些函数。"""
    await _root_ok()
    text = "\n".join(captured)
    assert "[trace]" in text
    assert "输出完成" in text
    assert "_branch" in text and "_leaf" in text
    # 默认**不带耗时**：只看「哪个 py 的哪个函数」的调用顺序（耗时见下一个用例）
    assert "用时" not in text, "默认不该打耗时，TRACE_SHOW_MS 控制"


async def test_show_ms_appends_durations(monkeypatch, captured) -> None:
    """打开 ``TRACE_SHOW_MS``（``settings.trace_show_ms``）时，汇总与每步才带耗时。"""
    monkeypatch.setattr(settings, "trace_enable", True)
    monkeypatch.setattr(settings, "trace_show_ms", True)
    await _root_ok()
    text = "\n".join(captured)
    assert "用时" in text
    assert "ms" in text


async def test_nested_root_emits_single_summary(trace_on, captured) -> None:
    """根里套根时，只允许一条汇总（否则日志会翻倍）。"""
    await _root_nested()
    assert "\n".join(captured).count("输出完成") == 1


async def test_exception_is_marked_and_reraised(trace_on, captured) -> None:
    """异常要标出来，但不能被追踪吞掉。"""
    with pytest.raises(RuntimeError, match="boom"):
        await _root_boom()
    assert "异常中断" in "\n".join(captured)
    assert current_steps() == []


async def test_parallel_branch_is_bucketed_not_mixed(trace_on) -> None:
    """``asyncio.create_task`` 起的并行支线要**记录下来**，但不能串乱主链的层级。

    这里有个真实的权衡：
    * 若按 context 完全隔离 —— 并行支线看不见了，但 LangGraph 的图节点恰好跑在子任务里，
      线上日志会恒为「0 步」（踩过这个坑）；
    * 若共享一个可变 list —— 支线会把步骤塞进主链，层级错乱。
    所以改成「整回合共享一个 _Turn、按执行分支分桶」：都看得到，且各自算层级。
    """
    seen: list[Step] = []

    async def _background() -> None:
        await _leaf()          # 支线里也调用了带 @traced 的函数

    @traced
    async def _main_side() -> None:
        await _leaf()

    @traced(root=True)
    async def _run() -> None:
        task = asyncio.create_task(_background())
        await _main_side()
        await task
        seen.extend(current_steps())

    await _run()
    leaves = [s for s in seen if s.label.endswith("_leaf")]
    assert len(leaves) == 2, f"主链与支线应各记一次 _leaf：{[s.label for s in seen]}"
    assert len({s.branch for s in seen}) >= 2, "主链与并行支线必须记在不同分支上"
    # 支线自成一栈（depth 0）；主链里的 _leaf 位于 _main_side 之下（depth 1）
    assert sorted(s.depth for s in leaves) == [0, 1]


async def test_names_show_file_and_function(trace_on, captured) -> None:
    """日志里能直接看出「哪个 py 的哪个函数」，不用再翻代码找。"""
    await _root_ok()
    text = "\n".join(captured)
    assert "tests/test_trace.py:_branch" in text, f"应形如 文件:函数，实际：{text}"


def test_log_format_includes_function_and_line() -> None:
    """日志格式带上「函数名:行号」，排查时不用再翻代码。

    直接读源码文本断言（而不是掏 loguru 的私有属性），避免版本升级把测试搞脆。
    """
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "config" / "logging.py").read_text(
        encoding="utf-8"
    )
    assert "{function}" in source and "{line}" in source
