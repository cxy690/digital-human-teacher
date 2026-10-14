"""自研任务图（``core/graph.py``）的节点语义与自检单测。

钉住三件事：
1. ``__skip__`` / ``__stop__`` 两条约定的语义（跳过不写上下文、收工仍写上下文）；
2. ``validate`` 能报出未知依赖与环，且成环时**拒绝执行**（拓扑序静默错位最难查）；
3. ``run_sync`` 只接受同步 handler，避免「async handler 被同步调用」而静默什么都不做。
"""
from __future__ import annotations

import pytest

from core.graph import Node, NodeSpec, TaskGraph


def _spec(name: str, *, deps: tuple[str, ...] = (), optional: bool = False) -> NodeSpec:
    return NodeSpec(
        name=name,
        summary=f"{name} 的说明",
        impl=f"handler.{name}",
        inputs=(),
        outputs=(),
        deps=deps,
        optional=optional,
    )


def test_skip_does_not_write_context_and_stop_ends_run() -> None:
    """``__skip__`` 不写上下文；``__stop__`` 写完其余键后收工（后续节点不执行）。"""
    called: list[str] = []

    def first(ctx: dict) -> dict:
        called.append("first")
        return {"__skip__": True, "ignored": 1}

    def second(ctx: dict) -> dict:
        called.append("second")
        return {"__stop__": True, "kept": 2}

    def third(ctx: dict) -> dict:
        called.append("third")
        return {"never": 3}

    graph = (
        TaskGraph("t")
        .add(Node.bind(_spec("first"), first))
        .add(Node.bind(_spec("second", deps=("first",)), second))
        .add(Node.bind(_spec("third", deps=("second",)), third))
    )
    run = graph.run_sync({})

    assert called == ["first", "second"], "收工后不应再执行后续节点"
    assert run.skipped == ["first"]
    assert run.stopped == ["second"]
    assert run.context == {"kept": 2}, "保留键不能进上下文，其余键要写入"


async def test_async_handler_works_in_run_and_rejected_by_run_sync() -> None:
    """同一个节点：``run`` 支持协程 handler，``run_sync`` 必须明确报错。"""
    async def handler(ctx: dict) -> dict:
        return {"done": True}

    graph = TaskGraph("t").add(Node.bind(_spec("a"), handler))

    run = await graph.run({})
    assert run.context == {"done": True}

    with pytest.raises(RuntimeError):
        graph.run_sync({})


def test_validate_reports_unknown_dep_and_cycle_and_refuses_to_run() -> None:
    """未知依赖只告警（节点退化为入口）；成环则拒绝执行。"""
    graph = TaskGraph("bad").add(Node.bind(_spec("a", deps=("ghost",)), lambda ctx: {}))
    assert any("unknown node ghost" in p for p in graph.validate())
    assert graph.run_sync({}).context == {}      # 软问题：照常跑

    cyc = (
        TaskGraph("cyc")
        .add(Node.bind(_spec("a", deps=("b",)), lambda ctx: {}))
        .add(Node.bind(_spec("b", deps=("a",)), lambda ctx: {}))
    )
    assert any(p.startswith("cycle detected") for p in cyc.validate())
    with pytest.raises(ValueError):
        cyc.run_sync({})


def test_optional_node_failure_is_skipped_not_fatal() -> None:
    """``optional=True`` 的节点抛异常只被跳过；非可选节点必须把异常抛出去。"""
    def boom(ctx: dict) -> dict:
        raise RuntimeError("boom")

    soft = (
        TaskGraph("soft")
        .add(Node.bind(_spec("boom", optional=True), boom))
        .add(Node.bind(_spec("after", deps=("boom",)), lambda ctx: {"ok": True}))
    )
    run = soft.run_sync({})
    assert run.skipped == ["boom"] and run.context == {"ok": True}

    hard = TaskGraph("hard").add(Node.bind(_spec("boom"), boom))
    with pytest.raises(RuntimeError):
        hard.run_sync({})


def test_from_specs_requires_every_handler_and_keeps_topological_order() -> None:
    """建图按声明顺序拓扑排序；缺 handler 直接报错（不静默漏节点）。"""
    def make(name: str):
        return lambda ctx: {}

    specs = (_spec("b", deps=("a",)), _spec("a"))
    graph = TaskGraph.from_specs("g", specs, {"a": make("a"), "b": make("b")})
    assert graph.order() == ["a", "b"], "拓扑序应把依赖排在前面"
    assert [spec.name for spec in graph.specs()] == ["a", "b"]
    assert graph.describe()[0]["summary"] == "a 的说明"

    with pytest.raises(ValueError):
        TaskGraph.from_specs("g", specs, {"a": make("a")})
