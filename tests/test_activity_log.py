"""活动日志（``core/activity_log.py`` + ``core/graph.py`` 埋点）单测。

钉住三件事：
1. **默认关闭**：不产生任何文件，图执行结果完全不受影响；
2. **开启后一次图执行能被读成一条链路**：连接顺序（拓扑序）、依赖、状态（执行/跳过/收工/失败）、耗时；
3. **观测故障绝不影响上课**：写不进去只 warning，不向调用方抛异常。
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Any

import pytest

from core import activity_log
from core.activity_log import ActivityLog
from core.graph import Node, NodeSpec, TaskGraph


@pytest.fixture
def log(tmp_path, monkeypatch) -> ActivityLog:
    """把进程级读写器换成写进临时目录的实例（不用改全局配置）。"""
    instance = ActivityLog(tmp_path, enabled=True)
    monkeypatch.setattr(activity_log, "get_activity_log", lambda: instance)
    return instance


def _spec(name: str, *, deps: tuple[str, ...] = (), optional: bool = False) -> NodeSpec:
    return NodeSpec(name=name, deps=deps, optional=optional)


def _add(graph: TaskGraph, name: str, handler: Any, *, deps: tuple[str, ...] = ()) -> TaskGraph:
    return graph.add(Node.bind(_spec(name, deps=deps), handler))


def test_disabled_writes_nothing(tmp_path, monkeypatch) -> None:
    """默认关闭：日志目录都不建，图执行结果照旧。"""
    monkeypatch.setattr(activity_log, "get_activity_log", lambda: ActivityLog(tmp_path, enabled=False))

    def only(ctx: dict) -> dict:
        return {"x": 1}

    run = TaskGraph("g").add(Node.bind(_spec("only"), only)).run_sync({})

    assert run.context == {"x": 1}
    assert list(tmp_path.iterdir()) == [], "关闭时不应产生任何落盘"


def test_run_records_chain_with_deps_and_status(log: ActivityLog) -> None:
    """一次执行 → 一条链路：拓扑序、依赖、状态、耗时、"跳过/收工"名单。"""
    session = SimpleNamespace(session_id="ses_1")

    def first(ctx: dict) -> dict:
        return {"a": 1}

    def second(ctx: dict) -> dict:
        return {"__skip__": True}

    def third(ctx: dict) -> dict:
        return {"__stop__": True, "b": 2}

    graph = TaskGraph("t.demo", session=session)
    _add(graph, "first", first)
    _add(graph, "second", second, deps=("first",))
    _add(graph, "third", third, deps=("second",))
    run = graph.run_sync({})

    runs = log.runs()
    assert len(runs) == 1
    item = runs[0]
    assert item["graph"] == "t.demo"
    assert item["session"] == "ses_1"
    assert item["order"] == ["first", "second", "third"], "run_start 应带完整连接顺序"
    assert [(n["node"], n["status"]) for n in item["nodes"]] == [
        ("first", activity_log.STATUS_OK),
        ("second", activity_log.STATUS_SKIP),
        ("third", activity_log.STATUS_STOP),
    ]
    assert item["nodes"][1]["deps"] == ["first"], "节点事件要带 deps，才看得出「从哪条连接进来」"
    assert item["skipped"] == ["second"] and item["stopped"] == ["third"]
    assert item["aborted"] is False
    assert item["total_ms"] == pytest.approx(sum(run.timings_ms.values()))
    # 每个节点都有 start 行（A：长节点执行期间也能看出「此刻在哪一步」）
    assert all(node["started_at"] for node in item["nodes"])
    assert item["running"] == [], "跑完的执行不应有「进行中」节点"
    kinds = [line["kind"] for line in log.lines()]
    assert kinds.count(activity_log.KIND_NODE_START) == 3
    assert kinds.count(activity_log.KIND_NODE) == 3


async def test_async_error_is_recorded_and_aborts(log: ActivityLog) -> None:
    """非 optional 节点失败：记 error + 中断标记，异常照旧抛给调用方。"""
    def bad(ctx: dict) -> dict:
        raise ValueError("boom")

    graph = TaskGraph("t.err", session="ses_e").add(Node.bind(_spec("bad"), bad))

    with pytest.raises(ValueError):
        await graph.run({})

    item = log.runs()[0]
    assert item["nodes"][0]["status"] == activity_log.STATUS_ERROR
    assert "ValueError: boom" in item["nodes"][0]["error"]
    assert item["aborted"] is True


def test_optional_error_marks_error_and_continues(log: ActivityLog) -> None:
    """optional 节点失败：记为 error 并计入跳过，后续节点继续跑。"""
    def bad(ctx: dict) -> dict:
        raise RuntimeError("nope")

    def after(ctx: dict) -> dict:
        return {"ok": True}

    graph = TaskGraph("t.opt")
    graph.add(Node.bind(_spec("bad", optional=True), bad))
    _add(graph, "after", after, deps=("bad",))
    run = graph.run_sync({})

    item = log.runs()[0]
    assert [n["status"] for n in item["nodes"]] == ["error", "ok"]
    assert item["skipped"] == ["bad"] and item["aborted"] is False
    assert run.context == {"ok": True}


def test_reader_filters_by_session_graph_and_limit(log: ActivityLog) -> None:
    """读取侧：按会话/图过滤，``limit`` 取最近几次，日期可从文件名还原。"""
    def noop(ctx: dict) -> dict:
        return {}

    for session_id in ("a", "b", "c"):
        _add(TaskGraph(f"g.{session_id}", session=session_id), "n", noop).run_sync({})

    assert [r["session"] for r in log.runs(session="b")] == ["b"]
    assert [r["graph"] for r in log.runs(graph="g.a")] == ["g.a"]
    assert [r["graph"] for r in log.runs(limit=2)] == ["g.b", "g.c"]
    assert log.days() == [time.strftime("%Y-%m-%d")]


def test_write_failure_never_raises(tmp_path) -> None:
    """落盘失败（目录位置被文件占住）只 warning，绝不向调用方抛异常。"""
    blocked = tmp_path / "not-a-dir"
    blocked.write_text("x", encoding="utf-8")
    instance = ActivityLog(blocked, enabled=True)

    instance.append({"kind": activity_log.KIND_NODE, "run_id": "r1"})

    assert instance.lines() == []
    assert instance.stats()["files"] == 0


def test_render_run_is_human_readable(log: ActivityLog) -> None:
    """查看脚本把链路渲染成「连接顺序 + 实际执行 + 失败原因」。"""
    from scripts.view_activity import render_run

    def bad(ctx: dict) -> dict:
        raise ValueError("炸了")

    graph = TaskGraph("g.render", session="s1")
    graph.add(Node.bind(_spec("bad", optional=True), bad))   # optional：不中断，便于渲染失败行
    graph.run_sync({})

    text = "\n".join(render_run(log.runs()[0]))

    assert "连接顺序: bad" in text
    assert "session=s1" in text
    assert "失败" in text and "炸了" in text
    assert "未收尾" not in text, "正常收尾的执行不该出现「未收尾」提示"


def test_interrupted_run_marks_running_node_and_unfinished(tmp_path) -> None:
    """进程被中断（只发 node_start 没发 node）：能看出卡在哪一步、以及距最后事件多久。

    直接构造 JSONL 行，而不是杀进程 —— 要钉的正是「事件不成对」这个形态。
    """
    from scripts.view_activity import render_run

    path = tmp_path / "activity" / "activity-2026-10-07.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        {"kind": "run_start", "run_id": "r1", "graph": "g.cut", "session": "s1",
         "order": ["a", "b"], "ts": 100.0},
        {"kind": "node_start", "run_id": "r1", "graph": "g.cut", "session": "s1", "node": "a", "ts": 100.1},
        {"kind": "node", "run_id": "r1", "graph": "g.cut", "session": "s1", "node": "a",
         "status": "ok", "elapsed_ms": 12.0, "ts": 100.2},
        {"kind": "node_start", "run_id": "r1", "graph": "g.cut", "session": "s1", "node": "b", "ts": 100.3},
    ]
    path.write_text("\n".join(json.dumps(item) for item in entries) + "\n", encoding="utf-8")

    log = ActivityLog(tmp_path, enabled=False)
    run = log.runs()[0]

    assert run["ended_at"] is None, "没有 run_end = 未收尾"
    assert run["nodes"][0]["started_at"] == pytest.approx(100.1)
    assert [item["node"] for item in run["running"]] == ["b"], "只发 start 没发 node 的节点即「进行中/中断」"

    text = "\n".join(render_run(run, now=130.3))

    assert "进行中/中断: b（起 " in text
    assert "已 30.0s" in text
    assert "未收尾" in text and "最后一条事件距今 30.0s" in text
