# -*- coding: utf-8 -*-
"""回合级调用链日志：一次学生输入经过了哪些函数、各花多久，输出后汇总成一条日志。

为什么要它
----------
答疑不是「一次函数调用」，而是一条链：

    入口 → 意图路由 → 分行（闲聊 / 控制 / 危机 / 答疑 / 判题）
         → LLM 生成 → 白板 → TTS 合成 → 下发

出问题时只看零散日志很难还原「这一步到底走了谁、卡在哪」。这里用**极轻量**的方式补齐——
给关键函数挂一个 ``@traced``，一次输出结束时自动打一条（缩进表示调用层级）：

    [trace] core/supervisor.py:Supervisor.handle_turn session=abc 输出完成 · 5 步
       core/supervisor.py:Supervisor._handle_turn_builtin
          core/router.py:IntentRouter.route_async
          core/supervisor.py:Supervisor._turn_teach
             agents/teacher/agent.py:TeacherAgent.answer

默认**只列「哪个 py 的哪个函数」**（``文件:函数``，按调用顺序缩进），不掺耗时，读起来干净；
需要看性能时把 ``settings.trace_show_ms``（``.env`` 的 ``TRACE_SHOW_MS=true``）打开，
汇总行与每步后面会追加耗时毫秒。

三条设计取舍
------------
1. **按「执行分支」分桶，而不是按 contextvar 隔离**：LangGraph 会把图节点放进 ``asyncio``
   子任务里跑，若按 context 隔离，根节点根本看不到节点日志（线上默认走 langgraph，
   日志会恒为「0 步」——这是踩过的坑）。所以整回合共享一个 ``_Turn`` 对象，
   各分支按任务 id 分桶记，汇总时合并，并行支线额外标注。
2. **用装饰器不手写 log**：加一处 ``@traced`` 就自动带出「模块.函数名 + 耗时」；
   闭包形式的图节点用 ``@traced(name="turn_graph.teach")`` 给个短名字。
   调「哪个 Agent 的哪个节点」这种动态名字用手动埋点 ``with step("a2a.call:teaching.judge")``。
3. **只在关键位置埋点**：不做全项目插桩，否则日志淹没主干。

开关：``settings.trace_enable``（默认开；排查完想清静可置 false，行为完全不变）。
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import functools
import inspect
import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence

from config.logging import get_logger
from config.settings import settings

log = get_logger("core.trace")

#: 一个回合最多记录多少步（超出的不再记，函数照常执行；防止异常情况下刷屏）
MAX_STEPS = 32


@dataclass
class Step:
    """调用链上的一步。"""

    seq: int            # 开始序号（越小越早）
    depth: int          # 在本分支内的嵌套层级（0 = 该分支最外层）
    label: str          # 步骤名（模块.函数，或自定义名）
    ms: float           # 耗时（毫秒）
    ok: bool = True     # 是否正常返回（False = 抛异常）
    branch: int = 0     # 执行分支标识（= asyncio 任务 id；0 = 无任务上下文）
    file: str = ""      # 所在文件（``core/supervisor.py``）—— 回答「哪个 py」
    func: str = ""      # 函数名（``Supervisor._present_slide``）—— 回答「哪个函数」
    desc: str = ""      # 中文说明（``呈现一页``）—— 让日志「说人话」，空则只显示 文件:函数


@dataclass
class _Turn:
    """一次输出的追踪上下文：**整回合共享**，各执行分支分桶记录。"""

    root_branch: int
    started: float = field(default_factory=time.perf_counter)
    counter: Iterator[int] = field(default_factory=lambda: itertools.count(1))
    total: int = 0                                                  # 已记录步数（含活跃）
    stacks: dict[int, list[Step]] = field(default_factory=dict)      # 分支 -> 活跃栈（算层级）
    done: dict[int, list[Step]] = field(default_factory=dict)        # 分支 -> 已完成步骤

    def steps(self) -> list[Step]:
        """所有分支的步骤，按开始顺序合并（活跃的排在各自分支末尾）。"""
        merged = [s for bucket in self.done.values() for s in bucket]
        merged.extend(s for stack in self.stacks.values() for s in stack)
        return sorted(merged, key=lambda s: s.seq)


#: 当前回合；None 表示没在追踪（未开启或不在回合内）
_TURN: contextvars.ContextVar[_Turn | None] = contextvars.ContextVar("turn_trace", default=None)


def tracing_enabled() -> bool:
    """是否开启调用链追踪（走 ``settings.trace_enable``）。"""
    return bool(getattr(settings, "trace_enable", True))


def current_steps() -> list[Step]:
    """当前回合已记录的步骤（含并行支线，按开始顺序；不在追踪中返回空列表）。"""
    turn = _TURN.get()
    return turn.steps() if turn is not None else []


def _branch_key() -> int:
    """当前执行分支：同一条 await 链是同一个；``create_task`` 出来的是新分支。"""
    try:
        task = asyncio.current_task()
    except RuntimeError:  # 没有事件循环（同步调用）
        return 0
    return id(task) if task is not None else 0


# ---------------------------------------------------------------------- 记录
def _enter(label: str, file: str = "", func: str = "", desc: str = "") -> Step | None:
    """登记一步「进入」；返回 None 表示这步不记（未开启或超出上限）。"""
    turn = _TURN.get()
    if turn is None or turn.total >= MAX_STEPS:
        return None
    branch = _branch_key()
    stack = turn.stacks.setdefault(branch, [])
    step = Step(
        seq=next(turn.counter), depth=len(stack), label=label, ms=0.0,
        branch=branch, file=file, func=func or label, desc=desc,
    )
    stack.append(step)
    turn.total += 1
    return step


def _leave(step: Step | None, start: float, ok: bool) -> None:
    """登记一步「离开」：出活跃栈、进已完成桶。"""
    turn = _TURN.get()
    if turn is None or step is None:
        return
    step.ok = ok
    step.ms = (time.perf_counter() - start) * 1000
    stack = turn.stacks.get(step.branch)
    if stack:
        for i in range(len(stack) - 1, -1, -1):
            if stack[i] is step:
                del stack[i]
                break
    turn.done.setdefault(step.branch, []).append(step)


# ---------------------------------------------------------------------- 装饰器
def _default_label(fn: Callable[..., Any]) -> str:
    """``模块.类.函数``；闭包函数（图节点、A2A 节点）去掉 ``<locals>`` 噪音。"""
    qual = str(getattr(fn, "__qualname__", fn.__name__)).replace(".<locals>.", ".")
    return f"{fn.__module__}.{qual}"


def _file_of(fn: Callable[..., Any]) -> str:
    """``core.supervisor`` -> ``core/supervisor.py``——回答「哪个 py」。"""
    module = str(getattr(fn, "__module__", "") or "")
    return module.replace(".", "/") + ".py" if module else ""


def _func_of(fn: Callable[..., Any], name: str | None) -> str:
    """显示用的函数名：给了自定义 ``name`` 就用它，否则用 qualname（去掉 ``<locals>``）。"""
    if name:
        return name
    return str(getattr(fn, "__qualname__", fn.__name__)).replace(".<locals>.", ".")


def traced(
    func: Callable[..., Any] | None = None,
    *,
    root: bool = False,
    name: str | None = None,
    desc: str | None = None,
):
    """把函数记为调用链上的一步。

    用法::

        @traced                      # 普通步骤：只在已开启的回合里记录
        async def _turn_teach(...): ...

        @traced(root=True)           # 回合根：由它开启追踪，并在返回时输出汇总日志
        async def handle_turn(...): ...

        @traced(name="turn_graph.teach")   # 自定义标签：闭包里的图节点用这个更好认
        async def teach(...): ...

        @traced(desc="呈现一页")      # 中文说明：日志打成「呈现一页  core/supervisor.py · _present_slide」
        async def _present_slide(...): ...  # —— 让不熟悉代码的人也能顺着日志读流程

    同步函数同样支持。``root=True`` 的函数若被另一层 root 包住，
    会自动退化为普通步骤（不会重复输出汇总）。
    """

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        label = name or _default_label(fn)
        file = _file_of(fn)
        func = _func_of(fn, name)
        step_desc = desc or ""
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                if _TURN.get() is None:
                    if root and tracing_enabled():
                        return await _root_async(fn, args, kwargs, label, file, func, step_desc)
                    return await fn(*args, **kwargs)
                return await _step_async(fn, args, kwargs, label, file, func, step_desc)

            return wrapper

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
            if _TURN.get() is None:
                if root and tracing_enabled():
                    return _root_sync(fn, args, kwargs, label, file, func, step_desc)
                return fn(*args, **kwargs)
            return _step_sync(fn, args, kwargs, label, file, func, step_desc)

        return wrapper

    return decorate(func) if func is not None else decorate


# ---------------------------------------------------------------------- 手动埋点
@contextlib.contextmanager
def step(label: str) -> Iterator[None]:
    """手动记录一步，标签由调用方给定 —— 用于「哪个 agent 的哪个节点」这类动态名字。

    用法::

        with step(f"a2a.call:{agent_name}.{node}"):
            task = await client.call(...)

    未开启追踪（或不在回合内）时是零开销的空上下文。
    """
    if _TURN.get() is None:
        yield
        return
    current = _enter(label, "", label)
    start = time.perf_counter()
    ok = True
    try:
        yield
    except BaseException:
        ok = False
        raise
    finally:
        _leave(current, start, ok)


# ---------------------------------------------------------------------- 执行
async def _root_async(fn, args, kwargs, label, file="", func="", desc=""):
    turn = _Turn(root_branch=_branch_key())
    token = _TURN.set(turn)
    ok = True
    try:
        return await fn(*args, **kwargs)
    except BaseException:
        ok = False
        raise
    finally:
        steps = turn.steps()
        elapsed = (time.perf_counter() - turn.started) * 1000
        _TURN.reset(token)
        _emit(label, steps, elapsed, ok, args, file, func, desc)


async def _step_async(fn, args, kwargs, label, file="", func="", desc=""):
    step_ = _enter(label, file, func, desc)
    start = time.perf_counter()
    ok = True
    try:
        return await fn(*args, **kwargs)
    except BaseException:
        ok = False
        raise
    finally:
        _leave(step_, start, ok)


def _root_sync(fn, args, kwargs, label, file="", func="", desc=""):
    turn = _Turn(root_branch=_branch_key())
    token = _TURN.set(turn)
    ok = True
    try:
        return fn(*args, **kwargs)
    except BaseException:
        ok = False
        raise
    finally:
        steps = turn.steps()
        elapsed = (time.perf_counter() - turn.started) * 1000
        _TURN.reset(token)
        _emit(label, steps, elapsed, ok, args, file, func, desc)


def _step_sync(fn, args, kwargs, label, file="", func="", desc=""):
    step_ = _enter(label, file, func, desc)
    start = time.perf_counter()
    ok = True
    try:
        return fn(*args, **kwargs)
    except BaseException:
        ok = False
        raise
    finally:
        _leave(step_, start, ok)


# ---------------------------------------------------------------------- 输出
def _session_hint(args: tuple) -> str:
    """尽量从根函数的第一个参数上取会话号（取不到就不带，绝不因此报错）。"""
    try:
        session = getattr(args[0], "session", None)
        sid = str(getattr(session, "session_id", "") or "")
    except Exception:  # noqa: BLE001 - 日志辅助信息，任何异常都不该影响主链路
        return ""
    return f" session={sid}" if sid else ""


def _describe(file: str, func: str, desc: str, fallback: str) -> str:
    """一步的显示名。

    * 有中文说明（课件链路都带了）：``呈现一页  core/supervisor.py · _present_slide``
      —— 不熟代码的人也能顺着读，同时保留「哪个 py 的哪个函数」便于定位；
    * 没有中文说明：退回 ``core/supervisor.py:Supervisor._present_slide``。
    """
    if desc:
        short = func.rsplit(".", 1)[-1] if func else ""     # 去掉类名，只留方法名
        where = f"{file} · {short}" if file else (short or func)
        return f"{desc}  {where}"
    if file and func:
        return f"{file}:{func}"
    return fallback


def _step_name(step: Step) -> str:
    """一步在日志里的显示名（见 :func:`_describe`）。"""
    return _describe(step.file, step.func, step.desc, step.label)


def _emit(
    label: str, steps: Sequence[Step], total_ms: float, ok: bool, args: tuple,
    file: str = "", func: str = "", desc: str = "",
) -> None:
    """把一次输出的调用链打出来（默认 INFO 级）。

    默认**只列调用了哪个 py 的哪个函数**（缩进表示层级），不掺耗时；
    带 ``desc`` 的埋点前面会多一句中文说明。要排查性能瓶颈时打开
    ``settings.trace_show_ms``，汇总与每步后面会追加毫秒。
    """
    show_ms = bool(getattr(settings, "trace_show_ms", False))
    root_name = _describe(file, func, desc, label)
    branches = {s.branch for s in steps}
    head = (
        f"[trace] {root_name}{_session_hint(args)} 输出完成 · {len(steps)} 步"
        + (f" · 用时 {total_ms:.0f}ms" if show_ms else "")
        + ("" if ok else " · 异常中断")
        # 多分支只在头部提一句：LangGraph 的节点本来就跑在子任务里，
        # 逐行标「并行」等于每行都标，反而成了噪音。
        + (f" · {len(branches)} 个执行分支" if len(branches) > 1 else "")
    )
    if not steps:
        log.info(head)
        return
    # 按「开始先后」还原顺序：已完成桶是按完成顺序追加的（内层先完成、父级最后），
    # 直接打印会看到父级排在最后、像个倒叙。
    shown = sorted(steps, key=lambda s: s.seq)[:MAX_STEPS]
    lines = [
        "{}{}{}{}".format(
            "   " * (s.depth + 1),
            _step_name(s),
            f"  {s.ms:.0f}ms" if show_ms else "",
            "  ✗" if not s.ok else "",
        )
        for s in shown
    ]
    if len(steps) > len(shown):
        lines.append(f"   … 另有 {len(steps) - len(shown)} 步未显示")
    log.info("{}\n{}", head, "\n".join(lines))


def describe() -> dict[str, Any]:
    """自检用：追踪模块当前状态。"""
    turn = _TURN.get()
    return {
        "enabled": tracing_enabled(),
        "in_turn": turn is not None,
        "steps": len(turn.steps()) if turn is not None else 0,
        "branches": len(turn.done) + len(turn.stacks) if turn is not None else 0,
        "max_steps": MAX_STEPS,
    }
