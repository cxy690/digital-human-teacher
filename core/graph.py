"""任务图：把一次答疑拆成有依赖关系的节点，按拓扑序执行。

相比写死的顺序调用，图化便于插入/跳过节点（例如作业模式下跳过「直接给答案」），
也方便观测每一步耗时。

节点有两层表示，刻意分开：

* :class:`NodeSpec` —— **静态说明**：节点名称、功能（``summary``）、读写的上下文键
  （``inputs`` / ``outputs``）、前置节点（``deps``，即连接顺序）、绑定的实现
  （``impl``，如 ``"TeacherAgent.answer"``）以及所需配置（``requires``）。
  每个 agent 的 ``agents/<角色>/nodes.py`` 先列 ``SPECS``，它既是构建图的输入，
  也是「谁负责哪些流程节点」这份说明的事实源（可被工具/自省读取）。
* :class:`Node` —— **可执行节点**：由 ``NodeSpec`` 与真实 handler 绑定而来
  （:meth:`Node.bind`），供 :class:`TaskGraph` 拓扑序执行。

节点通过返回值与图约定两件事：

* ``{"__skip__": True}`` —— 本节点本轮不适用（依赖未选中 / 条件不满足），跳过且**不写上下文**；
* ``{"__stop__": True}`` —— 本轮到此为止（校验拦截、用户打断等），其余键仍会写入上下文后收工。

这两条约定让「分支 + 早期返回」这类原本写成 if/elif 的流程也能图化，
而不必引入条件边（自研图刻意保持无分支依赖，便于与 LangGraph 版对照）。

执行轨迹（谁连谁、谁跳过、谁失败、各自多久）默认只攒在 :class:`GraphRun` 里随返回值丢失；
需要回溯时打开 ``settings.activity_log_enable``，逐节点落盘到 ``logs/activity/``
（见 :mod:`core.activity_log`，查看用 ``python scripts/view_activity.py``）。
每个节点**开始与结束各落一行**：长节点（LLM 生成可达十几秒）执行期间也能看出「此刻在哪一步」，
且只发 start 没发 finish 的那个节点，就是进程被中断时卡住的那一步。
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from config.logging import get_logger
from core import activity_log

log = get_logger("core.graph")

NodeHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]] | dict[str, Any]]

#: 图执行约定保留键：不写入上下文（``__skip__`` / ``__stop__`` 只作控制信号）
_CONTROL_KEYS = ("__skip__", "__stop__")


def _ms_since(started: float) -> float:
    """自 ``started`` 起经过的毫秒数（保留两位）。"""
    return round((time.perf_counter() - started) * 1000, 2)


@dataclass(frozen=True)
class NodeSpec:
    """节点声明（静态说明），见模块文档。"""

    #: 节点名（图内唯一，图执行与耗时统计都用它）
    name: str
    #: 一句话功能说明（人读；也用于自省输出）
    summary: str = ""
    #: 绑定的实现（人类可读，如 ``"Supervisor._turn_teach"``）
    impl: str = ""
    #: 从共享上下文读取的键
    inputs: tuple[str, ...] = ()
    #: 写回共享上下文的键
    outputs: tuple[str, ...] = ()
    #: 前置节点（连接顺序；先于本节点执行）
    deps: tuple[str, ...] = ()
    #: 失败是否只跳过本节点（True = 不中断整图）
    optional: bool = False
    #: 依赖的配置开关 / 外部能力（如 ``"settings.course_quiz_enable"``）
    requires: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """可序列化的节点说明（自省 / 前端展示用）。"""
        return {
            "name": self.name,
            "summary": self.summary,
            "impl": self.impl,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "deps": list(self.deps),
            "optional": self.optional,
            "requires": list(self.requires),
        }


@dataclass
class Node:
    """一个处理节点。"""

    name: str
    handler: NodeHandler
    deps: tuple[str, ...] = ()
    optional: bool = False
    #: 以下四项是说明性元数据（见 :class:`NodeSpec`），不参与执行
    summary: str = ""
    impl: str = ""
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()

    @classmethod
    def bind(cls, spec: NodeSpec, handler: NodeHandler) -> "Node":
        """按 ``spec`` 的声明绑定真实 handler，得到可执行节点。"""
        return cls(
            name=spec.name,
            handler=handler,
            deps=spec.deps,
            optional=spec.optional,
            summary=spec.summary,
            impl=spec.impl or getattr(handler, "__qualname__", ""),
            inputs=spec.inputs,
            outputs=spec.outputs,
            requires=spec.requires,
        )

    def to_spec(self) -> NodeSpec:
        """还原为静态声明（自省用）。"""
        return NodeSpec(
            name=self.name,
            summary=self.summary,
            impl=self.impl,
            inputs=self.inputs,
            outputs=self.outputs,
            deps=self.deps,
            optional=self.optional,
            requires=self.requires,
        )


@dataclass
class GraphRun:
    """一次图执行的结果与耗时。"""

    context: dict[str, Any]
    timings_ms: dict[str, float] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    #: 主动提前收工的节点（返回 ``__stop__``），其余节点未执行
    stopped: list[str] = field(default_factory=list)


class TaskGraph:
    """有向无环任务图：拓扑序执行，支持跳过（``__skip__``）与提前收工（``__stop__``）。

    执行入口两个：:meth:`run`（异步，handler 可同步可异步）与 :meth:`run_sync`
    （同步，仅同步 handler）—— 后者给「散落在同步调用栈里的确定性流程」用
    （如设备控制指令分发），不必为了图化把整条链路改成 async。
    执行前会 :meth:`validate`：缺依赖只告警，成环直接报错。
    """

    def __init__(self, name: str = "graph", *, session: Any = None) -> None:
        self.name = name
        #: 本图服务的会话（``SessionState`` 或 session_id）：只用于活动日志归组，不参与执行
        self.session = session
        self._nodes: dict[str, Node] = {}
        self._order: list[str] = []
        self._problems: list[str] | None = None

    # ---------------------------------------------------------------- 构建
    def add(self, node: Node) -> "TaskGraph":
        if node.name in self._nodes:
            log.warning("graph {}: node {} redefined", self.name, node.name)
        self._nodes[node.name] = node
        self._order.append(node.name)
        self._problems = None
        return self

    def bind_session(self, session: Any) -> "TaskGraph":
        """绑定会话（活动日志据此把一次课堂里的多条链路串起来）。"""
        self.session = session
        return self

    @classmethod
    def from_specs(
        cls,
        name: str,
        specs: tuple[NodeSpec, ...] | list[NodeSpec],
        handlers: dict[str, NodeHandler],
        *,
        session: Any = None,
    ) -> "TaskGraph":
        """按 ``specs`` 声明 + ``handlers``（``name -> handler``）装配一张图。

        缺少 handler 的节点**直接报错**：静默漏掉一个节点 = 流程少走一步，
        比构建期报错难查得多。
        """
        graph = cls(name, session=session)
        missing = [spec.name for spec in specs if spec.name not in handlers]
        if missing:
            # 宁可直接报错也不要静默漏节点：「流程少走一步」比「报错」难查得多
            raise ValueError(
                f"graph {name}: node(s) without handler: {', '.join(missing)}"
            )
        for spec in specs:
            graph.add(Node.bind(spec, handlers[spec.name]))
        return graph

    # ---------------------------------------------------------------- 自省 / 校验
    def order(self) -> list[str]:
        """拓扑序（即实际执行顺序）。"""
        return list(self._topo())

    def specs(self) -> list[NodeSpec]:
        """按执行顺序返回节点声明。"""
        return [self._nodes[name].to_spec() for name in self._topo()]

    def describe(self) -> list[dict[str, Any]]:
        """按执行顺序返回节点说明（自省 / 调试面板 / 文档生成用）。"""
        return [spec.to_dict() for spec in self.specs()]

    def validate(self) -> list[str]:
        """结构自检：未知依赖、自依赖、环。返回问题列表（空 = 配置正确）。"""
        if self._problems is None:
            self._problems = self._collect_problems()
        return list(self._problems)

    def _collect_problems(self) -> list[str]:
        problems: list[str] = []
        for name, node in self._nodes.items():
            for dep in node.deps:
                if dep == name:
                    problems.append(f"node {name} depends on itself")
                elif dep not in self._nodes:
                    problems.append(f"node {name} depends on unknown node {dep}")
        # 环检测（DFS 三色）：环会让拓扑序静默错位，必须显式报出来
        state: dict[str, int] = {}

        def walk(name: str, path: list[str]) -> None:
            state[name] = 1
            for dep in self._nodes[name].deps:
                if dep not in self._nodes:
                    continue
                if state.get(dep) == 1:
                    problems.append(
                        "cycle detected: " + " -> ".join([*path, name, dep])
                    )
                elif state.get(dep, 0) == 0:
                    walk(dep, [*path, name])
            state[name] = 2

        for name in self._order:
            if state.get(name, 0) == 0:
                walk(name, [])
        return problems

    def _topo(self) -> list[str]:
        visited: set[str] = set()
        in_progress: set[str] = set()       # 断环用：环上的节点按声明顺序排定，绝不无限递归
        result: list[str] = []

        def visit(name: str) -> None:
            if name in visited or name in in_progress:
                return
            in_progress.add(name)
            node = self._nodes[name]
            for dep in node.deps:
                if dep in self._nodes:
                    visit(dep)
            in_progress.discard(name)
            visited.add(name)
            result.append(name)

        for name in self._order:
            visit(name)
        return result

    # ---------------------------------------------------------------- 执行
    def _check_before_run(self) -> None:
        """执行前自检：缺依赖只告警（节点退化为入口），成环直接报错。

        环是**配置错误**而非运行期状况：拓扑序会静默错位，跑出来的流程看似正常、
        实际顺序不对，因此宁可直接抛错。
        """
        problems = self.validate()
        if not problems:
            return
        log.warning("graph {} config problem(s): {}", self.name, "；".join(problems))
        cycles = [p for p in problems if p.startswith("cycle detected")]
        if cycles:
            raise ValueError(f"graph {self.name} has a dependency cycle: {'; '.join(cycles)}")

    @staticmethod
    def _apply(
        run: GraphRun, node: Node, out: Any, context: dict[str, Any]
    ) -> bool:
        """把节点返回值写回上下文；返回 True 表示本轮到此收工（``__stop__``）。"""
        if not isinstance(out, dict):
            return False
        if out.get("__skip__"):
            run.skipped.append(node.name)
            return False
        context.update({k: v for k, v in out.items() if k not in _CONTROL_KEYS})
        if out.get("__stop__"):
            run.stopped.append(node.name)
            return True
        return False

    def _trace_node_start(self, run_id: str, node: Node) -> None:
        """节点开始执行就落一行：长节点执行期间据此看出「此刻在哪一步」。

        与 :meth:`_trace_node` 不成对出现的那个节点，正是进程被中断时卡住的那一步
        （查看侧归到 ``running``，渲染成「进行中/中断」）。
        """
        activity_log.node_started(
            run_id,
            graph=self.name,
            session=self.session,
            node=node.name,
            deps=node.deps,
        )

    def _trace_node(self, run_id: str, node: Node, status: str, error: str, elapsed_ms: float) -> None:
        """落一条节点活动日志（日志关闭时是空操作，开销可忽略）。"""
        activity_log.node_finished(
            run_id,
            graph=self.name,
            session=self.session,
            node=node.name,
            deps=node.deps,
            status=status,
            elapsed_ms=elapsed_ms,
            error=error,
        )

    def _trace_end(self, run_id: str, run: GraphRun, *, aborted: bool = False) -> None:
        """落一条图执行收尾日志（总耗时 + 执行 / 跳过 / 收工名单）。"""
        activity_log.run_finished(
            run_id,
            graph=self.name,
            session=self.session,
            total_ms=sum(run.timings_ms.values()),
            executed=list(run.timings_ms),
            skipped=run.skipped,
            stopped=run.stopped,
            aborted=aborted,
        )

    async def run(self, context: dict[str, Any]) -> GraphRun:
        """异步执行全图（handler 可为同步或异步函数）。"""
        self._check_before_run()
        run = GraphRun(context=context)
        order = self._topo()
        run_id = activity_log.run_started(self.name, session=self.session, order=order)
        for name in order:
            node = self._nodes[name]
            started = time.perf_counter()
            stop, status, error = False, activity_log.STATUS_OK, ""
            self._trace_node_start(run_id, node)
            try:
                out = node.handler(context)
                if asyncio.iscoroutine(out):
                    out = await out
                stop = self._apply(run, node, out, context)
                status = activity_log.status_of(
                    skipped=run.skipped, stopped=run.stopped, name=name, stop=stop
                )
            except Exception as err:  # noqa: BLE001
                log.exception("node {} failed: {}", name, err)
                status, error = activity_log.STATUS_ERROR, activity_log.error_text(err)
                elapsed = _ms_since(started)
                run.timings_ms[name] = elapsed
                self._trace_node(run_id, node, status, error, elapsed)
                if node.optional:
                    run.skipped.append(name)
                else:
                    self._trace_end(run_id, run, aborted=True)
                    raise
            else:
                elapsed = _ms_since(started)
                run.timings_ms[name] = elapsed
                self._trace_node(run_id, node, status, error, elapsed)
            if stop:
                break
        self._trace_end(run_id, run)
        return run

    def run_sync(self, context: dict[str, Any]) -> GraphRun:
        """同步执行全图（节点必须是**同步** handler）。

        用途：设备控制这类确定性、无 IO、且散落在同步调用栈里的流程
        （``Supervisor._handle_control`` 会被同步调用），不必为了图化把整条链路改成 async。
        节点若返回协程会立刻报错，避免「await 被漏掉」而静默什么都不做。
        """
        self._check_before_run()
        run = GraphRun(context=context)
        order = self._topo()
        run_id = activity_log.run_started(self.name, session=self.session, order=order)
        for name in order:
            node = self._nodes[name]
            started = time.perf_counter()
            stop, status, error = False, activity_log.STATUS_OK, ""
            self._trace_node_start(run_id, node)
            try:
                out = node.handler(context)
                if asyncio.iscoroutine(out):
                    out.close()          # 避免 "coroutine was never awaited" 告警
                    raise RuntimeError(
                        f"node {name} of graph {self.name} is async; use run() instead"
                    )
                stop = self._apply(run, node, out, context)
                status = activity_log.status_of(
                    skipped=run.skipped, stopped=run.stopped, name=name, stop=stop
                )
            except Exception as err:  # noqa: BLE001
                log.exception("node {} failed: {}", name, err)
                status, error = activity_log.STATUS_ERROR, activity_log.error_text(err)
                elapsed = _ms_since(started)
                run.timings_ms[name] = elapsed
                self._trace_node(run_id, node, status, error, elapsed)
                if node.optional:
                    run.skipped.append(name)
                else:
                    self._trace_end(run_id, run, aborted=True)
                    raise
            else:
                elapsed = _ms_since(started)
                run.timings_ms[name] = elapsed
                self._trace_node(run_id, node, status, error, elapsed)
            if stop:
                break
        self._trace_end(run_id, run)
        return run
