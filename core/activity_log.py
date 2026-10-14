"""活动日志：把「节点到节点」的图执行轨迹落成 append-only JSONL，落在 ``logs/activity/``。

为什么需要它：

* ``core/graph.py`` 的 :class:`~core.graph.GraphRun` 只把耗时/跳过/收工攒在内存里，
  调用方取走 ``response`` 之后这份轨迹就丢了 —— 事后无法回答「这次答疑走了哪几个节点、
  从哪条连接（deps）进来的、第几步挂了」；
* 各 handler 里的 info 日志是**业务痕迹**（讲了第几页、识别到什么意图），看不到**连接关系**；
* ``agents/protocol.py`` 的 ``AgentBus`` 只记 agent 之间的转交（80 条环缓，且不记逐回合调度），
  不覆盖图内节点。

设计取舍：

* **append-only JSONL**：追加写不会写坏，坏行读盘时跳过，纯文本可直接看 / grep / tail -f；
* **默认关闭**（``settings.activity_log_enable``）：逐节点记录量很大（连讲课件一页就有 4~5 条），
  长会话会把磁盘写满，因此只在排障时打开（``.env`` 里 ``ACTIVITY_LOG_ENABLE=true``）；
* **按天滚动 + 单文件上限 + 保留天数**：``activity-2026-10-07.jsonl``，超过
  ``activity_log_max_mb`` 后按 ``.1/.2`` 序号续写，写入时顺手清理过期文件；
* **绝不影响上课**：任何写盘异常都吞掉（只 warning 一次），调用方永远拿不到异常。

一次图执行产生 ``2N + 2`` 行：``run_start`` / (``node_start`` + ``node``) × N / ``run_end``，
按 ``run_id`` 归组即一条完整链路；查看用 ``python scripts/view_activity.py``。

``node_start`` 的价值有两层：

1. **长节点执行期间也能看出「此刻在哪一步」** —— LLM 生成可达十几秒，只有完成事件时
   这段时间日志一行不动，看不出是卡住了还是在正常发呆；
2. **与 ``node`` 不成对出现的那个节点 = 进程被中断时正在执行的节点** ——
   读取侧把它归到 ``running`` 字段，查看脚本渲染成「进行中/中断」（见 ``scripts/view_activity.py``）。
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from config.logging import get_logger
from config.settings import settings

log = get_logger("core.activity_log")

# ---------------------------------------------------------------- 事件类型
KIND_RUN_START = "run_start"
KIND_NODE_START = "node_start"
KIND_NODE = "node"
KIND_RUN_END = "run_end"

# ---------------------------------------------------------------- 节点状态
STATUS_OK = "ok"          # 正常执行完
STATUS_SKIP = "skip"      # 本轮不适用（``__skip__``）
STATUS_STOP = "stop"      # 主动收工（``__stop__``），后续节点不再执行
STATUS_ERROR = "error"    # 抛异常（optional 节点会被跳过，非 optional 会中断整图）

_STATUS_LABELS = {
    STATUS_OK: "执行",
    STATUS_SKIP: "跳过",
    STATUS_STOP: "收工",
    STATUS_ERROR: "失败",
}

#: 单条错误摘要长度上限：完整堆栈在 loguru 日志里，这里只留定位信息
ERROR_MAX_CHARS = 200

#: 文件名前缀（``activity-<日期>.jsonl``）
DAY_PREFIX = "activity-"


def status_label(status: str) -> str:
    """节点状态的中文标签（脚本 / 接口展示用）。"""
    return _STATUS_LABELS.get(status, status or "-")


def status_of(*, skipped: Iterable[str], stopped: Iterable[str], name: str, stop: bool) -> str:
    """按执行结果判定节点状态：收工 > 跳过 > 执行（``error`` 由调用方直接判定）。"""
    if stop or name in set(stopped):
        return STATUS_STOP
    if name in set(skipped):
        return STATUS_SKIP
    return STATUS_OK


def error_text(err: BaseException) -> str:
    """异常 → 一行摘要（换行压平 + 截断，避免把长堆栈塞进日志）。"""
    text = " ".join(str(err).split())
    return f"{type(err).__name__}: {text}"[:ERROR_MAX_CHARS]


def new_run_id() -> str:
    """一次图执行的唯一 id（节点事件靠它归组成链路）。"""
    return uuid.uuid4().hex[:12]


def _session_id(session: Any) -> str:
    """会话标识：``SessionState`` 取 ``session_id``，字符串原样用，其它给空串。"""
    if session is None:
        return ""
    if isinstance(session, str):
        return session
    return str(getattr(session, "session_id", "") or "")


class ActivityLog:
    """活动日志读写器（写入按天滚动，读取按 run 归组）。"""

    def __init__(self, root: str | Path | None = None, *, enabled: bool | None = None) -> None:
        #: 显式 root / enabled 只给测试与脚本用；留空则跟随配置
        self._root = Path(root) if root is not None else None
        self._enabled = enabled
        self._lock = threading.Lock()
        self._warned = False

    # ---------------------------------------------------------------- 配置
    @property
    def root(self) -> Path:
        """日志根目录（默认取配置 ``activity_log_dir``，便于测试/多环境改写）。"""
        return self._root or Path(settings.activity_log_dir)

    @property
    def dir(self) -> Path:
        """活动日志目录：``<根目录>/activity``（与其它运行日志同层分目录）。"""
        return self.root / "activity"

    @property
    def enabled(self) -> bool:
        return bool(settings.activity_log_enable if self._enabled is None else self._enabled)

    @property
    def max_bytes(self) -> int:
        """单文件体积上限（字节）；0 = 不限制。"""
        return max(0, int(float(settings.activity_log_max_mb) * 1024 * 1024))

    @property
    def keep_days(self) -> int:
        """保留天数；0 = 不清理。"""
        return max(0, int(settings.activity_log_keep_days))

    # ---------------------------------------------------------------- 写
    def append(self, entry: dict[str, Any]) -> None:
        """追加一条事件（自动补 ``ts``）。关闭时零开销，失败只 warning。"""
        if not self.enabled:
            return
        payload = {"ts": time.time(), **entry}
        try:
            with self._lock:
                self.dir.mkdir(parents=True, exist_ok=True)
                self._prune_locked()
                path = self._write_path(payload)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as err:  # noqa: BLE001 - 观测失败绝不能影响上课
            self._warn_once(f"活动日志写入失败（{self.dir}）：{err}")

    def _write_path(self, entry: dict[str, Any]) -> Path:
        """今天的落点：超上限就换序号（``activity-<日期>.1.jsonl`` ...）。"""
        day = _day_of(entry.get("ts"))
        if self.max_bytes <= 0:
            return self.dir / f"{DAY_PREFIX}{day}.jsonl"
        index = 0
        path = self.dir / f"{DAY_PREFIX}{day}.jsonl"
        while path.exists() and path.stat().st_size >= self.max_bytes:
            index += 1
            path = self.dir / f"{DAY_PREFIX}{day}.{index}.jsonl"
        return path

    def _prune_locked(self) -> None:
        """清理超过保留天数的旧文件（只在写入时顺手做，不另起后台线程）。"""
        if self.keep_days <= 0:
            return
        cutoff = date.today() - timedelta(days=self.keep_days)
        for path in self.dir.glob(f"{DAY_PREFIX}*.jsonl"):
            day = _day_from_name(path.name)
            if day is None or day >= cutoff:
                continue
            try:
                path.unlink()
            except OSError:  # noqa: BLE001 - 删不掉不影响继续写
                pass

    def _warn_once(self, message: str) -> None:
        if self._warned:
            return
        self._warned = True
        log.warning(message)

    # ---------------------------------------------------------------- 读
    def files(self, day: str | None = None) -> list[Path]:
        """按天的日志文件（``day`` 为空 = 全部，按文件名即时间序）。"""
        pattern = f"{DAY_PREFIX}{day}*.jsonl" if day else f"{DAY_PREFIX}*.jsonl"
        try:
            return sorted(self.dir.glob(pattern))
        except OSError:
            return []

    def days(self) -> list[str]:
        """已有记录的日期（升序）。"""
        found: set[str] = set()
        for path in self.files():
            day = _day_from_name(path.name)
            if day is not None:
                found.add(day.isoformat())
        return sorted(found)

    def lines(
        self,
        *,
        day: str | None = None,
        session: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """原始事件行（按时间升序；``limit`` 取最近若干条）。"""
        out: list[dict[str, Any]] = []
        for path in self.files(day):
            for entry in self._read_file(path):
                if session and str(entry.get("session") or "") != session:
                    continue
                out.append(entry)
        out.sort(key=lambda item: float(item.get("ts") or 0.0))
        return out[-limit:] if limit and limit > 0 else out

    def runs(
        self,
        *,
        day: str | None = None,
        session: str | None = None,
        graph: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """把事件行按 ``run_id`` 归组成链路（最近 ``limit`` 条，按时间升序）。"""
        grouped: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for entry in self.lines(day=day, session=session):
            run_id = str(entry.get("run_id") or "")
            if not run_id:
                continue
            if graph and str(entry.get("graph") or "") != graph:
                continue
            run = grouped.get(run_id)
            if run is None:
                run = {
                    "run_id": run_id,
                    "graph": str(entry.get("graph") or ""),
                    "session": str(entry.get("session") or ""),
                    "started_at": float(entry.get("ts") or 0.0),
                    "ended_at": None,
                    "last_ts": float(entry.get("ts") or 0.0),
                    "total_ms": None,
                    "order": [],
                    "nodes": [],
                    "running": [],
                    "executed": [],
                    "skipped": [],
                    "stopped": [],
                    "aborted": False,
                    "_pending": {},          # node_start 已发、node 未回的节点（内部字段，出参前转成 running）
                }
                grouped[run_id] = run
                order.append(run_id)
            self._merge(run, entry)
        out: list[dict[str, Any]] = []
        for run_id in order:
            run = grouped[run_id]
            pending = run.pop("_pending", {})
            # 有 start 无 finish = 此刻正在执行（或进程被中断时卡住的那个节点）
            run["running"] = [
                {"node": name, "started_at": ts}
                for name, ts in sorted(pending.items(), key=lambda item: item[1])
            ]
            out.append(run)
        return out[-limit:] if limit and limit > 0 else out

    @staticmethod
    def _merge(run: dict[str, Any], entry: dict[str, Any]) -> None:
        kind = entry.get("kind")
        run["last_ts"] = float(entry.get("ts") or run.get("last_ts") or 0.0)
        if kind == KIND_RUN_START:
            run["order"] = [str(name) for name in entry.get("order") or []]
            run["started_at"] = float(entry.get("ts") or run["started_at"])
        elif kind == KIND_NODE_START:
            run["_pending"][str(entry.get("node") or "")] = float(entry.get("ts") or 0.0)
        elif kind == KIND_NODE:
            node = str(entry.get("node") or "")
            run["nodes"].append({
                "node": node,
                "deps": [str(dep) for dep in entry.get("deps") or []],
                "status": str(entry.get("status") or STATUS_OK),
                "elapsed_ms": float(entry.get("elapsed_ms") or 0.0),
                "error": str(entry.get("error") or ""),
                "started_at": run["_pending"].pop(node, None),
            })
        elif kind == KIND_RUN_END:
            run["ended_at"] = float(entry.get("ts") or 0.0)
            run["total_ms"] = float(entry.get("total_ms") or 0.0)
            run["executed"] = [str(name) for name in entry.get("executed") or []]
            run["skipped"] = [str(name) for name in entry.get("skipped") or []]
            run["stopped"] = [str(name) for name in entry.get("stopped") or []]
            run["aborted"] = bool(entry.get("aborted"))

    def stats(self) -> dict[str, Any]:
        """概况（运维 / 接口查看用）。"""
        files = self.files()
        total = 0
        for path in files:
            try:
                total += path.stat().st_size
            except OSError:
                continue
        days = self.days()
        return {
            "dir": str(self.dir),
            "enabled": self.enabled,
            "days": len(days),
            "files": len(files),
            "bytes": total,
            "latest_day": days[-1] if days else "",
        }

    def _read_file(self, path: Path) -> list[dict[str, Any]]:
        """读一个文件的事件行；坏行跳过（追加写不怕写一半崩掉）。"""
        out: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        log.warning("活动日志坏行已跳过：{}:{}", path.name, line_no)
                        continue
                    if isinstance(raw, dict):
                        out.append(raw)
        except OSError as err:
            log.warning("读取活动日志失败（{}）：{}", path, err)
        return out


# ---------------------------------------------------------------- 埋点便捷函数
@lru_cache(maxsize=1)
def get_activity_log() -> ActivityLog:
    """进程级默认读写器（root / enabled 延迟到真正读写时从配置取）。"""
    return ActivityLog()


def run_started(graph: str, *, session: Any = None, order: Iterable[str] = ()) -> str:
    """记录一次图执行的开始（含完整拓扑序），返回 ``run_id`` 供后续事件归组。"""
    run_id = new_run_id()
    get_activity_log().append({
        "kind": KIND_RUN_START,
        "run_id": run_id,
        "graph": graph,
        "session": _session_id(session),
        "order": [str(name) for name in order],
    })
    return run_id


def node_started(
    run_id: str,
    *,
    graph: str,
    node: str,
    session: Any = None,
    deps: Iterable[str] = (),
) -> None:
    """记录节点**开始**执行：长节点（LLM 可达十几秒）执行期间也能看出「此刻在哪一步」。"""
    if not run_id:
        return
    get_activity_log().append({
        "kind": KIND_NODE_START,
        "run_id": run_id,
        "graph": graph,
        "session": _session_id(session),
        "node": node,
        "deps": [str(dep) for dep in deps],
    })


def node_finished(
    run_id: str,
    *,
    graph: str,
    node: str,
    session: Any = None,
    deps: Iterable[str] = (),
    status: str = STATUS_OK,
    elapsed_ms: float = 0.0,
    error: str = "",
) -> None:
    """记录一个节点的执行结果（状态 / 依赖 / 耗时 / 失败原因）。"""
    if not run_id:
        return
    get_activity_log().append({
        "kind": KIND_NODE,
        "run_id": run_id,
        "graph": graph,
        "session": _session_id(session),
        "node": node,
        "deps": [str(dep) for dep in deps],
        "status": status,
        "elapsed_ms": round(float(elapsed_ms), 2),
        "error": error,
    })


def run_finished(
    run_id: str,
    *,
    graph: str,
    session: Any = None,
    total_ms: float = 0.0,
    executed: Iterable[str] = (),
    skipped: Iterable[str] = (),
    stopped: Iterable[str] = (),
    aborted: bool = False,
) -> None:
    """记录一次图执行的收尾（总耗时 / 执行、跳过、收工名单 / 是否中断）。"""
    if not run_id:
        return
    get_activity_log().append({
        "kind": KIND_RUN_END,
        "run_id": run_id,
        "graph": graph,
        "session": _session_id(session),
        "total_ms": round(float(total_ms), 2),
        "executed": [str(name) for name in executed],
        "skipped": [str(name) for name in skipped],
        "stopped": [str(name) for name in stopped],
        "aborted": bool(aborted),
    })


# ---------------------------------------------------------------- 文件名解析
def _day_of(ts: Any) -> str:
    """时间戳 → 本地日期字符串（跨天自动换文件）。"""
    try:
        return time.strftime("%Y-%m-%d", time.localtime(float(ts)))
    except (TypeError, ValueError, OSError):
        return time.strftime("%Y-%m-%d")


def _day_from_name(name: str) -> date | None:
    """``activity-2026-10-07.1.jsonl`` → ``date(2026, 10, 7)``；解析不出返回 None。"""
    if not name.startswith(DAY_PREFIX):
        return None
    try:
        return date.fromisoformat(name[len(DAY_PREFIX):][:10])
    except ValueError:
        return None


__all__ = [
    "ActivityLog",
    "KIND_NODE",
    "KIND_NODE_START",
    "KIND_RUN_END",
    "KIND_RUN_START",
    "STATUS_ERROR",
    "STATUS_OK",
    "STATUS_SKIP",
    "STATUS_STOP",
    "error_text",
    "get_activity_log",
    "new_run_id",
    "node_finished",
    "node_started",
    "run_finished",
    "run_started",
    "status_label",
    "status_of",
]
