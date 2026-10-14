"""Skill loop 更新时间线：把 append-only JSONL 还原成「谁、在什么时候、被更新了什么」。

为什么需要它
------------
自我更新链路（复盘 → 落盘 → 注入 → 战绩回写 → 停用/恢复）每一步都不改文件，
只往 ``data/teacher_experience/<老师>.jsonl`` 与 ``data/skill_usage/<技能>.jsonl``
**追加一行快照**。行本身就是审计日志，但它是给机器读的：几十上百行里九成是
``pulls`` 递增，人眼翻不出「这个循环到底在干什么」。

本模块把这批行**重建为更新事件**（学习 / 强化 / 战绩回写 / 停用 / 恢复 / 损坏），
并算好两样人真正想看的东西：

* **前后对比**：这一条更新把哪些字段从什么改成了什么（``before → after``）；
* **结果**：``ok`` / ``warn`` / ``fail`` 三态 —— 成功、需要注意（停用、没帮上忙）、失败（文件损坏）。

接口与前端只负责展示，不再理解 JSONL 的折叠规则。

边界与取舍
----------
* **只读**：不写任何文件、不改学习链路的任何行为（看板崩了也不影响上课与学习）；
* **推断而非埋点**：行里没有「事件类型」字段，类型是按同一 ``id`` 相邻两次快照的
  字段差分推断的（够了，也避免为了看板去改热路径的写盘格式）；
* **批次合并**：同源同动作且在 :data:`BATCH_WINDOW_SECONDS` 内的连续写入算「一次更新」——
  否则时间线会被「pulls 1→2、2→3……」刷屏；合并后每个条目的完整差分仍然保留。
  窗口取值是折中：真实数据里一次记账内部的写入间隔在毫秒级，而两次记账之间
  至少隔着一个答疑回合，1 秒足以切开它们。
"""
from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from config.logging import get_logger
from core.skills.agent_skills.usage import SkillUsage
from core.skills.experience.models import KIND_LABELS, ExperienceItem

log = get_logger("core.skills.loop_timeline")

#: 事件动作 -> 中文标签（前端直接用，不在页面里再维护一份映射）
ACTION_LABELS: dict[str, str] = {
    "learn": "学到新经验",
    "reinforce": "命中强化",
    "outcome": "战绩回写",
    "retire": "停用经验",
    "restore": "恢复经验",
    "usage": "调用记账",
    "disable": "停用技能",
    "enable": "恢复技能",
    "corrupt": "记录损坏",
}

#: 结果三态 -> 中文标签
RESULT_LABELS: dict[str, str] = {"ok": "成功", "warn": "警告", "fail": "失败"}

#: 动作 -> 「这次更新为什么发生」：看板上每条事件都回答一次触发条件
_TRIGGER: dict[str, str] = {
    "learn": "答疑结束 → 异步复盘提炼出通用经验（长度合规、置信度过阈值、与已有条目不重复）",
    "reinforce": "复盘再次提到同一条经验（归一化相似度 ≥ 0.85）→ 命中次数累加，不新增条目",
    "outcome": "回合收口 → 按裁决结果（pass / degraded / failed）把实测收益回写进被注入的经验",
    "retire": "人工停用（经验抽检台）或超出条数上限时的自动淘汰",
    "restore": "人工恢复（经验抽检台）",
    "usage": "回合结束 → 被调用过的技能各记一次收益（整批同记，与经验层同一取舍）",
    "disable": "人工停用（技能抽检台）或战绩过差时的自动停用（需打开 AGENT_SKILL_AUTO_RETIRE）",
    "enable": "人工恢复（技能抽检台）",
    "corrupt": "读取落盘文件时发现无法解析的损坏行（写了一半被 kill 等）",
}

#: 连续同类写入合并成「一次更新」的时间窗（秒）
BATCH_WINDOW_SECONDS = 1.0

#: 单个文件最多解析多少行（超大文件只取尾部，防止一次请求扫爆内存）
MAX_LINES_PER_FILE = 4000

#: 默认 / 上限返回事件数（聚合后的条数）
DEFAULT_LIMIT = 500
MAX_LIMIT = 2000

_STATUS_LABELS: dict[str, str] = {"active": "生效中", "retired": "已停用", "disabled": "已停用"}

_RESULT_ORDER: dict[str, int] = {"ok": 0, "warn": 1, "fail": 2}

#: 经验快照的对比字段（顺序即展示顺序）
_EXP_FIELDS: tuple[tuple[str, str], ...] = (
    ("kind", "类型"),
    ("hits", "命中次数"),
    ("confidence", "置信度"),
    ("status", "状态"),
    ("pulls", "注入回合数"),
    ("wins", "有效回合数"),
    ("last_gain", "最近收益"),
)

#: 技能战绩快照的对比字段
_USAGE_FIELDS: tuple[tuple[str, str], ...] = (
    ("status", "状态"),
    ("pulls", "调用回合数"),
    ("wins", "有效回合数"),
    ("gain_rate", "有效率"),
    ("last_gain", "最近收益"),
)

_LEARN_FIELDS: tuple[tuple[str, str], ...] = (("kind", "类型"), ("confidence", "置信度"))
_STATUS_FIELDS: tuple[tuple[str, str], ...] = (("status", "状态"),)
_OUTCOME_FIELDS: tuple[tuple[str, str], ...] = (
    ("pulls", "注入回合数"),
    ("wins", "有效回合数"),
    ("last_gain", "最近收益"),
)
_REINFORCE_FIELDS: tuple[tuple[str, str], ...] = (("hits", "命中次数"), ("confidence", "置信度"))
_USAGE_FIELDS_CHANGED: tuple[tuple[str, str], ...] = (
    ("pulls", "调用回合数"),
    ("wins", "有效回合数"),
    ("gain_rate", "有效率"),
    ("last_gain", "最近收益"),
)
_USAGE_FIRST_FIELDS: tuple[tuple[str, str], ...] = (
    ("pulls", "调用回合数"),
    ("last_gain", "最近收益"),
)


# ---------------------------------------------------------------- 对外入口
def build_loop_timeline(
    experience_dir: str | Path,
    usage_dir: str | Path,
    *,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """重建 skill loop 更新时间线（最新在前）。

    返回 ``{generated_at, total_events, events, stats, sources}``：
    ``events`` 是聚合后的更新事件，``stats`` / ``sources`` 始终按**全量**统计，
    不受 ``limit`` 影响（看板顶部的 KPI 不该因为只取最近 N 条就变小）。
    """
    now = time.time()
    events: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []
    for path in _jsonl_files(Path(experience_dir)):
        batches, info = _experience_batches(path)
        events.extend(batches)
        meta.append(info)
    for path in _jsonl_files(Path(usage_dir)):
        batches, info = _usage_batches(path)
        events.extend(batches)
        meta.append(info)
    events.sort(
        key=lambda e: (-float(e.get("ts") or 0.0), str(e.get("scope") or ""), str(e.get("source") or ""))
    )
    cap = max(1, min(int(limit), MAX_LIMIT))
    return {
        "generated_at": now,
        "total_events": len(events),
        "events": [_public(e) for e in events[:cap]],
        "stats": _stats(events, meta, now),
        "sources": _source_summary(events, meta),
    }


# ---------------------------------------------------------------- 经验（教师 Skill 自我更新）
def _experience_batches(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """一个老师的经验文件 -> 更新事件 + 该老师的条目现状。"""
    teacher = path.stem
    lines, truncated = _read_lines(path)
    fallback_ts = _mtime(path)
    prev: dict[str, dict[str, Any]] = {}
    batches: list[dict[str, Any]] = []
    for line_no, raw in lines:
        item = _parse_row(raw, ExperienceItem)
        if item is None:
            _push_batch(
                batches,
                scope="experience",
                source=teacher,
                action="corrupt",
                result="fail",
                ts=fallback_ts,
                line_no=line_no,
                file=path.name,
                note="这一行无法解析成经验快照，读盘时已跳过（其余条目不受影响）",
                items=[{
                    "target": f"{path.name}:{line_no}",
                    "name": raw[:80],
                    "kind": "",
                    "kind_label": "",
                    "status": "",
                    "created": False,
                    "changes": [],
                    "sources": [],
                }],
            )
            continue
        current = item.to_dict()
        before = prev.get(item.id)
        prev[item.id] = current
        action, result, note, fields = _classify_experience(before, current)
        _push_batch(
            batches,
            scope="experience",
            source=teacher,
            action=action,
            result=result,
            ts=float(current.get("updated_at") or 0.0),
            line_no=line_no,
            file=path.name,
            note=note,
            items=[{
                "target": item.id,
                "name": current.get("text") or "",
                "kind": current.get("kind") or "",
                "kind_label": KIND_LABELS.get(str(current.get("kind") or ""), ""),
                "status": _STATUS_LABELS.get(str(current.get("status") or ""), ""),
                "created": before is None,
                "changes": _changes(before, current, fields),
                "sources": list(current.get("sources") or []),
            }],
        )
    info = {
        "scope": "experience",
        "source": teacher,
        "file": path.name,
        "items": len(prev),
        "active": sum(1 for it in prev.values() if it.get("status") == "active"),
        "truncated": truncated,
    }
    return batches, info


def _classify_experience(
    before: dict[str, Any] | None, after: dict[str, Any]
) -> tuple[str, str, str, Sequence[tuple[str, str]]]:
    """相邻两次快照 -> ``(动作, 结果, 说明, 对比字段)``。"""
    if before is None:
        return (
            "learn",
            "ok",
            "复盘产出一条新经验并立即生效：从下一轮讲解起参与注入",
            _LEARN_FIELDS,
        )
    if before.get("status") != after.get("status"):
        if after.get("status") == "active":
            return "restore", "ok", "经验已恢复：重新参与注入", _STATUS_FIELDS
        return (
            "retire",
            "warn",
            "经验已停用：不再注入提示词（人工止损或超限自动淘汰，可在经验抽检台恢复）",
            _STATUS_FIELDS,
        )
    if (
        before.get("pulls") != after.get("pulls")
        or before.get("wins") != after.get("wins")
        or before.get("last_gain") != after.get("last_gain")
    ):
        gain = float(after.get("last_gain") or 0.0)
        delta = int(after.get("pulls") or 0) - int(before.get("pulls") or 0)
        if gain >= 0.5:
            return (
                "outcome",
                "ok",
                f"本回合裁决通过（gain={gain:.0f}）：这批经验帮上了忙，注入次数 +{delta}",
                _OUTCOME_FIELDS,
            )
        return (
            "outcome",
            "warn",
            f"本回合裁决未通过（gain={gain:.0f}）：这批经验没帮上忙，已计入分母（注入次数 +{delta}）",
            _OUTCOME_FIELDS,
        )
    return (
        "reinforce",
        "ok",
        "复盘再次命中这条经验：命中次数累加（不新增条目）",
        _REINFORCE_FIELDS,
    )


# ---------------------------------------------------------------- Agent 技能战绩
def _usage_batches(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """一个技能的战绩文件 -> 更新事件 + 该技能现状。"""
    name = path.stem
    lines, truncated = _read_lines(path)
    fallback_ts = _mtime(path)
    prev: dict[str, Any] | None = None
    batches: list[dict[str, Any]] = []
    for line_no, raw in lines:
        usage = _parse_row(raw, SkillUsage)
        if usage is None:
            _push_batch(
                batches,
                scope="usage",
                source=name,
                action="corrupt",
                result="fail",
                ts=fallback_ts,
                line_no=line_no,
                file=path.name,
                note="这一行无法解析成战绩快照，读盘时已跳过",
                items=[{
                    "target": f"{path.name}:{line_no}",
                    "name": raw[:80],
                    "kind": "",
                    "kind_label": "",
                    "status": "",
                    "created": False,
                    "changes": [],
                    "sources": [],
                }],
            )
            continue
        current = usage.to_dict()
        before = prev
        prev = current
        action, result, note, fields = _classify_usage(before, current)
        _push_batch(
            batches,
            scope="usage",
            source=name,
            action=action,
            result=result,
            ts=float(current.get("updated_at") or 0.0),
            line_no=line_no,
            file=path.name,
            note=note,
            items=[{
                "target": name,
                "name": name,
                "kind": "",
                "kind_label": "",
                "status": _STATUS_LABELS.get(str(current.get("status") or ""), ""),
                "created": before is None,
                "changes": _changes(before, current, fields),
                "sources": [],
            }],
        )
    state = prev or {}
    info = {
        "scope": "usage",
        "source": name,
        "file": path.name,
        "items": 1 if prev else 0,
        "active": 1 if state.get("status") == "active" else 0,
        "status": str(state.get("status") or ""),
        "truncated": truncated,
    }
    return batches, info


def _classify_usage(
    before: dict[str, Any] | None, after: dict[str, Any]
) -> tuple[str, str, str, Sequence[tuple[str, str]]]:
    if before is None:
        return "usage", "ok", "技能开始记账：首次调用收益已落盘", _USAGE_FIRST_FIELDS
    if before.get("status") != after.get("status"):
        if after.get("status") == "active":
            return "enable", "ok", "技能已恢复：重新暴露给模型", _STATUS_FIELDS
        return (
            "disable",
            "warn",
            "技能已停用：不再暴露给模型（可在技能抽检台恢复）",
            _STATUS_FIELDS,
        )
    if (
        before.get("pulls") != after.get("pulls")
        or before.get("wins") != after.get("wins")
        or before.get("last_gain") != after.get("last_gain")
    ):
        gain = float(after.get("last_gain") or 0.0)
        delta = int(after.get("pulls") or 0) - int(before.get("pulls") or 0)
        if gain >= 0.5:
            return "usage", "ok", f"调用后裁决通过：技能帮上了忙（调用次数 +{delta}）", _USAGE_FIELDS_CHANGED
        return "usage", "warn", f"调用后裁决未通过：技能没帮上忙（调用次数 +{delta}）", _USAGE_FIELDS_CHANGED
    return "usage", "ok", "技能战绩快照更新", _USAGE_FIELDS_CHANGED


# ---------------------------------------------------------------- 批次与差分
def _push_batch(
    batches: list[dict[str, Any]],
    *,
    scope: str,
    source: str,
    action: str,
    result: str,
    ts: float,
    line_no: int,
    file: str,
    note: str,
    items: list[dict[str, Any]],
) -> None:
    """把一条快照并入「当前这次更新」，或另起一次（见模块头部的批次说明）。"""
    last = batches[-1] if batches else None
    if (
        last is not None
        and last["action"] == action
        and last["scope"] == scope
        and last["source"] == source
        and last["_last_ts"] <= ts <= last["_last_ts"] + BATCH_WINDOW_SECONDS
    ):
        _merge_items(last["items"], items)
        last["writes"] += 1
        last["line_to"] = line_no
        last["ts"] = max(float(last["ts"]), ts)
        last["_last_ts"] = max(float(last["_last_ts"]), ts)
        if _RESULT_ORDER.get(result, 0) > _RESULT_ORDER.get(str(last["result"]), 0):
            last["result"] = result
            last["result_label"] = RESULT_LABELS.get(result, result)
        return
    batches.append({
        "id": f"{scope}:{source}:{line_no}",
        "scope": scope,
        "source": source,
        "action": action,
        "action_label": ACTION_LABELS.get(action, action),
        "result": result,
        "result_label": RESULT_LABELS.get(result, result),
        "ts": ts,
        "note": note,
        "trigger": _TRIGGER.get(action, ""),
        "items": items,
        "writes": 1,
        "file": file,
        "line_from": line_no,
        "line_to": line_no,
        "_last_ts": ts,
    })


def _merge_items(target: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> None:
    """把新快照并进批次：同一目标只留一条，「变化」取**最早的 before → 最新的 after**。

    为什么必须归并：一次记账会把该批全部条目的快照逐行追加，同一 ``id`` 因此在
    同一批次里出现多次。若原样堆进事件，看板上同一条经验会重复出现好几行。
    """
    index = {row["target"]: row for row in target}
    for row in incoming:
        old = index.get(row["target"])
        if old is None:
            target.append(row)
            index[row["target"]] = row
            continue
        merged: dict[str, dict[str, Any]] = {c["field"]: dict(c) for c in old["changes"]}
        for change in row["changes"]:
            prev = merged.get(change["field"])
            if prev is None:
                merged[change["field"]] = dict(change)
            else:
                prev["after"] = change["after"]
        old["changes"] = [
            c for c in merged.values()
            if c.get("before") is None or c.get("before") != c.get("after")
        ]
        for key in ("name", "kind", "kind_label", "status"):
            if row.get(key):
                old[key] = row[key]


def _changes(
    before: dict[str, Any] | None,
    after: dict[str, Any],
    fields: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    """字段级前后对比（只保留真正变了的字段；新增条目时 before 为空）。"""
    rows: list[dict[str, Any]] = []
    for key, label in fields:
        raw_before = None if before is None else before.get(key)
        raw_after = after.get(key)
        if before is not None and raw_before == raw_after:
            continue
        rows.append({
            "field": key,
            "label": label,
            "before": _display(key, raw_before),
            "after": _display(key, raw_after),
        })
    return rows


def _display(field: str, value: Any) -> Any:
    """值 -> 展示文本（状态翻中文、浮点收敛到尾数，前端不再做格式化）。"""
    if value is None:
        return None
    if field == "status":
        return _STATUS_LABELS.get(str(value), str(value))
    if isinstance(value, float):
        return round(value, 3)
    return value


# ---------------------------------------------------------------- 统计
def _stats(events: list[dict[str, Any]], meta: list[dict[str, Any]], now: float) -> dict[str, Any]:
    """全量统计。

    频率口径按 **writes（真实落盘次数）** 而不是事件条数：时间线会把同一次更新里
    连续写入的多个条目合成一条展示，若按条数算频率，看板上的「最近 24 小时更新次数」
    会比实际落盘次数少一半。
    """
    by_action: dict[str, int] = {}
    by_result: dict[str, int] = {"ok": 0, "warn": 0, "fail": 0}
    marks: list[tuple[float, int]] = []
    learn_stamps: list[float] = []
    for event in events:
        action = str(event.get("action") or "")
        result = str(event.get("result") or "ok")
        by_action[action] = by_action.get(action, 0) + 1
        by_result[result] = by_result.get(result, 0) + 1
        ts = float(event.get("ts") or 0.0)
        if ts > 0:
            marks.append((ts, max(1, int(event.get("writes") or 1))))
            if action == "learn":
                learn_stamps.append(ts)

    def _window(seconds: float) -> int:
        return sum(count for ts, count in marks if 0 <= now - ts <= seconds)

    exp = [m for m in meta if m.get("scope") == "experience"]
    use = [m for m in meta if m.get("scope") == "usage"]
    return {
        "events": len(events),
        "writes": sum(count for _, count in marks),
        "ok": by_result.get("ok", 0),
        "warn": by_result.get("warn", 0),
        "fail": by_result.get("fail", 0),
        "by_action": by_action,
        "last_event_at": max(learn_stamps + [ts for ts, _ in marks]) if marks else 0.0,
        "last_learn_at": max(learn_stamps) if learn_stamps else 0.0,
        "writes_1h": _window(3600),
        "writes_24h": _window(86400),
        "writes_7d": _window(7 * 86400),
        "experience_items": sum(int(m.get("items") or 0) for m in exp),
        "experience_active": sum(int(m.get("active") or 0) for m in exp),
        "experience_files": len(exp),
        "usage_skills": len(use),
        "usage_active": sum(1 for m in use if m.get("status") == "active"),
        "truncated": any(bool(m.get("truncated")) for m in meta),
    }


def _source_summary(
    events: list[dict[str, Any]], meta: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """按来源（老师 / 技能）聚合，供看板的来源筛选下拉直接使用。"""
    agg: dict[tuple[str, str], dict[str, Any]] = {}

    def _row(scope: str, source: str) -> dict[str, Any]:
        return agg.setdefault((scope, source), {
            "scope": scope,
            "source": source,
            "events": 0,
            "writes": 0,
            "ok": 0,
            "warn": 0,
            "fail": 0,
            "last_at": 0.0,
            "items": 0,
            "active": 0,
            "truncated": False,
        })

    for event in events:
        row = _row(str(event.get("scope") or ""), str(event.get("source") or ""))
        row["events"] += 1
        row["writes"] += max(1, int(event.get("writes") or 1))
        result = str(event.get("result") or "ok")
        if result in ("ok", "warn", "fail"):
            row[result] += 1
        row["last_at"] = max(float(row["last_at"]), float(event.get("ts") or 0.0))
    for item in meta:
        row = _row(str(item.get("scope") or ""), str(item.get("source") or ""))
        row["items"] = int(item.get("items") or 0)
        row["active"] = int(item.get("active") or 0)
        row["file"] = item.get("file") or ""
        row["truncated"] = bool(item.get("truncated"))
    return sorted(agg.values(), key=lambda r: (-float(r["last_at"]), r["scope"], r["source"]))


# ---------------------------------------------------------------- 读盘（只读、坏行不炸）
def _jsonl_files(root: Path) -> list[Path]:
    try:
        return sorted(p for p in root.glob("*.jsonl") if p.is_file())
    except OSError:
        return []


def _read_lines(path: Path) -> tuple[list[tuple[int, str]], bool]:
    """读文件 -> ``[(真实行号, 行文本)]``；超大文件只取尾部（保证接口不被打爆）。"""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as err:
        log.warning("读取落盘文件失败（{}）：{}", path, err)
        return [], False
    all_lines = text.splitlines()
    truncated = len(all_lines) > MAX_LINES_PER_FILE
    lines = all_lines[-MAX_LINES_PER_FILE:] if truncated else all_lines
    start = len(all_lines) - len(lines) + 1
    return [(start + i, line) for i, line in enumerate(lines)], truncated


def _parse_row(raw: str, model: Any) -> Any:
    """一行 -> 模型对象；空行 / 损坏行返回 ``None``（由调用方记成 corrupt 事件）。"""
    text = (raw or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return model.from_dict(data)


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _public(event: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in event.items() if not k.startswith("_")}


__all__ = ["ACTION_LABELS", "RESULT_LABELS", "build_loop_timeline"]
