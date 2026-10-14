"""会话数据导入：检查点数据 → 数据库三张表（students / sessions / messages）。

两条使用路径共用本模块，保证语义完全一致：

* ``scripts/pull_chat_to_mysql.py`` —— 批量同步从云端拉回来的检查点文件；
* ``db/runtime_sync.py`` —— 本机运行时，会话落盘的同时实时镜像写库。

幂等约定（与脚本历史行为一致）：students / sessions 按主键 upsert
（sessions 保留更晚的 ended_at）；messages 按会话「先删后插」重建 ——
同一会话重复导入不会产生重复消息，随时可重放。
"""
from __future__ import annotations

import time

from sqlalchemy.orm import Session

from db.models import MessageRecord, SessionRecord, Student


def msg_ts_ms(item: dict) -> float:
    """兼容两种消息时间字段：ts_ms（毫秒，现网格式）/ ts（秒，早期格式）。"""
    if item.get("ts_ms"):
        return float(item["ts_ms"])
    if item.get("ts"):
        return float(item["ts"]) * 1000.0
    return 0.0


def upsert_session(
    db: Session,
    *,
    session_id: str,
    student_id: str,
    history: list[dict],
    saved_at: float | None = None,
) -> int:
    """把一个会话连同其消息写入数据库（幂等），返回写入的消息条数。

    调用方负责事务边界（commit / rollback）；本函数内部只 flush，
    便于批量导入时「单个坏文件不影响整批」。
    """
    history = list(history or [])
    saved_at = float(time.time() if saved_at is None else saved_at)

    if db.get(Student, student_id) is None:
        db.add(Student(id=student_id, name="", grade=""))
        db.flush()

    first_ms = next((t for t in (msg_ts_ms(m) for m in history) if t > 0), saved_at * 1000.0)
    sess = db.get(SessionRecord, session_id)
    if sess is None:
        db.add(SessionRecord(
            id=session_id, student_id=student_id,
            started_at=first_ms / 1000.0, ended_at=saved_at,
        ))
    else:  # 已导入过：保留更晚的结束时间，不覆盖
        sess.ended_at = max(sess.ended_at or 0.0, saved_at)
    db.flush()

    # 按会话重建消息（幂等的关键）：重复同步不会累积重复消息
    db.query(MessageRecord).filter(MessageRecord.session_id == session_id).delete()
    count = 0
    for item in history:
        db.add(MessageRecord(
            session_id=session_id,
            role=str(item.get("role") or "")[:16],
            text=str(item.get("text") or ""),
            ts=msg_ts_ms(item) / 1000.0,
        ))
        count += 1
    return count
