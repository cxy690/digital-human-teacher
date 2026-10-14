"""运行时写库测试：db.importer 幂等语义 / db.runtime_sync 开关与调度保护。

注意：这里全部用独立 SQLite 临时库验证导入逻辑，**不触碰** .env 指向的
真实数据库；调度保护（pytest 期间不落库、最新快照覆盖）单独验证。
"""
from __future__ import annotations

import threading
import time

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db.importer import msg_ts_ms, upsert_session


def _mkdb(tmp_path):
    """独立 SQLite 临时库。"""
    from db import models  # noqa: F401 - 触发模型注册
    from db.base import Base

    engine = create_engine(f"sqlite:///{tmp_path / 't.db'}")
    Base.metadata.create_all(engine)
    return engine


def _hist(*rows):
    return [{"role": r, "text": t, "ts_ms": ms} for r, t, ms in rows]


def test_upsert_new_then_idempotent(tmp_path):
    engine = _mkdb(tmp_path)
    rows = _hist(("student", "a", 1000), ("ai", "b", 2000))
    with Session(engine) as db:
        n1 = upsert_session(db, session_id="s1", student_id="u1", history=rows, saved_at=10.0)
        db.commit()
    assert n1 == 2
    with Session(engine) as db:
        n2 = upsert_session(db, session_id="s1", student_id="u1", history=rows, saved_at=20.0)
        db.commit()
    assert n2 == 2
    from db.models import MessageRecord, SessionRecord

    with Session(engine) as db:
        assert db.query(MessageRecord).filter_by(session_id="s1").count() == 2
        # 保留更晚的结束时间，不被旧快照覆盖
        assert db.get(SessionRecord, "s1").ended_at == 20.0


def test_upsert_history_extends_no_dup(tmp_path):
    """二次同步历史变长：消息整体重建为新集合，不累积重复。"""
    engine = _mkdb(tmp_path)
    with Session(engine) as db:
        upsert_session(db, session_id="s2", student_id="u2", history=_hist(("student", "a", 1000)), saved_at=10.0)
        db.commit()
        upsert_session(
            db, session_id="s2", student_id="u2",
            history=_hist(("student", "a", 1000), ("ai", "b", 2000)), saved_at=11.0,
        )
        db.commit()
    from db.models import MessageRecord

    with Session(engine) as db:
        assert db.query(MessageRecord).filter_by(session_id="s2").count() == 2


def test_msg_ts_compat():
    assert msg_ts_ms({"ts_ms": 1500}) == 1500
    assert msg_ts_ms({"ts": 1.5}) == 1500.0
    assert msg_ts_ms({}) == 0.0


def test_enabled_matrix(monkeypatch):
    """开关矩阵：off 永关；auto 仅 MySQL；on 强制。"""
    from config.settings import settings
    from db import runtime_sync

    monkeypatch.setattr(settings, "database_url", "sqlite:///x.db")
    monkeypatch.setattr(settings, "runtime_db_write", "auto")
    assert runtime_sync.enabled() is False
    monkeypatch.setattr(settings, "database_url", "mysql+pymysql://u:p@h/db")
    assert runtime_sync.enabled() is True
    monkeypatch.setattr(settings, "runtime_db_write", "off")
    assert runtime_sync.enabled() is False
    monkeypatch.setattr(settings, "runtime_db_write", "on")
    assert runtime_sync.enabled() is True


class _Msg:
    def __init__(self, role: str, text: str) -> None:
        self.role, self.text = role, text

    def model_dump(self):
        return {"role": self.role, "text": self.text, "ts_ms": 1}


class _State:
    def __init__(self, sid: str, history) -> None:
        self.session_id = sid
        self.student_id = "u"
        self.history = history


def test_maybe_sync_skipped_under_pytest(monkeypatch):
    """pytest 运行期间（PYTEST_CURRENT_TEST 存在）即使开关为 on 也不落库。"""
    from db import runtime_sync

    monkeypatch.setattr(runtime_sync, "enabled", lambda: True)
    runtime_sync.maybe_sync(_State("pytest-guard", []))
    with runtime_sync._lock:
        assert "pytest-guard" not in runtime_sync._pending


def test_maybe_sync_latest_wins(monkeypatch):
    """快速连续落盘：后台线程最终写入的是最新快照（覆盖式合并）。"""
    from db import runtime_sync

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(runtime_sync, "enabled", lambda: True)
    calls: list[list[str]] = []

    def fake_write(data):
        calls.append([m["text"] for m in data["history"]])

    monkeypatch.setattr(runtime_sync, "_write", fake_write)
    runtime_sync.maybe_sync(_State("merge", [_Msg("student", "v1")]))
    runtime_sync.maybe_sync(_State("merge", [_Msg("student", "v1"), _Msg("ai", "v2")]))

    deadline = time.time() + 5
    while time.time() < deadline:
        with runtime_sync._lock:
            if "merge" not in runtime_sync._running:
                break
        time.sleep(0.01)
    with runtime_sync._lock:
        assert "merge" not in runtime_sync._running
    assert calls, "后台线程应至少写入一次"
    assert calls[-1] == ["v1", "v2"], "最终写入的必须是最新快照"
