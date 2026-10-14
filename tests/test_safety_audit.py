"""P1.5 安全审计与教师端监控台测试。

守住三件事：

1. **落盘语义** —— 只追加、倒序读取、关闭时不写、单行损坏不拖垮整份日志；
2. **接口契约** —— ``/api/safety/events`` 返回可复核的记录，且 limit 有钳制；
3. **前端安全带** —— 事件文本来自学生输入与模型输出，必须转义后再拼 HTML。
"""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from config.settings import settings
from core.safety_audit import SafetyAudit
from server import routes
from server.app import app

TEACHER_PAGE = (
    Path(__file__).resolve().parents[1] / "clients" / "teacher_web" / "index.html"
)


# ============================================================ 落盘语义
def test_record_and_recent_roundtrip(tmp_path) -> None:
    audit = SafetyAudit(tmp_path / "events.jsonl")
    audit.record({"kind": "crisis", "source": "student", "hits": ["自杀"], "text": "甲"})
    audit.record({"kind": "crisis", "source": "model", "hits": ["想死"], "text": "乙"})

    events = audit.recent(10)
    assert [e["source"] for e in events] == ["model", "student"], "应倒序：最新在前"
    assert all(e.get("ts") and e.get("ts_iso") for e in events)


def test_recent_empty_and_limit(tmp_path) -> None:
    audit = SafetyAudit(tmp_path / "events.jsonl")
    assert audit.recent(10) == [], "文件不存在时返回空列表，不能抛"

    for i in range(5):
        audit.record({"kind": "crisis", "n": i})
    assert [e["n"] for e in audit.recent(2)] == [4, 3]
    assert audit.recent(0) == []
    assert audit.recent(-1) == []


def test_disabled_audit_writes_nothing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "safety_audit_enable", False)
    audit = SafetyAudit(tmp_path / "events.jsonl")
    assert audit.record({"kind": "crisis"}) is None
    assert not (tmp_path / "events.jsonl").exists()


def test_broken_line_does_not_break_the_log(tmp_path) -> None:
    """写一半被 kill 会留下半截行；它不能让整份日志不可读。"""
    path = tmp_path / "events.jsonl"
    audit = SafetyAudit(path)
    audit.record({"kind": "crisis", "n": 1})
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{半行损坏\n")
    audit.record({"kind": "crisis", "n": 2})

    assert [e["n"] for e in audit.recent(10)] == [2, 1]


def test_record_failure_is_silent(tmp_path) -> None:
    """审计写不进去也不能影响主链路（抛异常就等于把答疑搞挂了）。

    构造方式是「把路径指向一个目录」，让追加必然以 OSError 失败。
    """
    audit = SafetyAudit(tmp_path)
    assert audit.record({"kind": "crisis"}) is None
    assert audit.recent(10) == []


# ============================================================ REST 接口
def _client_with(tmp_path, monkeypatch, records: list[dict]) -> TestClient:
    audit = SafetyAudit(tmp_path / "events.jsonl")
    for item in records:
        audit.record(item)
    monkeypatch.setattr(routes, "safety_audit", audit)
    return TestClient(app)


def test_safety_events_endpoint(tmp_path, monkeypatch) -> None:
    client = _client_with(tmp_path, monkeypatch, [
        {"kind": "crisis", "source": "student", "hits": ["不想活"], "text": "我不想活了"},
    ])
    resp = client.get("/api/safety/events?limit=10")
    assert resp.status_code == 200

    body = resp.json()
    assert body["enabled"] is True
    assert body["total"] == 1
    assert body["events"][0]["hits"] == ["不想活"]
    assert body["events"][0]["text"] == "我不想活了"


def test_safety_events_endpoint_clamps_limit(tmp_path, monkeypatch) -> None:
    client = _client_with(tmp_path, monkeypatch, [])
    assert client.get("/api/safety/events?limit=0").status_code == 200
    assert client.get("/api/safety/events?limit=-5").status_code == 200
    assert client.get("/api/safety/events?limit=99999").json()["total"] == 0


# ============================================================ 教师端页面契约
def test_teacher_console_has_safety_panel() -> None:
    html = TEACHER_PAGE.read_text(encoding="utf-8")
    assert "/api/safety/events" in html, "教师端必须能取安全事件"
    assert "setInterval" in html, "危机事件需要轮询刷新，否则要手工刷新才看得到"


def test_teacher_console_escapes_event_text() -> None:
    """事件文本来自学生输入与模型输出——不转义直接拼 HTML 就是一个 XSS 入口。"""
    html = TEACHER_PAGE.read_text(encoding="utf-8")
    assert "const esc =" in html
    # 渲染事件的两处都必须过 esc()
    assert "esc(ev.text" in html
    assert "esc(ev.source" in html
