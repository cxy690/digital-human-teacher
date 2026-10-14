"""技能管理平台（仅开发人员）的 REST 接口测试。

钉住五件事：
1. 门禁 = **登录账号角色**：未登录 401、非 admin 403、``AUTH_ENABLE=false`` 403
   （没有身份体系就没有「谁能进管理面」的判定依据，宁可不可用也不裸奔）；
2. 清单 / 文件读取：admin 可列出技能与文件、读取 SKILL.md 与 references 原文；
3. 保存校验：SKILL.md 结构不完整（缺必填字段 / 缺必备小节）拒收并列出问题；
   合法文本保存成功、写前留 ``.bak`` 手滑保护；
4. 路径安全：只允许 ``SKILL.md`` 与 ``references/*.md``（拒目录穿越）；
5. 自我更新看板：``/api/admin/skills/loop`` 把 append-only JSONL 还原成更新事件 ——
   相邻快照差分推导动作、同批复盘合并、前后对比、结果三态、坏行记成失败事件。

隔离：``SKILLS_DIR`` 切到 tmp_path（绝不触碰真实 ``.agents/skills``）；
看板数据源（``SKILL_EXPERIENCE_DIR`` / ``AGENT_SKILL_USAGE_DIR``）同样切到 tmp_path；
数据库用临时 SQLite（同 test_auth 的约定）。
"""
from __future__ import annotations

import json
import time

import pytest

from config.settings import settings

# 一份「合法且完整」的 SKILL.md（frontmatter 必填 + 必备小节全齐）
SAMPLE = """---
subject_key: "math"
subject_name: "数学"
skill_id: "teacher.math.demo"
display_name: "演示数学老师"
persona: "严谨演示"
tone: "平稳"
depth: "standard"
speech_rate: 1.0
teaching_style: "plain"
style_persona: "简洁清楚。"
---
## 说话风格
简洁清楚，一次说一件事。
## 讲课方式
先例题后总结。
## 人格与教学
你是一位严谨的数学老师。
## 开场方式
先问今天想学什么。
## 追问方式
追一步为什么。
## 鼓励方式
肯定进步。
## 表扬方式
具体指出做对的地方。
## 纠错方式
指出错在哪一步。
## 学生说「我不会」时
拆小步骤带一遍。
## 学生连续答错时
换例子再讲。
## 白板组织
左边题干右边步骤。
## 语气与节奏
不疾不徐。
## 进场问候
同学你好，{where}开始啦。
"""


@pytest.fixture()
def skills_dir(tmp_path, monkeypatch):
    """把技能根目录切到临时目录（绝不触碰真实 .agents/skills）。"""
    from config.teacher_skills import md as skills_md

    root = tmp_path / ".agents" / "skills"
    (root / "math").mkdir(parents=True)
    (root / "math" / "SKILL.md").write_text(SAMPLE, encoding="utf-8")
    (root / "math" / "references").mkdir()
    (root / "math" / "references" / "tricks.md").write_text("# 技巧\n演示。\n", encoding="utf-8")
    monkeypatch.setattr(skills_md, "SKILLS_DIR", root)
    skills_md.reload_docs()
    yield root
    skills_md.reload_docs()


@pytest.fixture()
def sqlite_db(tmp_path):
    """把数据层切到临时 SQLite 并建表（同 test_auth 的约定）。"""
    from db import base as db_base

    db_base.set_database_url(f"sqlite:///{tmp_path / 'skill_admin.db'}")
    db_base.init_db()
    yield
    db_base.set_database_url(f"sqlite:///{tmp_path / 'skill_admin_after.db'}")


def _client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from server.routes import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture()
def tokens(sqlite_db, monkeypatch):
    """预置演示账号并登录拿到 admin / student 两枚 token。"""
    monkeypatch.setattr(settings, "auth_enable", True)
    from core.auth.service import seed_demo_users

    seed_demo_users()
    client = _client()
    admin = client.post("/api/auth/login", json={"username": "admin", "password": "123456"}).json()
    stu = client.post("/api/auth/login", json={"username": "student", "password": "123456"}).json()
    return {"admin": admin["token"], "student": stu["token"]}


def _h(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------- 门禁
def test_requires_developer_role(skills_dir, tokens) -> None:
    """未登录 401 / 非 admin 403 / admin 放行。"""
    client = _client()
    assert client.get("/api/admin/skills").status_code == 401
    assert client.get("/api/admin/skills", headers=_h(tokens["student"])).status_code == 403
    assert client.get("/api/admin/skills", headers=_h(tokens["admin"])).status_code == 200


def test_rejected_when_auth_disabled(skills_dir, tokens, monkeypatch) -> None:
    """AUTH_ENABLE=false：身份体系没开，管理面直接拒绝（不能裸奔）。"""
    monkeypatch.setattr(settings, "auth_enable", False)
    client = _client()
    resp = client.get("/api/admin/skills", headers=_h(tokens["admin"]))
    assert resp.status_code == 403
    assert "AUTH_ENABLE" in resp.json()["detail"]


# ---------------------------------------------------------------- 列表与读取
def test_list_and_read(skills_dir, tokens) -> None:
    client = _client()
    data = client.get("/api/admin/skills", headers=_h(tokens["admin"])).json()
    assert data["ok"] and data["operator"] == "admin"
    math = next(s for s in data["skills"] if s["key"] == "math")
    assert math["complete"] is True
    assert math["display_name"] == "演示数学老师"
    assert "references/tricks.md" in math["files"]

    got = client.get("/api/admin/skills/math/file", headers=_h(tokens["admin"])).json()
    assert got["text"].startswith("---") and "演示数学老师" in got["text"]
    ref = client.get(
        "/api/admin/skills/math/file",
        params={"path": "references/tricks.md"},
        headers=_h(tokens["admin"]),
    ).json()
    assert "技巧" in ref["text"]


def test_reject_path_traversal(skills_dir, tokens) -> None:
    """只允许 SKILL.md 与 references/*.md —— 目录穿越 / 非法后缀一律 400。"""
    client = _client()
    for bad in ("../x.md", "references/../SKILL.md", "/etc/passwd", "references/a.txt", "a.py"):
        resp = client.get(
            "/api/admin/skills/math/file", params={"path": bad}, headers=_h(tokens["admin"]))
        assert resp.status_code == 400, bad


# ---------------------------------------------------------------- 保存
def test_save_valid_updates_and_backs_up(skills_dir, tokens) -> None:
    client = _client()
    new_text = SAMPLE.replace("演示数学老师", "演示数学老师 v2")
    resp = client.put(
        "/api/admin/skills/math/file",
        headers=_h(tokens["admin"]),
        json={"path": "SKILL.md", "text": new_text},
    )
    assert resp.status_code == 200 and resp.json()["ok"]
    assert (skills_dir / "math" / "SKILL.md").read_text(encoding="utf-8") == new_text
    assert (skills_dir / "math" / "SKILL.md.bak").exists(), "写前应留 .bak 手滑保护"


def test_save_rejects_incomplete_md(skills_dir, tokens) -> None:
    """缺必填字段 / 缺必备小节的半成品拒收并列出问题；拒收不落盘。"""
    client = _client()
    broken = SAMPLE.replace('display_name: "演示数学老师"\n', "")
    broken = broken.replace("## 白板组织\n左边题干右边步骤。\n", "")
    resp = client.put(
        "/api/admin/skills/math/file",
        headers=_h(tokens["admin"]),
        json={"path": "SKILL.md", "text": broken},
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "display_name" in detail and "白板组织" in detail
    assert (skills_dir / "math" / "SKILL.md").read_text(encoding="utf-8") == SAMPLE


# ---------------------------------------------------------------- 自我更新看板
def _loop_dirs(tmp_path, monkeypatch):
    """把看板的数据源切到临时目录（绝不读真实 data/teacher_experience）。"""
    exp = tmp_path / "teacher_experience"
    use = tmp_path / "skill_usage"
    exp.mkdir()
    use.mkdir()
    monkeypatch.setattr(settings, "skill_experience_dir", str(exp))
    monkeypatch.setattr(settings, "agent_skill_usage_dir", str(use))
    return exp, use


def _exp_row(item_id: str, text: str, *, ts: float, hits: int = 1, status: str = "active",
             pulls: int = 0, wins: int = 0, gain: float = 0.0) -> str:
    """一行经验快照（字段与 ExperienceItem.to_dict 一致）。"""
    return json.dumps({
        "id": item_id, "teacher_key": "math", "kind": "pitfall", "text": text,
        "hits": hits, "confidence": 0.9, "status": status,
        "created_at": ts, "updated_at": ts, "sources": ["数学｜例题"],
        "pulls": pulls, "wins": wins, "last_gain": gain,
    }, ensure_ascii=False)


def _usage_row(name: str, *, ts: float, pulls: int, wins: int, gain: float,
               status: str = "active") -> str:
    return json.dumps({
        "name": name, "pulls": pulls, "wins": wins, "last_gain": gain,
        "status": status, "updated_at": ts,
    }, ensure_ascii=False)


def test_loop_requires_developer_role(skills_dir, tokens) -> None:
    """看板与技能清单同一门禁：未登录 401 / 非 admin 403 / admin 放行。"""
    client = _client()
    assert client.get("/api/admin/skills/loop").status_code == 401
    assert client.get("/api/admin/skills/loop", headers=_h(tokens["student"])).status_code == 403
    assert client.get("/api/admin/skills/loop", headers=_h(tokens["admin"])).status_code == 200


def test_loop_rebuilds_experience_timeline(skills_dir, tokens, tmp_path, monkeypatch) -> None:
    """经验 JSONL -> 更新事件：学习合并、命中强化、战绩回写、停用、坏行。"""
    exp, _use = _loop_dirs(tmp_path, monkeypatch)
    base = time.time()
    rows = [
        # 同一批复盘学到两条（时间相隔 0.5s < 合并窗口）→ 应合成一次更新
        _exp_row("exp-a", "学生常把移项符号搞错。", ts=base - 300.0),
        _exp_row("exp-b", "讲浮力时用「船为什么会漂」类比很有效。", ts=base - 299.5),
        # 复盘再次命中已有经验 → 命中次数 +1
        _exp_row("exp-a", "学生常把移项符号搞错。", ts=base - 200.0, hits=2),
        # 回合收口按裁决回写收益
        _exp_row("exp-a", "学生常把移项符号搞错。", ts=base - 100.0, hits=2, pulls=1, wins=1, gain=1.0),
        # 人工停用（止损）
        _exp_row("exp-a", "学生常把移项符号搞错。", ts=base - 50.0, hits=2, status="retired",
                 pulls=1, wins=1, gain=1.0),
        "{坏行",
    ]
    (exp / "math.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")

    client = _client()
    data = client.get("/api/admin/skills/loop", headers=_h(tokens["admin"])).json()
    assert data["ok"] and data["operator"] == "admin"

    by_action: dict[str, list] = {}
    for event in data["events"]:
        by_action.setdefault(event["action"], []).append(event)
    assert set(by_action) == {"learn", "reinforce", "outcome", "retire", "corrupt"}

    learn = by_action["learn"][0]
    assert learn["result"] == "ok" and learn["writes"] == 2 and len(learn["items"]) == 2
    assert learn["items"][0]["created"] is True
    assert learn["items"][0]["changes"][0]["before"] is None, "新增条目没有前值"

    reinforce = by_action["reinforce"][0]
    hits = next(c for c in reinforce["items"][0]["changes"] if c["field"] == "hits")
    assert (hits["before"], hits["after"]) == (1, 2)

    outcome = by_action["outcome"][0]
    assert outcome["result"] == "ok"
    fields = {c["field"]: (c["before"], c["after"]) for c in outcome["items"][0]["changes"]}
    assert fields["pulls"] == (0, 1) and fields["wins"] == (0, 1)

    retire = by_action["retire"][0]
    assert retire["result"] == "warn"
    assert retire["items"][0]["changes"][0]["after"] == "已停用"

    corrupt = by_action["corrupt"][0]
    assert corrupt["result"] == "fail" and corrupt["line_from"] == 6

    # 最新在最前；统计按全量算（不受 limit 影响）
    assert data["events"][0]["ts"] >= data["events"][-1]["ts"]
    assert data["stats"]["experience_items"] == 2
    assert data["stats"]["experience_active"] == 1        # exp-a 已停用
    assert data["stats"]["fail"] == 1
    assert [s["source"] for s in data["sources"]] == ["math"]


def test_loop_rebuilds_usage_timeline(skills_dir, tokens, tmp_path, monkeypatch) -> None:
    """Agent 技能战绩 JSONL -> 调用记账（成功 / 没帮上忙）+ 停用。"""
    _exp, use = _loop_dirs(tmp_path, monkeypatch)
    base = time.time()
    rows = [
        _usage_row("praise", ts=base - 100.0, pulls=1, wins=1, gain=1.0),
        _usage_row("praise", ts=base - 50.0, pulls=2, wins=2, gain=0.0),
        _usage_row("praise", ts=base - 20.0, pulls=2, wins=2, gain=0.0, status="disabled"),
    ]
    (use / "praise.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")

    client = _client()
    data = client.get("/api/admin/skills/loop", headers=_h(tokens["admin"])).json()
    actions = [e["action"] for e in data["events"]]
    assert actions == ["disable", "usage", "usage"]      # 最新在前
    assert data["events"][1]["result"] == "warn", "裁决未通过 = 警告（没帮上忙）"
    assert data["events"][0]["items"][0]["changes"][0]["after"] == "已停用"
    assert data["stats"]["usage_skills"] == 1 and data["stats"]["usage_active"] == 0


def test_loop_config_and_limit(skills_dir, tokens, tmp_path, monkeypatch) -> None:
    """看板要带上触发条件（配置阈值）并遵守 limit；空数据也要能开。"""
    _loop_dirs(tmp_path, monkeypatch)
    client = _client()

    empty = client.get("/api/admin/skills/loop", headers=_h(tokens["admin"])).json()
    assert empty["events"] == [] and empty["total_events"] == 0

    data = client.get("/api/admin/skills/loop", params={"limit": 1}, headers=_h(tokens["admin"])).json()
    assert data["config"]["self_update_enable"] is settings.skill_self_update_enable
    assert "reflect_min_confidence" in data["config"]
    assert "agent_skill_min_pulls" in data["config"]
    assert len(data["events"]) <= 1


def test_references_editable(skills_dir, tokens) -> None:
    """references 支持编辑（同一套接口，不做 SKILL.md 的结构校验）。"""
    client = _client()
    resp = client.put(
        "/api/admin/skills/math/file",
        headers=_h(tokens["admin"]),
        json={"path": "references/tricks.md", "text": "# 新技巧\n改过了。\n"},
    )
    assert resp.status_code == 200
    saved = (skills_dir / "math" / "references" / "tricks.md").read_text(encoding="utf-8")
    assert "改过了" in saved
