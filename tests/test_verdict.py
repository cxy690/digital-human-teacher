"""L0 裁决落盘（core/verdict.py）的行为测试。

守护三件事：
1. **判定口径**（:func:`judge`）—— 「什么算降级」只此一处，改口径必须同步改这里；
2. **落盘与聚合自洽** —— 写进去的东西能被聚合出来，且分组统计对得上；
3. **观测层不得影响主链路** —— 关闭时零开销、写失败不抛、损坏行不拖垮整份数据。

第 3 条尤其重要：它是「加分项功能」的底线，一旦观测层能把课堂搞挂，
这个功能就比没有更糟。
"""
from __future__ import annotations

import json

import pytest

from config.settings import settings
from core.state.models import AIResponse
from core.verdict import (
    VerdictStore,
    infer_kind,
    judge,
    record_turn_verdict,
    summarize,
)


# ---------------------------------------------------------------------- 工具
class _Step:
    """替身：``core.trace.Step`` 的最小可辨识子集。"""

    def __init__(self, label: str, *, ms: float = 1.0, ok: bool = True) -> None:
        self.label = label
        self.file = "core/supervisor.py"
        self.func = label
        self.desc = ""
        self.ms = ms
        self.ok = ok


class _Session:
    session_id = "sess-1"
    student_id = "stu-1"
    subject = "数学"
    teacher = "math"
    course_active = True


class _Turn:
    text = "这道二次函数题怎么做？"


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """把落盘目录指到 tmp，并确保开关是开的（不污染真实 data/）。"""
    monkeypatch.setattr(settings, "verdict_dir", str(tmp_path))
    monkeypatch.setattr(settings, "verdict_enable", True)
    monkeypatch.setattr(settings, "verdict_keep_steps", True)
    return VerdictStore(tmp_path)


# ---------------------------------------------------------------------- 判定
def test_clean_turn_is_pass() -> None:
    """正常产物 = pass，且不带任何理由。"""
    resp = AIResponse(speech="先移项再合并同类项，得到 x=5")
    verdict, reasons = judge(response=resp)
    assert verdict == "pass"
    assert reasons == []


def test_json_degraded_counts_as_degraded() -> None:
    """JSON 降级（板书整块丢弃）必须被记成 degraded —— 这是最常见的静默劣化。"""
    resp = AIResponse(speech="抱歉，这道题我这边没组织好", meta={"json_degraded": True})
    verdict, reasons = judge(response=resp)
    assert verdict == "degraded"
    assert any("JSON 降级" in r for r in reasons)


def test_quality_deduction_is_recorded_with_reason() -> None:
    """质检扣分要连分数与问题一起进理由，否则聚合出来只剩「有问题」说不清是什么。"""
    resp = AIResponse(
        speech="讲解",
        meta={"quality": {"passed": True, "score": 85, "issues": ["话术偏长"]}},
    )
    verdict, reasons = judge(response=resp)
    assert verdict == "degraded"
    assert any("85" in r for r in reasons)
    assert any("话术偏长" in r for r in reasons)


def test_exception_outranks_everything() -> None:
    """抛异常一律 failed —— 哪怕同时还有别的产物。"""
    resp = AIResponse(speech="讲了一半")
    verdict, reasons = judge(response=resp, error=TimeoutError("t"))
    assert verdict == "failed"
    assert any("TimeoutError" in r for r in reasons)


def test_empty_reply_is_failed() -> None:
    """话术与板书都空 = 没交付任何东西。"""
    verdict, _ = judge(response=AIResponse(speech="", board_ops=[]))
    assert verdict == "failed"


def test_no_response_is_degraded_not_failed() -> None:
    """无响应产物记 degraded —— 有些分支（控制指令）本就不产出回复，不该算失败。"""
    verdict, reasons = judge(response=None)
    assert verdict == "degraded"
    assert "无响应产物" in reasons


def test_failed_step_degrades_the_turn() -> None:
    """调用链里有步骤抛异常（但整体没崩）也要算 degraded。"""
    resp = AIResponse(speech="讲解")
    verdict, reasons = judge(response=resp, steps=[_Step("X.f", ok=False)])
    assert verdict == "degraded"
    assert any("步骤异常" in r for r in reasons)


# ---------------------------------------------------------------------- 场景推断
@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Supervisor._turn_teach", "teach"),
        ("Supervisor._turn_crisis", "crisis"),
        ("Supervisor._play_slides", "courseware"),
        ("Supervisor._turn_chitchat", "chitchat"),
        ("Supervisor.on_quiz_request", "quiz"),
        ("Supervisor.something_else", "other"),
    ],
)
def test_infer_kind(label: str, expected: str) -> None:
    """场景名从调用链推得出来（这样各分支不用自报，避免改一圈调用点）。"""
    assert infer_kind([_Step(label)]) == expected


def test_crisis_outranks_teach_when_both_present() -> None:
    """同一条链里同时出现多种场景时，按 :data:`_KIND_BY_STEP` 的优先级取。"""
    assert infer_kind([_Step("Supervisor._turn_teach"), _Step("Supervisor._turn_crisis")]) == "crisis"


# ---------------------------------------------------------------------- 落盘
def test_record_roundtrip_and_fields(store: VerdictStore) -> None:
    """写进去的记录，关键字段都取得到。"""
    entry = record_turn_verdict(
        session=_Session(), turn=_Turn(),
        response=AIResponse(speech="讲解内容", board_ops=[]),
        elapsed_ms=1234.5, kind="teach",
    )
    assert entry is not None
    assert entry["verdict"] == "pass"
    assert entry["subject"] == "数学"
    assert entry["elapsed_ms"] == 1234.5

    path = store.path_for(entry["day"])
    assert path.exists(), "必须真的落到当天那个文件里"
    line = json.loads(path.read_text(encoding="utf-8").strip())
    assert line["session_id"] == "sess-1"
    assert line["verdict"] == "pass"


def test_disabled_writes_nothing(tmp_path, monkeypatch) -> None:
    """关闭时零开销、零落盘 —— 配了开关就得真的能关掉。"""
    monkeypatch.setattr(settings, "verdict_dir", str(tmp_path))
    monkeypatch.setattr(settings, "verdict_enable", False)
    out = record_turn_verdict(
        session=_Session(), turn=_Turn(), response=AIResponse(speech="x"), elapsed_ms=1.0,
    )
    assert out is None
    assert list(tmp_path.glob("*.jsonl")) == []


def test_broken_line_does_not_poison_the_day(store: VerdictStore) -> None:
    """单行损坏（写一半被 kill）只跳过该行，同一天其余记录照读。"""
    store.record({"day": "2026-10-12", "verdict": "pass", "kind": "teach", "subject": "数学"})
    path = store.path_for("2026-10-12")
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"day": "2026-10-12", "verdict": "pas\n')   # 截断的半行
    store.record({"day": "2026-10-12", "verdict": "failed", "kind": "teach", "subject": "数学"})

    win = store.summarize(days=7)
    assert win.total == 2, "损坏行不该被计入，也不该让另外两行丢失"


def test_write_failure_is_silent(tmp_path, monkeypatch) -> None:
    """目录不可写时只能返回 None，绝不能把异常抛回主链路。"""
    monkeypatch.setattr(settings, "verdict_dir", str(tmp_path))
    monkeypatch.setattr(settings, "verdict_enable", True)

    store = VerdictStore(tmp_path)
    monkeypatch.setattr(
        type(store), "path_for",
        lambda self, day: tmp_path / "no_such_dir" / f"verdict-{day}.jsonl",
    )
    assert store.record({"day": "2026-10-12", "verdict": "pass"}) is None


# ---------------------------------------------------------------------- 接线（端到端）
async def test_real_turn_writes_a_verdict_record(monkeypatch, tmp_path) -> None:
    """端到端：走**真实 Supervisor** 跑一轮，裁决必须落到文件里。

    这条测试是「接线对不对」的唯一证据 —— 上面那些单测只能证明模块本身没问题，
    证明不了 ``handle_turn`` 真的调了它（本项目已经有过一次反面教材：
    经验层写好了、测试也在，但就是没接进答疑链路，一直没人发现）。
    """
    # 必须用**生产链路那个** Supervisor（core.harness / core.loop 都导入它）：
    # 项目里另有一份 agents/supervisor/agent.py 的同名类，只有测试在用，两者已经分叉
    # （core 那份没有 _schedule_skill_reflection，agents 那份没有 _turn_realtime_voice）。
    # 用错那份就会「测试通过但生产没接线」——本项目已有过这种反面教材。
    from core.state.models import StudentTurn
    from core.state.session import SessionState
    from agents.supervisor import Supervisor

    monkeypatch.setattr(settings, "verdict_dir", str(tmp_path))
    monkeypatch.setattr(settings, "verdict_enable", True)

    state = SessionState(session_id="v-e2e", student_id="s1")
    state.subject = "数学"
    state.scratch["subject"] = "数学"
    supervisor = Supervisor(state)
    await supervisor.on_start()

    await supervisor.handle_turn(StudentTurn(text="老师，这道方程题怎么解？"))

    files = list(tmp_path.glob("verdict-*.jsonl"))
    assert files, "handle_turn 必须落下一条裁决"
    rec = json.loads(files[0].read_text(encoding="utf-8").strip().splitlines()[-1])
    assert rec["session_id"] == "v-e2e"
    assert rec["verdict"] in ("pass", "degraded", "failed")
    assert rec["subject"] == "数学"
    assert rec["elapsed_ms"] > 0, "耗时必须真的量到"
    assert rec["kind"] != "other", "场景应能从调用链推出来（推不出说明 trace 没接上）"


async def test_real_turn_records_elapsed_and_steps(monkeypatch, tmp_path) -> None:
    """落盘记录里要带调用链，且每步耗时非负 —— 这是「卡在哪一步」的数据来源。"""
    # 必须用**生产链路那个** Supervisor（core.harness / core.loop 都导入它）：
    # 项目里另有一份 agents/supervisor/agent.py 的同名类，只有测试在用，两者已经分叉
    # （core 那份没有 _schedule_skill_reflection，agents 那份没有 _turn_realtime_voice）。
    # 用错那份就会「测试通过但生产没接线」——本项目已有过这种反面教材。
    from core.state.models import StudentTurn
    from core.state.session import SessionState
    from agents.supervisor import Supervisor

    monkeypatch.setattr(settings, "verdict_dir", str(tmp_path))
    monkeypatch.setattr(settings, "verdict_enable", True)
    monkeypatch.setattr(settings, "verdict_keep_steps", True)

    state = SessionState(session_id="v-e2e2", student_id="s2")
    state.subject = "数学"
    supervisor = Supervisor(state)
    await supervisor.handle_turn(StudentTurn(text="什么是数轴？"))

    rec = json.loads(
        next(tmp_path.glob("verdict-*.jsonl")).read_text(encoding="utf-8").strip().splitlines()[-1]
    )
    assert rec["steps"], "开着 VERDICT_KEEP_STEPS 就该存下调用链"
    for step in rec["steps"]:
        assert step["ms"] >= 0
        assert "ok" in step


def test_disabled_switch_also_silences_real_turns(monkeypatch, tmp_path) -> None:
    """开关关掉后，真实回合也不该落盘（能关得掉才算开关）。"""
    import asyncio

    # 必须用**生产链路那个** Supervisor（core.harness / core.loop 都导入它）：
    # 项目里另有一份 agents/supervisor/agent.py 的同名类，只有测试在用，两者已经分叉
    # （core 那份没有 _schedule_skill_reflection，agents 那份没有 _turn_realtime_voice）。
    # 用错那份就会「测试通过但生产没接线」——本项目已有过这种反面教材。
    from core.state.models import StudentTurn
    from core.state.session import SessionState
    from agents.supervisor import Supervisor

    monkeypatch.setattr(settings, "verdict_dir", str(tmp_path))
    monkeypatch.setattr(settings, "verdict_enable", False)

    state = SessionState(session_id="v-off", student_id="s3")
    state.subject = "数学"
    supervisor = Supervisor(state)
    asyncio.run(supervisor.handle_turn(StudentTurn(text="老师好")))
    assert list(tmp_path.glob("*.jsonl")) == []


# ---------------------------------------------------------------------- 聚合
def test_summarize_groups_and_rates(store: VerdictStore) -> None:
    """聚合要能算出受影响率、分组最差排序、最慢步骤与主要理由。"""
    today = store.days(1)[-1]
    store.record({"day": today, "verdict": "pass", "kind": "teach", "subject": "数学",
                  "elapsed_ms": 1000, "quality": {"score": 100, "issues": []}})
    store.record({"day": today, "verdict": "degraded", "kind": "teach", "subject": "数学",
                  "elapsed_ms": 3000, "reasons": ["JSON 降级（板书整块丢弃）"],
                  "quality": {"score": 80, "issues": ["话术偏长"]},
                  "steps": [{"file": "agents/teacher/agent.py", "func": "TeacherAgent.answer",
                             "desc": "", "ms": 2500, "ok": True}]})
    store.record({"day": today, "verdict": "failed", "kind": "chitchat", "subject": "物理",
                  "elapsed_ms": 9000, "reasons": ["异常：TimeoutError"]})

    out = store.summarize(days=7).as_dict()

    assert out["total"] == 3
    assert out["verdicts"] == {"pass": 1, "degraded": 1, "failed": 1}
    # 受影响率 = (degraded + failed) / total
    assert out["rates_pct"]["affected"] == pytest.approx(66.7, abs=0.1)
    # 质检均分只统计有分数的回合：(100 + 80) / 2
    assert out["quality"]["avg_score"] == 90.0
    # 物理那一条全废 -> 受影响率最高，必须排在最前面（最差的先看见）
    assert next(iter(out["by_subject"])) == "物理"
    assert out["slow_steps"][0]["name"] == "TeacherAgent.answer"
    assert out["top_reasons"][0][1] == 1


def test_summarize_empty_window_is_safe(store: VerdictStore) -> None:
    """没有任何数据时不能除零、不能抛 —— 新部署第一天就该能跑。"""
    out = summarize(days=7)
    assert out["total"] == 0
    assert out["rates_pct"]["failed"] == 0.0
    assert out["quality"]["avg_score"] is None
    assert out["latency_ms"]["p50"] is None
