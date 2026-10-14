"""L3 失败聚类与根因分析的测试。

钉住三件事：
1. **归一化对了**：同一类缺陷不该被拆成几十个只差一个数字的簇，而不同的病不该被并成一个；
2. **聚类是有信息量的**：每个簇能看出「卡在哪一步、偏向哪个场景/学科、伴随什么」——
   这正是它比四张独立频次表强的地方；
3. **根因分析不炸**：关掉时直接说不做、模型给坏 JSON 时降级、绝不抛异常
   （排障工具本身报错，会让人以为是系统坏了）。
"""
from __future__ import annotations

from config.settings import settings
from core.failure_cluster import (
    analyze_cluster,
    cluster,
    describe,
    failed_step,
    primary_reason,
    reason_key,
    report,
)


# ---------------------------------------------------------------- 夹具
def _rec(
    *,
    reasons: list[str],
    kind: str = "teach",
    subject: str = "数学",
    verdict: str = "degraded",
    steps: list[dict] | None = None,
    question: str = "解方程 3x+5=20",
    day: str = "2026-10-12",
    teacher: str = "math",
) -> dict:
    return {
        "day": day,
        "verdict": verdict,
        "reasons": reasons,
        "kind": kind,
        "subject": subject,
        "teacher": teacher,
        "question": question,
        "session_id": "s1",
        "speech_len": 120,
        "board_ops": 0,
        "elapsed_ms": 8200.0,
        "steps": steps or [],
    }


def _bad_step(func: str = "generate", file: str = "core/langgraph_flow.py") -> list[dict]:
    return [
        {"file": file, "func": "retrieve", "ok": True, "ms": 12},
        {"file": file, "func": func, "ok": False, "ms": 5000},
    ]


# ---------------------------------------------------------------- 归一化
def test_reason_key_merges_scores_but_keeps_kinds() -> None:
    """分数要抹掉（85 和 92 是同一种病），但具体的质检问题文本要保留（是两种病）。"""
    assert reason_key("质检扣分 85")[0] == reason_key("质检扣分 92")[0] == "quality_deduct"
    assert reason_key("质检问题：话术太长")[0] == reason_key("质检问题：没有板书")[0]
    assert reason_key("质检问题：话术太长")[1] != reason_key("质检问题：没有板书")[1]


def test_primary_reason_prefers_the_root_one() -> None:
    """一条回合常有多个原因，取**最接近根子**的那个当主签名。"""
    # JSON 降级往往伴随质检扣分：主签名该是前者（结构坏了才导致内容不过关）
    key, _ = primary_reason(["质检扣分 85", "JSON 降级（板书整块丢弃）"])
    assert key == "json_degraded"
    # 抛异常比什么都优先
    key, _ = primary_reason(["JSON 降级（板书整块丢弃）", "异常：ConnectionError"])
    assert key == "exception"


def test_failed_step_only_counts_errors_not_slow_steps() -> None:
    """慢 ≠ 坏：慢属于性能问题（``Window.slow_steps``），不该混进失败簇。"""
    assert failed_step(_rec(reasons=["x"], steps=_bad_step())) == "core/langgraph_flow.py:generate"
    slow_only = [{"file": "a.py", "func": "slow", "ok": True, "ms": 9999}]
    assert failed_step(_rec(reasons=["x"], steps=slow_only)) == ""


# ---------------------------------------------------------------- 聚类
def test_cluster_separates_by_step() -> None:
    """同样是「JSON 降级」，卡在生成节点还是收尾节点，修法完全不同 —— 必须分开。"""
    records = [
        _rec(reasons=["JSON 降级（板书整块丢弃）"], steps=_bad_step("generate")),
        _rec(reasons=["JSON 降级（板书整块丢弃）"], steps=_bad_step("generate")),
        _rec(reasons=["JSON 降级（板书整块丢弃）"], steps=_bad_step("solve", "core/postprocess.py")),
    ]
    items = cluster(records)
    assert len(items) == 2, "带步骤的签名应把两类拆开"
    top = items[0]
    assert top.count == 2
    assert top.step == "core/langgraph_flow.py:generate"


def test_cluster_keeps_samples_and_distributions() -> None:
    """簇要有「现场」：原问题、场景/学科分布、伴随现象 —— 否则人还是得自己去翻日志。"""
    records = [
        _rec(reasons=["JSON 降级（板书整块丢弃）", "质检扣分 80"], kind="teach", subject="数学",
             question="帮我讲讲二次函数"),
        _rec(reasons=["JSON 降级（板书整块丢弃）"], kind="teach", subject="物理",
             question="讲讲牛顿第一定律"),
        _rec(reasons=["JSON 降级（板书整块丢弃）"], kind="courseware", subject="化学",
             question="讲这一页"),
    ]
    top = cluster(records)[0]
    assert top.count == 3
    assert top.by_kind["teach"] == 2 and top.by_kind["courseware"] == 1
    assert top.by_subject["数学"] == 1
    assert top.co_reasons, "「JSON 降级」伴随「质检扣分」这条线索要留下来"
    assert [s["question"] for s in top.samples][:1] == ["帮我讲讲二次函数"]
    assert top.first_seen and top.last_seen


def test_cluster_ignores_pass_records() -> None:
    records = [
        _rec(reasons=[], verdict="pass"),
        _rec(reasons=["JSON 降级（板书整块丢弃）"], verdict="degraded"),
    ]
    items = cluster(records)
    assert len(items) == 1 and items[0].count == 1
    assert sum(c.count for c in items) == 1, "干净的回合不该进任何簇"


def test_report_reads_verdict_files(monkeypatch, tmp_path) -> None:
    """端到端：从 L0 落盘的裁决文件读起，出聚类报告。"""
    from core.verdict import VerdictStore

    monkeypatch.setattr(settings, "verdict_dir", str(tmp_path))
    store = VerdictStore(tmp_path)
    for i in range(3):
        store.record(_rec(reasons=["JSON 降级（板书整块丢弃）"], steps=_bad_step()))
    store.record(_rec(reasons=[], verdict="pass"))

    data = report(days=7)
    assert data["records"] == 4
    assert data["failures"] == 3
    assert data["clusters"][0]["count"] == 3
    assert data["clusters"][0]["share_pct"] == 100.0
    assert data["clusters"][0]["step"] == "core/langgraph_flow.py:generate"


def test_describe_renders_the_cluster(monkeypatch, tmp_path) -> None:
    from core.verdict import VerdictStore

    monkeypatch.setattr(settings, "verdict_dir", str(tmp_path))
    store = VerdictStore(tmp_path)
    store.record(_rec(reasons=["JSON 降级（板书整块丢弃）"], steps=_bad_step()))

    text = describe(days=7)
    assert "JSON 降级" in text
    assert "core/langgraph_flow.py:generate" in text, "要能看出卡在哪一步"
    assert "解方程" in text, "要带上真实样本"


# ---------------------------------------------------------------- 根因分析
class _StubLLM:
    """返回固定 JSON 的桩模型（顺带记录收到的 prompt，便于断言喂了证据）。"""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.seen: list[str] = []

    async def chat(self, messages, **kwargs):  # noqa: ANN001, ANN003 - 与真实签名对齐
        self.seen.append("\n".join(str(m.get("content", "")) for m in messages))
        return self.payload


async def test_analyze_returns_structured_conclusion() -> None:
    llm = _StubLLM(
        '{"root_cause": "解答类契约的 token 预算不足，长讲解被截断",'
        ' "evidence": ["话术断在半句", "板书为 0"],'
        ' "suggestions": ["把 ANSWER_MAX_TOKENS 提到 1800"],'
        ' "confidence": 0.8}'
    )
    data = {
        "label": "JSON 降级（板书整块丢弃）",
        "count": 3,
        "share_pct": 60.0,
        "step": "core/langgraph_flow.py:generate",
        "by_kind": {"teach": 3},
        "samples": [{"question": "帮我讲讲二次函数", "kind": "teach",
                     "reasons": ["JSON 降级"], "speech_len": 0, "board_ops": 0,
                     "elapsed_ms": 8000}],
    }
    result = await analyze_cluster(data, llm=llm)
    assert result["ok"] is True
    assert "token" in result["root_cause"]
    assert result["suggestions"] and result["confidence"] == 0.8
    joined = llm.seen[0]
    assert "帮我讲讲二次函数" in joined, "样本必须喂给模型，否则结论是瞎猜"
    assert "core/langgraph_flow.py:generate" in joined, "卡在哪一步也要喂进去"


async def test_analyze_survives_bad_model_output() -> None:
    """模型给的不是 JSON：返回错误说明，**绝不抛** —— 排障工具报错会让人以为系统坏了。"""
    result = await analyze_cluster({"label": "x", "count": 1}, llm=_StubLLM("我觉得可能是提示词的问题"))
    assert result["ok"] is False and "JSON" in result["error"]


async def test_analyze_honours_switch(monkeypatch) -> None:
    monkeypatch.setattr(settings, "failure_analysis_enable", False)
    result = await analyze_cluster({"label": "x", "count": 1}, llm=_StubLLM("{}"))
    assert result["ok"] is False and "FAILURE_ANALYSIS_ENABLE" in result["error"]


# ---------------------------------------------------------------- L3 产出的修复
async def test_model_failure_degrades_but_stays_visible(monkeypatch) -> None:
    """L3 挖出来的那个缺陷已修：模型调用失败时**兜底，但不把问题藏起来**。

    修之前：``turn_graph.teach`` 没有兜底，上游 LLM 一次 HTTP 抖动（429/5xx）
    就让整个回合失败 —— 学生那边老师干脆不说话。根因分析给出的三条建议里，
    这条是「加降级分支」，本条测试就是它的验收。

    修之后必须**同时**成立两件事：

    1. 学生一定听得到东西（兜底话术）—— 本项目对答疑链路的硬要求；
    2. 观测层一定看得见这次失败（``meta["llm_error"]`` → :func:`judge` 判为降级）。

    只做第 1 件就成了「把故障藏起来」，比不兜底更糟：学生听到兜底、统计里全是 pass，
    L0/L3 从此再也发现不了这类上游故障。
    """
    from agents.teacher.agent import TeacherAgent
    from core.state.models import StudentTurn
    from core.state.session import SessionState
    from core.verdict import judge
    from models.base import BaseLLMClient

    class _BoomLLM(BaseLLMClient):
        """一调就炸的模型：模拟供应商 5xx / 网络抖动。"""

        async def chat(self, messages, **kwargs):  # noqa: ANN001, ANN003 - 与真实签名对齐
            raise RuntimeError("HTTPStatusError: 503 from upstream")

    monkeypatch.setattr(settings, "agent_react_enable", False)   # 走普通答疑路径，确定性
    teacher = TeacherAgent(llm=_BoomLLM())
    state = SessionState(session_id="boom", student_id="s")
    state.scratch.update(grade="初中", subject="数学", subject_id="math")
    state.subject = "数学"

    response = await teacher.answer(StudentTurn(text="什么是数轴？"), state)

    assert response.speech, "学生必须有东西可听（兜底话术）"
    assert response.meta.get("llm_error"), "失败必须如实标出来，供 L0/L3 看见"

    verdict, reasons = judge(response=response)
    assert verdict == "degraded", "兜底话术不算干净交付"
    assert any("模型调用失败" in r for r in reasons), f"判定依据里要能看出根因：{reasons}"
