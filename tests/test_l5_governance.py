"""L5 治理层测试：实验记录（``evals/results.tsv``）与人工抽检台。

钉住两件事：

1. **实验记录要能回答「趋势」** —— 一行一轮、表头只写一次、相对基线差多少如实记录；
   写不进去也不能把一次评测搞挂（它是旁路）。
2. **抽检台要按风险排** —— 四个信号各自能触发、健康条目不该出现、
   风险高的排前面。人的几分钟就那么点，排序错了这个闸门就等于没有。
"""
from __future__ import annotations

import json

import pytest

import scripts.eval_agent as eval_agent
from core.skills.experience import review_queue
from core.skills.experience.models import ExperienceItem
from core.skills.experience.store import ExperienceStore


# ---------------------------------------------------------------- 实验记录
def _result(**summary) -> dict:
    base = {
        "total": 95.0, "rule_avg": 100.0, "judge_avg": 90.0,
        "judge_enabled": True, "cases": 12, "errors": 0, "degraded": 0,
        "latency_p50_ms": 9000.0,
    }
    base.update(summary)
    return {"stamp": "2026-10-12_20-00-00", "summary": base, "rows": []}


def test_results_tsv_writes_header_once_and_one_row_per_run(monkeypatch, tmp_path) -> None:
    """一行一轮、表头只写一次 —— 这样 ``tail`` 和 ``git diff`` 才有意义。"""
    tsv = tmp_path / "results.tsv"
    monkeypatch.setattr(eval_agent, "RESULTS_PATH", tsv)
    monkeypatch.setattr(eval_agent, "BASELINE_PATH", tmp_path / "none.json")

    eval_agent.append_results_tsv(_result(), kind="run", note="第一次")
    eval_agent.append_results_tsv(_result(total=96.0), kind="gate", note="第二次")

    lines = tsv.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3, "表头 + 两行数据"
    assert lines[0].split("\t") == list(eval_agent._RESULTS_HEADER)
    assert "第一次" in lines[1] and "第二次" in lines[2]

    cols = lines[1].split("\t")
    assert cols[1] == "run" and cols[2] == "95.0"
    assert cols[9] == "-", "还没基线时如实写「-」，不要编一个 0"


def test_results_tsv_reports_delta_and_flags_mode_mismatch(monkeypatch, tmp_path) -> None:
    """相对基线差多少要如实记；裁判模式不同时换成规则分比**并标出来**。

    这一点很关键：带裁判时总分是「规则与裁判各半」，免费模式只有规则分，
    直接比会得出假的「通过」—— 记录里必须看得出这次比的是哪个口径。
    """
    base = tmp_path / "baseline.json"
    base.write_text(json.dumps(_result(total=93.0)), encoding="utf-8")
    tsv = tmp_path / "results.tsv"
    monkeypatch.setattr(eval_agent, "RESULTS_PATH", tsv)
    monkeypatch.setattr(eval_agent, "BASELINE_PATH", base)

    # 同模式：直接比总分
    assert eval_agent.compare_with_baseline(_result(total=95.0)) == "+2.0"
    # 本次无裁判 / 基线有裁判：总分算法不同 → 比规则均分并标注
    assert eval_agent.compare_with_baseline(
        _result(total=100.0, judge_enabled=False, judge_avg=None)
    ) == "+0.0(规则)"


def test_results_tsv_never_breaks_the_run(monkeypatch, tmp_path) -> None:
    """写记录失败只是旁路故障，**绝不能让一次评测崩掉**。"""
    monkeypatch.setattr(eval_agent, "RESULTS_PATH", tmp_path / "no" / "such" / "dir" / "r.tsv")
    monkeypatch.setattr(eval_agent, "BASELINE_PATH", tmp_path / "none.json")
    eval_agent.append_results_tsv(_result(), note="目录不存在也不该抛")


# ---------------------------------------------------------------- 抽检台
def _store(tmp_path) -> ExperienceStore:
    return ExperienceStore(root=tmp_path)


def test_review_flags_low_gain_rate(monkeypatch, tmp_path) -> None:
    """上过场、干净交付率却明显偏低 —— 最该先看的一条。"""
    store = _store(tmp_path)
    item = store.observe("math", "pitfall", "这条经验其实没什么用，占了名额", confidence=0.9)
    assert item is not None
    for _ in range(8):
        store.record_outcome("math", [item.id], gain=0.0)

    rows = review_queue("math", store=store)
    assert rows and rows[0]["id"] == item.id
    assert any("干净交付" in r for r in rows[0]["reasons"])
    assert rows[0]["risk"] > 0.5


def test_review_flags_unverified_high_confidence(monkeypatch, tmp_path) -> None:
    """模型自评很高却一次都没上过场 —— 值得先扫一眼描述准不准。"""
    store = _store(tmp_path)
    item = store.observe("math", "analogy", "把分数比作切蛋糕，先切几刀看分母", confidence=0.95)
    assert item is not None and item.pulls == 0

    rows = review_queue("math", store=store)
    assert any(r["id"] == item.id and any("还没上过场" in x for x in r["reasons"]) for r in rows)


def test_review_flags_near_duplicates(monkeypatch, tmp_path) -> None:
    """两条意思重复、各占一个注入名额 —— 白浪费，值得合并。

    注意这两句的相似度是**实测定过的**（归一化后 0.757）：低于存储层去重阈值
    （``SIMILARITY_THRESHOLD = 0.85``，再高就被 ``observe`` 合并成一条了），
    又高于抽检台的提醒阈值（``SIMILARITY_FOR_REVIEW = 0.72``）。
    写成「只差一个『哦』」那种会被存储层直接合并，等于什么都没测。
    """
    store = _store(tmp_path)
    store.observe("math", "pitfall", "学生常只对一边运算，忘记等式两边要同步变化", confidence=0.7)
    store.observe("math", "pitfall", "学生常只对一边运算，忘了两边要一起变", confidence=0.7)
    assert len(store.active("math")) == 2, "这两条必须都活下来，否则测不到「近重复」"

    rows = review_queue("math", store=store)
    assert rows, "近重复应该被列出来"
    assert any("高度重合" in r for x in rows for r in x["reasons"])


def test_review_duplicate_threshold_does_not_false_positive(tmp_path) -> None:
    """阈值压到 0.62 不能变成「什么都报重复」。

    实测依据：真实重复的相似度落在 0.71~0.76，而彼此无关的条目落在 0.07 附近 ——
    中间隔着十倍。本用例把「无关」那一侧钉住，防止以后有人手滑把阈值继续往下压。
    """
    store = _store(tmp_path)
    store.observe("math", "pitfall", "移项要变号，两边同时加减同一个数", confidence=0.9)
    store.observe("math", "pitfall", "讲数轴要强调原点、正方向、单位长度", confidence=0.9)
    store.observe("math", "analogy", "可用温度计或直尺类比数轴帮助学生理解", confidence=0.9)

    rows = review_queue("math", store=store)
    assert rows, "高置信未验证的条目本来就该在队列里"
    assert not any("高度重合" in r for x in rows for r in x["reasons"]), (
        "这些条目彼此无关，不该被报成重复"
    )


def test_review_ignores_healthy_items_and_retired(monkeypatch, tmp_path) -> None:
    """战绩良好、且没什么可疑之处的条目**不该出现在抽检台**（否则注意力被稀释）。"""
    store = _store(tmp_path)
    good = store.observe("math", "pitfall", "移项要变号，这是最常见的错误", confidence=0.5)
    assert good is not None
    for _ in range(10):
        store.record_outcome("math", [good.id], gain=1.0)

    dead = store.observe("math", "pitfall", "这条已经被停用了，不该再提示", confidence=0.95)
    assert dead is not None
    store.retire("math", dead.id)

    assert review_queue("math", store=store) == []


def test_review_sorts_by_risk(monkeypatch, tmp_path) -> None:
    """风险高的排前面 —— 这个顺序就是「人的几分钟先花在哪」。"""
    store = _store(tmp_path)
    bad = store.observe("math", "pitfall", "实测偏低的那一条经验放在这里", confidence=0.5)
    mild = store.observe("math", "analogy", "高置信但还没验证过的那一条放在这里", confidence=0.9)
    assert bad is not None and mild is not None
    for _ in range(8):
        store.record_outcome("math", [bad.id], gain=0.0)

    rows = review_queue("math", store=store)
    assert [r["id"] for r in rows][0] == bad.id
    assert rows[0]["risk"] >= rows[-1]["risk"]


def test_review_limits_and_scopes(monkeypatch, tmp_path) -> None:
    """``limit`` 生效；不给老师时看全部老师。

    条目内容刻意取得**彼此无关**：写「第 1 条 / 第 2 条」这种只差一个数字的，
    会被 ``observe`` 的去重直接合并成一条，条数就对不上了。
    """
    store = _store(tmp_path)
    math_notes = [
        "移项要变号，两边同时加减同一个数",
        "去分母别忘记每一项都乘",
        "负号看错会导致整题全错",
        "解完方程记得代回原式检验",
        "未知数与系数相邻是乘法不是加法",
    ]
    physics_notes = [
        "牛顿第一定律讲的是惯性",
        "受力分析先画重力再找支持力",
        "速度与加速度方向可以相反",
        "摩擦力方向与相对运动趋势相反",
        "能量守恒要先确定系统边界",
    ]
    for note in math_notes:
        store.observe("math", "pitfall", note, confidence=0.9)
    for note in physics_notes:
        store.observe("physics", "pitfall", note, confidence=0.9)

    assert len(store.active("math")) == len(math_notes), "这些条目彼此无关，不该被去重合并"
    assert len(review_queue("math", limit=3, store=store)) == 3
    assert {r["teacher"] for r in review_queue("", limit=50, store=store)} == {"math", "physics"}


def test_risk_of_returns_actionable_reasons() -> None:
    """每个理由都要对应一个**能立刻做的动作**，不能只是描述现象。"""
    from core.skills.experience.review import risk_of

    item = ExperienceItem(
        id="x", teacher_key="math", kind="pitfall", text="一条测试经验",
        hits=2, confidence=0.9, pulls=20, wins=2,
    )
    risk, reasons = risk_of(item)
    assert risk > 0 and reasons
    assert any("考虑停用或改措辞" in r for r in reasons)
    assert any("影响面大" in r for r in reasons), "注入 20 次要提示影响面"


@pytest.mark.parametrize("pulls,rate_ok", [(4, False), (5, True)])
def test_min_pulls_threshold(tmp_path, pulls: int, rate_ok: bool) -> None:
    """样本太少时不下「实测偏低」的结论（4 次的命中率没有意义）。"""
    store = _store(tmp_path)
    item = store.observe("math", "pitfall", "样本量边界的经验条目在这里", confidence=0.5)
    assert item is not None
    for _ in range(pulls):
        store.record_outcome("math", [item.id], gain=0.0)

    rows = review_queue("math", store=store)
    flagged = any("干净交付" in r for x in rows for r in x["reasons"])
    assert flagged is rate_ok
