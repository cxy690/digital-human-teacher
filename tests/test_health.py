"""L1 健康诊断（core/health.py）的行为测试。

为什么这层最需要测试
--------------------
L0 算错一个比例只是数字不准；**L1 判错会直接产生误导性的告警** ——
「狼来了」喊多了，真出问题时没人看。所以本文件重点钉三类边界：

1. **小样本不许下判断**（早上跑 3 条就说「失败率 33%」是纯噪声）；
2. **没变差不许报**（阈值要真的起作用）；
3. **不可比不许比**（用 ``--only 单题`` 跑出来的分数不能和整套基线相减）。

判定口径（阈值）本身不测死 —— 它是会随阶段调的产品参数；这里测的是
「规则被正确执行」，而不是「阈值恰好等于某个数」。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from config.settings import settings
from core.health import (
    MIN_SAMPLES,
    Metric,
    _window,
    as_dict,
    collect,
    coverage_of,
    diagnose,
    last_gain_of,
    staleness_of,
)


# ---------------------------------------------------------------------- 工具
class _FakeStore:
    """只实现 ``read`` 的桩：诊断层只用到这一个方法。"""

    def __init__(self, records: list[dict]) -> None:
        self._records = records

    def read(self, days: int, limit: int = 0) -> list[dict]:
        return list(self._records)


def _rec(day: str, *, verdict: str = "pass", subject: str = "数学", **extra) -> dict:
    return {
        "day": day,
        "verdict": verdict,
        "subject": subject,
        "kind": "teach",
        "elapsed_ms": 1000.0,
        **extra,
    }


def _day_block(day: str, n: int, *, bad: int = 0, degraded: int = 0,
               json_degraded: int = 0, quality: int = 100, subject: str = "数学") -> list[dict]:
    """造某一天的 n 条记录：前 bad 条 failed、接着 degraded 条 degraded、其余 pass。"""
    out: list[dict] = []
    for i in range(n):
        v = "failed" if i < bad else ("degraded" if i < bad + degraded else "pass")
        out.append(_rec(
            day, verdict=v, subject=subject,
            json_degraded=i < json_degraded,
            quality={"score": 60 if i < bad else quality, "issues": []},
        ))
    return out


@pytest.fixture()
def quiet_learning(tmp_path, monkeypatch):
    """把学习目录指到 tmp 并放一份「新鲜」的经验文件。

    否则每个用例都会因「经验库不存在」而多出一条 staleness 告警，
    干扰对指标判断的断言。
    """
    exp = tmp_path / "exp"
    exp.mkdir()
    (exp / "math.jsonl").write_text('{"id":"x"}\n', encoding="utf-8")
    monkeypatch.setattr(settings, "skill_experience_dir", str(exp))
    monkeypatch.setattr(settings, "memory_dir", str(tmp_path / "mem"))
    return exp


def _dims(diag) -> list[str]:
    return [f.dimension for f in diag.findings]


def _by_dim(diag, dim: str) -> list:
    return [f for f in diag.findings if f.dimension == dim]


# ---------------------------------------------------------------------- collect
def test_collect_computes_ratios() -> None:
    """比例算对：affected = degraded + failed。"""
    records = _day_block("2026-10-01", 10, bad=2, degraded=3)
    vals = collect(records)
    assert vals["failed_ratio"] == pytest.approx(0.2)
    assert vals["degraded_ratio"] == pytest.approx(0.3)
    assert vals["affected_ratio"] == pytest.approx(0.5)


def test_collect_handles_empty() -> None:
    """空窗返回全 None，不能除零。"""
    vals = collect([])
    assert vals["affected_ratio"] is None
    assert vals["quality_avg"] is None


def test_collect_step_error_ratio_is_finer_than_turn_failure() -> None:
    """步骤异常率统计的是「每一步」，比整回合失败更细 —— 它是更早的信号。"""
    records = [
        _rec("2026-10-01", steps=[
            {"func": "a", "ok": True}, {"func": "b", "ok": False},
            {"func": "c", "ok": True}, {"func": "d", "ok": True},
        ]),
    ]
    assert collect(records)["step_error_ratio"] == pytest.approx(0.25)


# ---------------------------------------------------------------------- 窗口
def test_window_slices_by_day_and_offset() -> None:
    """窗口按「天」切；offset=1 取紧邻的前一个等长窗口。"""
    records = [_rec(f"2026-10-{d:02d}") for d in range(1, 11)]
    cur = _window(records, 3, offset=0)
    base = _window(records, 3, offset=1)
    assert {r["day"] for r in cur} == {"2026-10-08", "2026-10-09", "2026-10-10"}
    assert {r["day"] for r in base} == {"2026-10-05", "2026-10-06", "2026-10-07"}


def test_window_offset_beyond_history_is_empty() -> None:
    """历史不足时基线窗口为空（而不是回绕到当前窗口，那会得出「毫无变化」的假象）。"""
    records = [_rec(f"2026-10-{d:02d}") for d in range(1, 4)]
    assert _window(records, 3, offset=1) == []


# ---------------------------------------------------------------------- 判定
def test_no_change_reports_no_metric_finding(quiet_learning) -> None:
    """两个窗口一模一样 -> 不许报任何指标退化（防误报的第一道关）。"""
    records = _day_block("2026-10-01", 25) + _day_block("2026-10-02", 25)
    diag = diagnose(days=1, store=_FakeStore(records))
    assert _by_dim(diag, "失败率") == []
    assert _by_dim(diag, "先行信号") == []


def test_small_sample_refuses_to_judge() -> None:
    """样本不足时只报事实、不下结论 —— 哪怕比例看起来吓人。"""
    # 3 条里 2 条 failed = 66%，若不加门禁必然误报
    records = _day_block("2026-10-01", 25) + _day_block("2026-10-02", 3, bad=2)
    diag = diagnose(days=1, store=_FakeStore(records))
    tiers = [f for f in diag.findings if "样本不足" in f.title]
    assert tiers, f"应当提示样本不足，实际：{_dims(diag)}"
    assert all(f.level == "ok" for f in tiers), "样本不足只是提示，不该是告警"
    assert _by_dim(diag, "失败率") == []


def test_failure_spike_is_flagged(quiet_learning) -> None:
    """失败率显著上升要报，且等级随幅度升级。"""
    base = _day_block("2026-10-01", 30, bad=1)          # 3.3%
    cur = _day_block("2026-10-02", 30, bad=15)          # 50%
    diag = diagnose(days=1, store=_FakeStore(base + cur))
    hits = [f for f in _by_dim(diag, "失败率") if "失败率" in f.title]
    assert hits, "失败率从 3.3% 涨到 50% 必须报警"
    assert hits[0].level == "critical"


def test_json_degrade_is_reported_as_leading_signal(quiet_learning) -> None:
    """JSON 降级率要归到「先行信号」维度，而不是笼统的失败率。

    这是 L1 的核心价值：降级是板书丢失的**上游**，先于学生察觉。
    """
    base = _day_block("2026-10-01", 30, json_degraded=0)
    cur = _day_block("2026-10-02", 30, json_degraded=12)   # 0% -> 40%
    diag = diagnose(days=1, store=_FakeStore(base + cur))
    hits = [f for f in diag.findings if "JSON 降级率" in f.title]
    assert hits, f"JSON 降级率 0%→40% 应触发先行信号，实际：{_dims(diag)}"
    assert hits[0].dimension == "先行信号"
    assert "max_tokens" in hits[0].advice, "应给出可执行的排查方向"


def test_quality_drop_is_detected(quiet_learning) -> None:
    """质检均分下降要能抓到（它是「讲得变差」的先行指标）。"""
    base = _day_block("2026-10-01", 30, quality=95)
    cur = _day_block("2026-10-02", 30, quality=60)         # 95 -> 60
    diag = diagnose(days=1, store=_FakeStore(base + cur))
    assert [f for f in diag.findings if "质检均分" in f.title], _dims(diag)


def test_tiny_absolute_change_is_not_reported(quiet_learning) -> None:
    """相对涨幅再大，绝对变化太小也不报（0.5% → 1% 属于噪声）。"""
    base = _day_block("2026-10-01", 200, bad=1)     # 0.5%
    cur = _day_block("2026-10-02", 200, bad=2)      # 1.0%，相对 +100%
    diag = diagnose(days=1, store=_FakeStore(base + cur))
    assert [f for f in _by_dim(diag, "失败率") if "失败率变差" in f.title] == [], \
        "0.5%→1% 不该报（相对翻倍，但绝对只有 0.5 个百分点）"


def test_subject_outlier_beats_global_average(quiet_learning) -> None:
    """某个学科显著差于全局时要单独点出来 —— 比全局微涨更值得先看。"""
    base = (_day_block("2026-10-01", 20, subject="数学")
            + _day_block("2026-10-01", 20, subject="物理"))
    cur = (_day_block("2026-10-02", 20, subject="数学", bad=1)      # 5%
           + _day_block("2026-10-02", 20, subject="物理", bad=12))  # 60%
    diag = diagnose(days=1, store=_FakeStore(base + cur))
    hits = [f for f in _by_dim(diag, "失败率") if "物理" in f.title]
    assert hits, f"物理受影响率远高于数学，应单独提示，实际：{[f.title for f in diag.findings]}"


def test_missing_subjects_reported_as_blind_spot(quiet_learning) -> None:
    """零数据 ≠ 零问题：没数据的学科要作为「覆盖」盲区报出来。"""
    diag = diagnose(days=1, store=_FakeStore(_day_block("2026-10-01", 30)))
    hits = [f for f in _by_dim(diag, "覆盖") if "没有任何数据" in f.title]
    assert hits, "一个学科只有数据时，其余学科应被标为盲区"


# ---------------------------------------------------------------------- 覆盖 / staleness
def test_coverage_counts_subjects_kinds_days() -> None:
    records = [_rec("2026-10-01", subject="数学"), _rec("2026-10-02", subject="物理")]
    cov = coverage_of(records, 7)
    assert cov["subject_count"] == 2
    assert cov["days_with_data"] == 2
    assert cov["window_days"] == 7


def test_staleness_flags_absent_experience(tmp_path, monkeypatch) -> None:
    """经验库不存在时明确报出来 —— 学习停滞不会体现在失败率上。"""
    monkeypatch.setattr(settings, "skill_experience_dir", str(tmp_path / "nope"))
    monkeypatch.setattr(settings, "memory_dir", str(tmp_path / "nope2"))
    st = staleness_of()
    assert st["experience"]["exists"] is False
    assert st["memory"]["exists"] is False


def test_staleness_reports_age(tmp_path, monkeypatch) -> None:
    """有数据时给出「多久没更新」，供判定是否停学。"""
    exp = tmp_path / "exp"
    exp.mkdir()
    (exp / "math.jsonl").write_text('{"a":1}\n{"a":2}\n', encoding="utf-8")
    monkeypatch.setattr(settings, "skill_experience_dir", str(exp))
    monkeypatch.setattr(settings, "memory_dir", str(tmp_path / "mem"))
    st = staleness_of()
    assert st["experience"]["exists"] is True
    assert st["experience"]["items"] == 2
    assert st["experience"]["age_hours"] is not None
    assert st["experience"]["age_hours"] < 1.0, "刚写的文件应视为新鲜"


def test_stale_experience_triggers_warning(tmp_path, monkeypatch) -> None:
    """经验超过阈值没更新 -> staleness 告警。"""
    exp = tmp_path / "exp"
    exp.mkdir()
    f = exp / "math.jsonl"
    f.write_text('{"a":1}\n', encoding="utf-8")
    old = time.time() - 100 * 3600          # 100 小时前
    import os

    os.utime(f, (old, old))
    monkeypatch.setattr(settings, "skill_experience_dir", str(exp))
    monkeypatch.setattr(settings, "memory_dir", str(tmp_path / "mem"))

    diag = diagnose(days=1, store=_FakeStore(_day_block("2026-10-01", 30)))
    assert [f for f in _by_dim(diag, "staleness") if "没有更新" in f.title], _dims(diag)


# ---------------------------------------------------------------------- last_gain
def _write_eval(tmp_path: Path, *, base_cases: int, base_total: float,
                run_cases: int, run_total: float) -> Path:
    evals = tmp_path / "evals"
    (evals / "runs" / "2026-10-12_10-00-00").mkdir(parents=True)
    (evals / "baseline.json").write_text(json.dumps({
        "stamp": "2026-10-12_09-00-00",
        "summary": {"total": base_total, "cases": base_cases},
        "per_case": {},
    }), encoding="utf-8")
    (evals / "runs" / "2026-10-12_10-00-00" / "result.json").write_text(json.dumps({
        "stamp": "2026-10-12_10-00-00",
        "summary": {"total": run_total, "cases": run_cases},
        "rows": [],
    }), encoding="utf-8")
    return evals


def test_last_gain_comparable(tmp_path) -> None:
    """题数一致才给 delta。"""
    evals = _write_eval(tmp_path, base_cases=12, base_total=90.0,
                        run_cases=12, run_total=94.0)
    g = last_gain_of(evals)
    assert g["available"] and g["comparable"]
    assert g["delta"] == pytest.approx(4.0)


def test_last_gain_incomparable_does_not_report_delta(tmp_path) -> None:
    """题数不一致时必须拒绝比较。

    这条防的是一个很容易犯的错：用 ``--only 某题`` 跑单题验证，产出的 1 题结果
    若与 12 题的基线相减，会得出「+18 分，一切正常」这种完全错误的乐观结论。
    """
    evals = _write_eval(tmp_path, base_cases=12, base_total=80.0,
                        run_cases=1, run_total=100.0)
    g = last_gain_of(evals)
    assert g["available"]
    assert g["comparable"] is False
    assert g["delta"] is None, "不可比时不许给出 delta"


def test_incomparable_last_gain_is_not_an_alarm(tmp_path, monkeypatch) -> None:
    """不可比只是一条提示（ok），绝不能因此报「评测下滑」。"""
    evals = _write_eval(tmp_path, base_cases=12, base_total=94.8,
                        run_cases=1, run_total=100.0)
    monkeypatch.setattr("core.health.last_gain_of", lambda *_: last_gain_of(evals))
    exp = tmp_path / "exp"
    exp.mkdir()
    (exp / "math.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(settings, "skill_experience_dir", str(exp))
    monkeypatch.setattr(settings, "memory_dir", str(tmp_path / "mem"))

    diag = diagnose(days=1, store=_FakeStore(_day_block("2026-10-01", 30)))
    hits = _by_dim(diag, "last_gain")
    assert hits and hits[0].level == "ok"
    assert "不可比" in hits[0].title


# ---------------------------------------------------------------------- 输出
def test_metric_direction_is_normalized() -> None:
    """``worse()`` 要把方向归一：耗时上涨与均分下跌都算「变差」（正值）。"""
    up = Metric(key="latency_p95", label="耗时", unit="ms", bad_when="up",
                value=2000.0, baseline=1000.0)
    down = Metric(key="quality_avg", label="质检", unit="score", bad_when="down",
                  value=80.0, baseline=100.0)
    better = Metric(key="quality_avg", label="质检", unit="score", bad_when="down",
                    value=110.0, baseline=100.0)
    assert up.worse() > 0 and down.worse() > 0
    assert better.worse() < 0, "变好要返回负数，才不会被当成退化"


def test_as_dict_is_json_serializable(quiet_learning) -> None:
    """诊断结果要能直接进接口 / JSON 文件（不能夹带 dataclass 或 NaN）。"""
    diag = diagnose(days=1, store=_FakeStore(_day_block("2026-10-01", 30)))
    text = json.dumps(as_dict(diag), ensure_ascii=False)
    assert "findings" in text
    assert MIN_SAMPLES > 0


# ---------------------------------------------------------------------- HTTP 接口
@pytest.fixture()
def _client(tmp_path, monkeypatch):
    """挂上路由的测试客户端；裁决目录指到 tmp，避免读真实数据。"""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from server.routes import router

    monkeypatch.setattr(settings, "verdict_dir", str(tmp_path / "verdicts"))
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_diagnosis_endpoint_shape(_client) -> None:
    """接口要返回可直接消费的结构，且在**没有任何数据时也不崩**。"""
    r = _client.get("/api/health/diagnosis?days=7")
    assert r.status_code == 200
    body = r.json()
    for key in ("window", "level", "metrics", "findings", "coverage", "staleness"):
        assert key in body, f"缺少字段 {key}"
    assert isinstance(body["findings"], list)
    assert body["level"] in ("ok", "warn", "critical", "unknown")


def test_diagnosis_endpoint_clamps_absurd_params(_client) -> None:
    """窗口天数要夹到合法范围 —— 接口是公开的，不能让一次请求去扫几年的数据。"""
    body = _client.get("/api/health/diagnosis?days=99999").json()
    assert body["window"]["days"] <= 90

    body = _client.get("/api/health/diagnosis?days=0").json()
    assert body["window"]["days"] >= 1


def test_health_probe_stays_separate_and_cheap(_client) -> None:
    """``/health`` 必须仍是「便宜且永远 ok」的探活接口 —— 诊断逻辑不能混进去。"""
    r = _client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    # 探活不该带诊断字段（带了说明有人把慢 IO 塞进了探活路径）
    assert "findings" not in body
    assert "staleness" not in body
