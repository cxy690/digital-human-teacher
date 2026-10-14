"""L4 验证守门层：AI 教师的**真实质量评测 + 回归门禁**。

为什么需要它
------------
项目有 56 个测试文件，但 ``tests/conftest.py`` 把 LLM/ASR/TTS 全部降级为 Mock
—— 只能证明「代码逻辑没坏」，**证明不了「回答质量没退」**。于是出现过一个典型盲区：

    经验层写好了、测试也在、但没接进答疑链路 —— 测试全绿，生产什么都没跑。

也就是说：**单元测试守不住提示词与链路级别的退化**。本脚本补上这一层：

* ``run``     跑一遍评测集（真实 LLM），逐题打分并落盘；
* ``promote`` 把最近一次结果定为**基线**（供门禁比较）；
* ``gate``    跑一遍并与基线比，**掉分超过阈值就非零退出**（CI / 上线前用）。

评分两层，成本与用途分明
------------------------
============  ==========  ====================  ==================
层            成本        判什么                为什么需要
============  ==========  ====================  ==================
规则分        零（不调    **结构完整性**：       最常见的退化就是这一类
              LLM）       话术/板书/步数/          （改提示词把 JSON 契约
                          知识点/降级/质检        弄坏），且它确定性、免费
裁判分        +1 次 LLM   **内容质量**：要点覆盖   结构完好但「讲得变差」是
              （--judge）  / 正确性 / 好懂度       规则分看不出来的
============  ==========  ====================  ==================

总分 = judge 开启时两者各半，否则纯规则分。

用法
----
::

    python scripts/eval_agent.py run                 # 跑一遍（免费，只用规则分）
    python scripts/eval_agent.py run --judge         # 连内容质量一起评（多花 12 次调用）
    python scripts/eval_agent.py promote             # 用最近一次结果当基线
    python scripts/eval_agent.py gate                # 回归门禁（默认带裁判）
    python scripts/eval_agent.py gate --no-judge     # 免费模式门禁

评测期间会**临时关掉自我学习**（经验复盘 / 学情蒸馏）—— 否则每题会多花两次调用，
而且评测产生的「经验」会污染真实经验库。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVALS_DIR = ROOT / "evals"
CASES_PATH = EVALS_DIR / "cases.jsonl"
RUNS_DIR = EVALS_DIR / "runs"
BASELINE_PATH = EVALS_DIR / "baseline.json"

#: L5 实验记录：每跑一轮就**在末尾追一行**（表头只写一次）。
#:
#: 为什么要有它：单份 ``runs/<stamp>/result.json`` 只能回答「**这一次**多少分」，
#: 回答不了「**这周质量在升还是在降**」—— 要回答后者，得把 runs/ 底下的目录
#: 一个个翻过去。一行一轮之后：
#:
#: * ``tail -5 evals/results.tsv`` 五秒钟看清趋势；
#: * 它是纯文本，**可以进 git** —— 于是「这次改动让质量动了多少」能 ``git diff``，
#:   而不是靠记忆或者翻聊天记录（Karpathy 那套实验记录的全部价值就在这一点上）。
#:
#: 列的含义见 :data:`_RESULTS_HEADER`。
RESULTS_PATH = EVALS_DIR / "results.tsv"

_RESULTS_HEADER = (
    "时间", "类型", "总分", "规则均分", "裁判均分", "题数",
    "异常", "降级", "耗时中位ms", "相对基线", "说明",
)


def compare_with_baseline(result: dict[str, Any]) -> str:
    """本轮结果相对基线差多少（没基线 / 裁判模式不同时**如实说明**，不硬凑一个数）。

    与 ``do_gate`` 同一套口径：带裁判与不带裁判的**总分算法不同**，直接混比会得出
    假的「通过」；模式不同时改用规则均分比，并在值后面标出 ``(规则)``。
    """
    if not BASELINE_PATH.exists():
        return "-"
    try:
        base = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        if bool(base["summary"].get("judge_enabled")) == bool(
            result["summary"].get("judge_enabled")
        ):
            delta = float(result["summary"]["total"]) - float(base["summary"]["total"])
            return f"{delta:+.1f}"
        delta = float(result["summary"].get("rule_avg") or 0) - float(
            base["summary"].get("rule_avg") or 0
        )
        return f"{delta:+.1f}(规则)"
    except Exception:  # noqa: BLE001 - 记录一行不值得让整轮评测失败
        return "?"


def append_results_tsv(result: dict[str, Any], *, kind: str = "run", note: str = "") -> None:
    """把这一轮追进 :data:`RESULTS_PATH`（表头只写一次）。

    写失败只告警、不抛：实验记录是**旁路**，绝不能因为磁盘问题把一次评测搞挂
    （与 ``core.verdict`` / ``core.safety_audit`` 同一取舍）。
    """
    s = result.get("summary") or {}

    def _num(key: str) -> str:
        value = s.get(key)
        return "-" if value is None else str(value)

    row = (
        str(result.get("stamp") or ""),
        kind,
        _num("total"),
        _num("rule_avg"),
        _num("judge_avg"),
        _num("cases"),
        _num("errors"),
        _num("degraded"),
        _num("latency_p50_ms"),
        compare_with_baseline(result),
        (note or "").replace("\t", " ").replace("\n", " ").strip(),
    )
    try:
        fresh = not RESULTS_PATH.exists()
        with RESULTS_PATH.open("a", encoding="utf-8") as handle:
            if fresh:
                handle.write("\t".join(_RESULTS_HEADER) + "\n")
            handle.write("\t".join(row) + "\n")
    except OSError as err:
        print(f"（实验记录写入失败，不影响本轮结果：{err}）")

#: 门禁阈值：总分相对基线**下跌**超过这个值即判定回归。
#:
#: 为什么不设成 0：LLM 有随机性，同一份代码两次跑的总分天然会飘几分。
#: 3 分是「比噪声大、又比真实退化小」的经验值 —— 实测同代码重跑波动在 1~2 分内。
GATE_TOLERANCE = 3.0

#: 裁判模型的最低分（低于它认为不合格，仅用于报告，不直接决定门禁）
JUDGE_SUBSCORE_FLOOR = 60.0


# ---------------------------------------------------------------------- 用例
def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    """读评测用例（JSONL，一行一题）。"""
    if not path.exists():
        raise SystemExit(f"找不到评测集：{path}")
    cases: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            cases.append(json.loads(line))
    if not cases:
        raise SystemExit(f"评测集是空的：{path}")
    return cases


# ---------------------------------------------------------------------- 打分
def score_by_rules(response: Any) -> tuple[float, list[str]]:
    """规则分（0~100）：只判**结构完整性**，不判内容质量。

    为什么结构也值得单独立分：绝大多数的提示词退化**不是「讲错了」而是「讲不出来」**
    —— 契约被改坏 → JSON 解析失败 → 板书整块丢弃 → 学生只看到一句兜底话术。
    这一类规则分能零成本、确定性地抓到。
    """
    notes: list[str] = []
    if response is None:
        return 0.0, ["无产物"]

    score = 0.0
    speech = str(getattr(response, "speech", "") or "").strip()
    board = list(getattr(response, "board_ops", None) or [])
    meta = getattr(response, "meta", None) or {}
    steps = list(meta.get("steps") or [])
    points = list(getattr(response, "knowledge_points", None) or [])

    # 话术充足（兜底话术通常很短，这条能把它们滤掉）
    if len(speech) >= 20:
        score += 25
    elif speech:
        score += 10
        notes.append(f"话术偏短({len(speech)}字)")
    else:
        notes.append("无话术")

    if board:
        score += 20
    else:
        notes.append("无板书")

    if len(steps) >= 3:
        score += 15
    elif steps:
        score += 7
        notes.append(f"步骤偏少({len(steps)}步)")
    else:
        notes.append("无步骤")

    if points:
        score += 15
    else:
        notes.append("无知识点")

    if not meta.get("json_degraded"):
        score += 15
    else:
        notes.append("JSON 降级")

    quality = meta.get("quality") or {}
    if isinstance(quality, dict) and quality.get("score") == 100:
        score += 10
    elif isinstance(quality, dict) and quality.get("score") is not None:
        notes.append(f"质检扣分({quality.get('score')})")
    else:
        notes.append("无质检结论")

    return round(score, 1), notes


_JUDGE_PROMPT = """你在给一次 AI 老师的讲解打分。请**严格但公道**。

【题目】{question}
【学生应该学到】{expect}
【老师的口语讲解】{speech}
【白板上的分步】{steps}

请只输出 JSON：
{{"coverage": 0~100,     // 「应该学到」的要点覆盖了多少
  "correctness": 0~100,  // 有没有硬伤：概念讲错、算错、跳步、条件遗漏
  "clarity": 0~100,      // 对中小学学生是否好懂（太抽象 / 术语堆砌要扣）
  "comment": "一句话点评（指出最该改的一点）"}}
"""


async def score_by_judge(case: dict[str, Any], response: Any) -> dict[str, Any] | None:
    """裁判分（0~100）：让一个 LLM 按要点覆盖 / 正确性 / 好懂度打分。

    判不了就返回 ``None``，调用方按「无裁判分」处理 —— **裁判挂掉不能把门禁搞崩**。
    """
    if response is None:
        return None
    from models.llm import get_llm

    meta = getattr(response, "meta", None) or {}
    steps = "；".join(str(s) for s in (meta.get("steps") or [])[:6]) or "（无）"
    prompt = _JUDGE_PROMPT.format(
        question=case.get("question", ""),
        expect="、".join(case.get("expect") or []) or "（未指定）",
        speech=str(getattr(response, "speech", "") or "")[:600],
        steps=steps[:600],
    )
    try:
        raw = await get_llm().chat(
            [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=400,
            json_mode=True,
        )
        from core.llm_json import parse_json_object  # noqa: PLC0415 - 局部导入避免冷启动开销

        data = parse_json_object(raw or "")
    except Exception as err:  # noqa: BLE001 - 裁判失败不阻断门禁
        return {"error": f"{type(err).__name__}: {err}"}
    if not isinstance(data, dict):
        return {"error": "裁判返回不是对象"}

    parts: list[float] = []
    for key in ("coverage", "correctness", "clarity"):
        try:
            parts.append(max(0.0, min(100.0, float(data.get(key)))))
        except (TypeError, ValueError):
            pass
    if not parts:
        return {"error": "裁判没给出可用分数"}
    return {
        "score": round(sum(parts) / len(parts), 1),
        "coverage": data.get("coverage"),
        "correctness": data.get("correctness"),
        "clarity": data.get("clarity"),
        "comment": str(data.get("comment") or "")[:120],
    }


# ---------------------------------------------------------------------- 跑一题
async def run_case(case: dict[str, Any], *, use_judge: bool, speak: bool) -> dict[str, Any]:
    """跑一题：走**生产链路**（core.harness 装配的 Supervisor）。"""
    from config.settings import settings

    import core.harness as harness
    from core.state.models import StudentTurn
    from core.state.session import SessionState

    state = SessionState(session_id=f"eval-{case['id']}", student_id="eval-student")
    state.subject = case.get("subject", "")
    state.scratch["subject"] = case.get("subject", "")
    if case.get("teacher"):
        state.teacher = case["teacher"]
        state.scratch["teacher"] = case["teacher"]

    supervisor = harness.Supervisor(state)
    if not speak:
        # 跳过 TTS 与数字人渲染：它们不影响 LLM 产出的文本质量，但会让评测慢一个量级
        async def _skip(_response, **_kw):  # type: ignore[no-untyped-def]
            return _response

        supervisor._push_audio = _skip      # type: ignore[method-assign]
        supervisor._push_avatar_video = _skip  # type: ignore[method-assign]

    started = time.perf_counter()
    error = ""
    response = None
    try:
        response = await supervisor.handle_turn(StudentTurn(text=case["question"]))
    except Exception as err:  # noqa: BLE001 - 单题失败不该中断整轮评测
        error = f"{type(err).__name__}: {err}"
    elapsed = time.perf_counter() - started

    try:
        await supervisor.on_session_end(state)
    except Exception:  # noqa: BLE001 - 收尾失败无所谓
        pass

    rule_score, notes = score_by_rules(response)
    judge = await score_by_judge(case, response) if (use_judge and not error) else None
    judge_score = float(judge["score"]) if isinstance(judge, dict) and "score" in judge else None
    final = round((rule_score + judge_score) / 2, 1) if judge_score is not None else rule_score

    return {
        "id": case["id"],
        "subject": case.get("subject", ""),
        "question": case["question"],
        "rule_score": rule_score,
        "judge_score": judge_score,
        "judge": judge,
        "final_score": final,
        "notes": notes,
        "error": error,
        "elapsed_ms": round(elapsed * 1000, 0),
        "speech": str(getattr(response, "speech", "") or "")[:200],
        "board_ops": len(getattr(response, "board_ops", None) or []) if response else 0,
    }


# ---------------------------------------------------------------------- 汇总
def summarize(rows: list[dict[str, Any]], *, use_judge: bool) -> dict[str, Any]:
    """整轮汇总：总分 + 分学科 + 失败清单 + 耗时。"""
    finals = [r["final_score"] for r in rows]
    rules = [r["rule_score"] for r in rows]
    judges = [r["judge_score"] for r in rows if r["judge_score"] is not None]
    lat = [r["elapsed_ms"] for r in rows if r["elapsed_ms"]]

    by_subject: dict[str, list[float]] = {}
    for r in rows:
        by_subject.setdefault(r["subject"] or "综合", []).append(r["final_score"])

    weak = sorted(rows, key=lambda r: r["final_score"])[:3]
    return {
        "total": round(statistics.fmean(finals), 1) if finals else 0.0,
        "rule_avg": round(statistics.fmean(rules), 1) if rules else 0.0,
        "judge_avg": round(statistics.fmean(judges), 1) if judges else None,
        "judge_enabled": use_judge,
        "cases": len(rows),
        "errors": sum(1 for r in rows if r["error"]),
        "degraded": sum(1 for r in rows if "JSON 降级" in r["notes"]),
        "by_subject": {
            k: round(statistics.fmean(v), 1)
            for k, v in sorted(by_subject.items(), key=lambda kv: statistics.fmean(kv[1]))
        },
        "latency_p50_ms": round(statistics.median(lat), 0) if lat else None,
        "weakest": [
            {"id": r["id"], "score": r["final_score"], "notes": r["notes"][:3]}
            for r in weak
        ],
    }


# ---------------------------------------------------------------------- 主流程
async def do_run(args: argparse.Namespace) -> int:
    from config.settings import settings

    cases = load_cases()
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        cases = [c for c in cases if c["id"] in wanted]
        if not cases:
            raise SystemExit(f"--only 没有匹配到任何用例：{args.only}")

    # 评测期隔离：关掉自我学习（省两次调用/题），把裁决落到本轮目录（不污染 data/）
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = RUNS_DIR / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    keep = {
        "realtime_voice_enable": settings.realtime_voice_enable,
        "skill_self_update_enable": settings.skill_self_update_enable,
        "memory_distill_enable": settings.memory_distill_enable,
        "verdict_dir": settings.verdict_dir,
        # 答疑链路同样**必须钉住**：``BOARD_FIRST_MODE`` 决定走「板书先行」还是
        # 「常规答疑管线」，两条链路产出的结构不同。不钉住就会出现
        # 「本机测的是 A 路径、云端跑的是 B 路径」—— 本机全绿不代表云端没退化。
        # 实测过一次：同一份代码同一道题，本机（.env=single_call）规则分 100，
        # 云端（.env 未设 -> off）规则分 35。基线必须来自同一条路径才有可比性。
        "board_first_mode": settings.board_first_mode,
    }
    settings.realtime_voice_enable = False      # 评测针对「结构化答疑」这条主路径
    settings.skill_self_update_enable = False   # 别把评测里的东西学进真实经验库
    settings.memory_distill_enable = False
    settings.board_first_mode = "off"           # 见上：钉成生产用的那条，保证可复现、可跨机比较
    settings.verdict_dir = str(run_dir / "verdicts")

    print(f"评测集：{len(cases)} 题   {'（含裁判）' if args.judge else '（仅规则分，免费）'}")
    print(f"结果目录：{run_dir}")
    print("-" * 72)

    rows: list[dict[str, Any]] = []
    try:
        for i, case in enumerate(cases, 1):
            row = await run_case(case, use_judge=args.judge, speak=args.speak)
            rows.append(row)
            flag = "✗" if row["error"] else ("!" if row["final_score"] < JUDGE_SUBSCORE_FLOOR else "✓")
            judge_txt = f"  裁判 {row['judge_score']:>5.1f}" if row["judge_score"] is not None else ""
            print(
                f"  [{i:>2}/{len(cases)}] {flag} {row['id']:<24} "
                f"规则 {row['rule_score']:>5.1f}{judge_txt}  合计 {row['final_score']:>5.1f}"
                f"  {row['elapsed_ms']:>6.0f}ms"
            )
            if row["notes"]:
                print(f"         ↳ {'、'.join(row['notes'][:4])}")
    finally:
        for key, val in keep.items():
            setattr(settings, key, val)

    summary = summarize(rows, use_judge=args.judge)
    result = {"stamp": stamp, "summary": summary, "rows": rows}
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    print("-" * 72)
    print(f"总分 {summary['total']}   规则均分 {summary['rule_avg']}"
          + (f"   裁判均分 {summary['judge_avg']}" if summary["judge_avg"] is not None else ""))
    print(f"异常 {summary['errors']} 题   JSON 降级 {summary['degraded']} 题   "
          f"耗时中位 {summary['latency_p50_ms']}ms")
    if summary["weakest"]:
        print("最弱三题：")
        for w in summary["weakest"]:
            print(f"  {w['score']:>5.1f}  {w['id']}  {'、'.join(w['notes'])}")
    print(f"\n结果已落盘：{run_dir / 'result.json'}")

    # L5 实验记录：**在 promote 之前**写这一行 —— 要记的是「相对**旧**基线差多少」，
    # 那才是有意义的数；promote 之后再写就恒等于 0 了。
    append_results_tsv(
        result, kind=str(getattr(args, "cmd", "run")), note=getattr(args, "note", "")
    )

    # ``promote`` 只有 ``run`` 子命令才定义：``gate`` 会复用本函数（do_gate -> do_run），
    # 而它的 namespace 里没有这个属性。写成 ``args.promote`` 会让门禁跑完 12 题后
    # **崩在最后一步** —— 看起来像门禁「跑过了」，其实每次都是 traceback 退出，
    # 退出码根本没起到守门作用（本脚本自己的第一次实跑就是这么暴露的）。
    if getattr(args, "promote", False):
        save_baseline(result)
        print(f"已设为基线：{BASELINE_PATH}")
    return 0


def save_baseline(result: dict[str, Any]) -> None:
    """把一轮结果写成基线文件（只留汇总 + 逐题分数，便于人读 diff）。"""
    payload = {
        "stamp": result["stamp"],
        "summary": result["summary"],
        "per_case": {r["id"]: r["final_score"] for r in result["rows"]},
    }
    BASELINE_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def latest_run() -> Path:
    """最近一次运行目录。"""
    if not RUNS_DIR.exists():
        raise SystemExit(f"还没有任何评测结果（{RUNS_DIR} 不存在），先跑 run")
    runs = sorted((p for p in RUNS_DIR.iterdir() if (p / "result.json").exists()), key=lambda p: p.name)
    if not runs:
        raise SystemExit(f"还没有任何评测结果，先跑 run")
    return runs[-1]


async def do_gate(args: argparse.Namespace) -> int:
    """回归门禁：跑一遍 → 与基线比 → 掉分超阈值即非零退出。"""
    if not BASELINE_PATH.exists():
        raise SystemExit(
            f"没有基线（{BASELINE_PATH}）。先跑一次：\n"
            f"    python scripts/eval_agent.py run {'--judge' if args.judge else '--no-judge'} --promote"
        )
    base = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    base_total = float(base["summary"]["total"])

    rc = await do_run(args)

    run_dir = latest_run()
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    cur_total = float(result["summary"]["total"])

    # 裁判模式必须一致才能比总分：带裁判时总分是「规则与裁判各半」，
    # 免费模式只有规则分。两者混比会得出**假的结论** —— 免费模式天然高 4~5 分，
    # 拿它去比带裁判的基线必然「通过」，门禁就废了。模式不同时退化成只比规则均分：
    # 两边都有这个数，而它本来就是「结构退化」的主要信号。
    base_judged = bool(base["summary"].get("judge_enabled"))
    if base_judged != bool(args.judge):
        base_cmp = float(base["summary"]["rule_avg"])
        cur_cmp = float(result["summary"]["rule_avg"])
        label = "规则均分"
        print("⚠️  裁判模式与基线不同"
              f"（基线{'开启' if base_judged else '关闭'} / 本次{'开启' if args.judge else '关闭'}）——"
              f"总分算法不同，改为只比 {label}")
    else:
        base_cmp, cur_cmp, label = base_total, cur_total, "总分"
    delta = round(cur_cmp - base_cmp, 1)

    print("=" * 72)
    print(f"回归门禁：基线 {base_cmp}（{base['stamp']}）  →  本次 {cur_cmp}   差 {delta:+.1f}   [{label}]")

    # 逐题对比：定位「哪几道题掉了」——只看总分不知道去改什么
    per_case = {r["id"]: r["final_score"] for r in result["rows"]}
    dropped = [
        (cid, base["per_case"][cid], per_case[cid])
        for cid in per_case
        if cid in base.get("per_case", {}) and per_case[cid] < base["per_case"][cid] - GATE_TOLERANCE
    ]
    if dropped:
        print("掉分的题：")
        for cid, was, now in sorted(dropped, key=lambda x: x[2] - x[1]):
            print(f"  {was:>5.1f} → {now:>5.1f}   {cid}")

    failed = delta < -GATE_TOLERANCE
    print("-" * 72)
    if failed:
        print(f"❌ 门禁未通过：总分下跌 {abs(delta)} 分（容差 {GATE_TOLERANCE}）")
    else:
        print(f"✅ 门禁通过（容差 {GATE_TOLERANCE} 分内）")
    print(f"若确认本次退化可接受，用这条更新基线：")
    print(f"    python scripts/eval_agent.py promote --from-run {run_dir.name}")
    return 1 if failed else max(rc, 0)


def do_promote(args: argparse.Namespace) -> int:
    """Promote：把指定（默认最近一次）结果设为基线。"""
    run_dir = RUNS_DIR / args.from_run if args.from_run else latest_run()
    path = run_dir / "result.json"
    if not path.exists():
        raise SystemExit(f"找不到结果文件：{path}")
    save_baseline(json.loads(path.read_text(encoding="utf-8")))
    print(f"基线已更新：{BASELINE_PATH}")
    return 0


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(description="AI 教师质量评测 + 回归门禁（L4）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--judge", dest="judge", action="store_true", default=True,
                       help="启用裁判分（多花每题一次 LLM 调用）")
        p.add_argument("--no-judge", dest="judge", action="store_false",
                       help="只用规则分（免费、快）")
        p.add_argument("--speak", action="store_true",
                       help="含 TTS 与数字人渲染（默认跳过，只评文本质量）")
        p.add_argument("--only", default="", help="只跑指定用例，逗号分隔的 id")
        p.add_argument("--note", default="",
                       help="这一轮的说明，写进 evals/results.tsv（例：--note '接入 L2 bandit'）")

    p_run = sub.add_parser("run", help="跑一遍评测")
    add_common(p_run)
    p_run.add_argument("--promote", action="store_true", help="跑完顺便设为基线")

    p_gate = sub.add_parser("gate", help="回归门禁（低于基线即失败）")
    add_common(p_gate)

    p_pro = sub.add_parser("promote", help="把结果设为基线")
    p_pro.add_argument("--from-run", default="", help="指定运行目录名（默认最近一次）")

    args = parser.parse_args()
    if args.cmd == "run":
        return asyncio.run(do_run(args))
    if args.cmd == "gate":
        return asyncio.run(do_gate(args))
    return do_promote(args)


if __name__ == "__main__":
    raise SystemExit(main())
