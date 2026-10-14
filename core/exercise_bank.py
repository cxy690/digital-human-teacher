"""真题库只读访问层：数字人出题的**真题来源**（``ai_teacher.exercise_item``）。

背景：
    教研侧已把初中数学真题（29k+ 道，7/8/9 年级，来源为教辅 docx 解析）结构化入
    MySQL 库 ``ai_teacher.exercise_item``（见 rag/online/config.yaml 的同名库）。
    出题链路（:meth:`agents.teacher.agent.TeacherAgent.generate_quiz`）**优先从真题库选题**，
    库不可用 / 无匹配时回退 LLM 即兴生成 —— 与项目「可选组件缺失即降级」的哲学一致，
    绝不因题库故障阻断课堂。

设计要点：
* **只读**：只发 SELECT（与 rag/online/store.py 同一铁律：写库只发生在离线链路）；
* **质量过滤**：默认只取 ``has_answer=1``（有标准答案才能判分）、
  ``needs_human_review=0 AND suspicious=0``（跳过质检存疑题，``EXERCISE_BANK_ALLOW_REVIEW`` 可放宽）；
  题型默认只取 choice/fill（solve/other 的答案形态不适合自动判分，见 data/stores 题表统计）；
* **全降级**：连接、查询、转换任一步异常 → 返回 ``None``，由调用方回退 LLM，**绝不抛**；
* **去重**：进程级 FIFO 记住最近发出过的 ``exercise_id``，短期不重复出同一道题；
* **可测**：``set_picker()`` 注入假题源（模式同 ``rag/online/embedder.set_encoder``），
  单测无需连真库。
"""
from __future__ import annotations

import json
import logging
import re
from collections import deque
from typing import Any, Callable

from config.settings import settings

log = logging.getLogger("core.exercise_bank")

#: 最近发出过的题目 id（进程级；防同一道题短期内重复发给同一个/其它学生）
_RECENT_IDS: deque[str] = deque(maxlen=30)

#: 惰性连接（pymysql），失败后允许下次重连
_CONN: Any | None = None

#: 测试注入的假题源：签名与 :func:`pick` 相同
_PICKER: Callable[..., dict[str, Any] | None] | None = None


# ---------------------------------------------------------------------------- 年级
_CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def normalize_grade(text: str) -> int | None:
    """把各种写法的年级归一成初中学段数字（7/8/9）；无法判定返回 ``None``。

    支持的写法（登录用户 ``users.grade`` 与 ``session.scratch["grade"]`` 的实测值）::

        "初一" / "初二" / "初三"          -> 7 / 8 / 9
        "七年级" / "八年级下册" / "九年级" -> 7 / 8 / 9（初中教材口径：七年级=初一）
        "7" / "8" / "9"                  -> 7 / 8 / 9

    刻意不臆测的情况（返回 ``None``，调用方按「不限年级」处理）::

        "" / "初中" / "小学一年级到初三"   —— 范围表达（含「到/~/～」）一律不猜；
        "一年级"（小学）等其它学段        —— 题库只覆盖初中，不硬凑。
    """
    t = str(text or "").strip()
    if not t:
        return None
    if any(ch in t for ch in ("到", "~", "～")):
        return None
    m = re.search(r"初([一二三])", t)
    if m:
        return {"一": 7, "二": 8, "三": 9}[m.group(1)]
    m = re.search(r"([七八九])年级", t)
    if m:
        return _CN_NUM[m.group(1)]
    m = re.search(r"(?<!\d)([789])(?!\d)", t)
    if m:
        return int(m.group(1))
    return None


def is_supported_subject(subject: str) -> bool:
    """题库当前只收录数学（数据源皆为初中数学教辅）；其它学科直接走 LLM 出题。"""
    s = str(subject or "").strip().lower()
    return s in ("数学", "math")


# ---------------------------------------------------------------------------- 查询
def _connect() -> Any:
    """惰性建立并复用一条只读连接（``ping(reconnect=True)`` 抗 MySQL 空闲断连）。"""
    global _CONN
    import pymysql  # 延迟导入：未装 pymysql 的环境不受影响
    from sqlalchemy.engine import make_url

    url = make_url(settings.exercise_bank_url)
    if _CONN is None:
        _CONN = pymysql.connect(
            host=url.host or "127.0.0.1",
            port=url.port or 3306,
            user=url.username or "root",
            password=url.password or "",
            database=url.database or "ai_teacher",
            charset=(url.query or {}).get("charset", "utf8mb4"),
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True,
        )
    else:
        _CONN.ping(reconnect=True)
    return _CONN


def _like_escape(text: str) -> str:
    """转义 LIKE 通配符（topic 来自会话上下文，防止 % / _ 把匹配放大）。"""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _allowed_types() -> list[str]:
    """settings 中配置的允许题型（逗号分隔），只保留认识的四个值。"""
    known = ("choice", "fill", "solve", "other")
    got = [t.strip() for t in str(settings.exercise_bank_types or "").split(",")]
    return [t for t in got if t in known] or ["choice", "fill"]


#: 候选集大小：库里部分题的 answer_text 是解析段（错位）或题面残缺，
#: 单条直取命中率低；取一小批候选在 Python 侧逐条质检，取第一条合格者。
_CANDIDATE_LIMIT = 12

#: 填空题答案里的噪声词（解析引用而非答案本身）
_ANSWER_NOISE = {"略", "见解析", "答案见解析", "略。"}


def _answer_ok(row: dict[str, Any]) -> dict[str, Any] | None:
    """答案形态质检（**数据质量推断规则，不是业务逻辑**）：

    实测定性：``choice`` 题里只有约 1/3 的 ``answer_text`` 是单字母（其余是解析段错位）；
    ``fill`` 题答案应为短短语（长答案/含「（N分）」的是解析开头）。
    不合格返回 ``None``（换下一条候选），合格则把答案**归一后**写回行内返回。

    规则刻意偏保守：宁可少用几道题，也不给学生发错答案——判分错了比不出题伤得多。
    """
    ans = str(row.get("answer_text") or "").strip()
    qtype = str(row.get("question_type") or "").strip().lower()
    if not ans:
        return None
    if "⟦" in ans or "⟧" in ans:
        return None  # 公式占位符（⟦Fxx⟧）：判分无意义（SQL 侧已粗筛，这里兜底）
    if qtype == "choice":
        m = re.fullmatch(r"[（(]?\s*([A-Ha-h])\s*[）)。．、]?", ans)
        if not m:
            return None
        row["answer_text"] = m.group(1).upper()   # 「（D）」→「D」
        return row
    if qtype == "fill":
        if len(ans) > 20 or "\n" in ans or ans in _ANSWER_NOISE:
            return None
        if re.search(r"[（(]\d+\s*分", ans):      # 「（9分）……」= 解析段错位
            return None
        return row
    # solve / other（默认不选题型；若配置开启则放宽到 60 字）
    return row if len(ans) <= 60 else None


def _topic_hits(rows: list[dict[str, Any]], topic: str) -> list[dict[str, Any]]:
    """只保留「题干命中主题词」的候选（``topic_hit`` 由 SQL 计算）。

    无主题词（topic 空）时原样返回——随机选题是合理的（泛化练习）。
    有主题词却**一道都没命中**时宁可返回空（``pick`` 返回 None，调用方回退
    LLM 按主题即兴出题），也不随机抽一道无关的真题 —— 实测「讲配方法时
    抽到同位角的题」就是这里随机抽样造成的驴唇不对马嘴。
    """
    if not topic:
        return rows
    return [r for r in rows if r.get("topic_hit")]


def _query_candidates(grade_num: int | None, topic: str) -> list[dict[str, Any]]:
    """按（年级, 主题命中, 纯文本优先, 随机）取一批候选；年级无命中时放宽到全年级。

    ``topic`` 非空时**只返回主题命中的题**（见 :func:`_topic_hits`）；
    没有命中就返回空列表让上层回退 LLM 出题，而不是随机抽无关题。
    """
    types = _allowed_types()
    quality = "" if settings.exercise_bank_allow_review else " AND needs_human_review = 0 AND suspicious = 0"
    # 注意：pymysql 用 ``query % args`` 做参数替换，SQL 里字面量 ``%`` 必须写 ``%%``
    # （「%如图%」写成裸 % 会抛 "unsupported format character"）
    figure = " AND stem_plain NOT LIKE '%%如图%%'" if settings.exercise_bank_skip_figure else ""
    # 答案含公式占位符（⟦Fxx⟧，docx 解析时公式被剥离留下的占位）→ 判分无意义，SQL 侧先粗筛
    # （实测 choice 364 条 / fill 936 条），Python 侧 ``_answer_ok`` 再兜一道
    no_placeholder = " AND answer_text NOT LIKE '%%⟦%%' AND stem_plain NOT LIKE '%%⟦%%'"
    exclude = [i for i in _RECENT_IDS if i]
    ph = ", ".join(["%s"] * len(exclude))
    pattern = f"%{_like_escape(topic)}%" if topic else "%"

    def run(grade_cond: str, params: list[Any]) -> list[dict[str, Any]]:
        sql = (
            # topic_hit：题干是否命中主题词（与 ORDER BY 同一表达式，供 _topic_hits 过滤）
            "SELECT exercise_id, grade, question_type, stem_plain, options, answer_text, "
            "(stem_plain LIKE %s) AS topic_hit "
            "FROM exercise_item "
            "WHERE status = 'active' AND has_answer = 1 "
            "AND question_type IN (" + ", ".join(["%s"] * len(types)) + ") "
            "AND stem_plain IS NOT NULL AND stem_plain <> '' "
            "AND CHAR_LENGTH(stem_plain) <= %s"
            + quality
            + figure
            + no_placeholder
            + grade_cond
            + (f" AND exercise_id NOT IN ({ph})" if exclude else "")
            # 排序三段：主题命中优先（布尔→1/0）→ 纯文本题优先（公式剥离会让题面出空档）
            # → 随机。一条 SQL 完成「优先匹配、否则随机」，无需多次查询。
            + " ORDER BY (stem_plain LIKE %s) DESC, has_formula ASC, RAND() LIMIT %s"
        )
        full_params: list[Any] = [pattern, *types, int(settings.exercise_bank_max_stem_chars), *params]
        if exclude:
            full_params.extend(exclude)
        full_params.extend([pattern, _CANDIDATE_LIMIT])
        cur = _connect().cursor()
        try:
            cur.execute(sql, full_params)
            return [dict(r) for r in cur.fetchall()]
        finally:
            cur.close()

    if grade_num is not None:
        hits = _topic_hits(run(" AND grade = %s", [grade_num]), topic)
        if hits:
            return hits
        # 该年级没有主题命中的题（或被质量过滤/去重）→ 放宽年级，仍要求主题命中
    return _topic_hits(run("", []), topic)


def pick(
    *,
    subject: str,
    grade: str = "",
    topic: str = "",
    exclude: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    """选一道真题；返回 ``exercise_item`` 行（dict）或 ``None``（降级信号）。

    参数：
        subject: 学科中文名（或 key）；非数学直接返回 ``None``（题库未覆盖）；
        grade:   年级原文（``"初一"`` / ``"7"`` / ``""`` 均可，内部归一化）；
        topic:   当前知识点（如 ``"三角形内角和"``）；命中的题优先，无命中随机；
        exclude: 额外排除的题目 id（叠加在进程级最近去重之上）。

    任何异常（未部署题库 / 连不上 / 查询失败）都记一条 warning 后返回 ``None``。
    """
    if not settings.exercise_bank_enable:
        return None
    if not is_supported_subject(subject):
        return None  # 题库只覆盖数学：契约先于假题源，与实现无关
    try:
        for i in exclude:
            if i:
                _RECENT_IDS.append(str(i))
        if _PICKER is not None:  # 测试注入的假题源
            return _PICKER(subject=subject, grade=grade, topic=topic, exclude=exclude)
        for row in _query_candidates(normalize_grade(grade), str(topic or "").strip()):
            ok = _answer_ok(row)
            if ok is not None:
                _RECENT_IDS.append(str(ok.get("exercise_id") or ""))
                return ok
        return None
    except Exception as err:  # noqa: BLE001 - 题库是可选组件：连不上/查不了都不能阻断课堂
        log.warning("真题库不可用（回退 LLM 出题）：%s: %s", type(err).__name__, err)
        return None


# ---------------------------------------------------------------------------- 转换
def _clean_stem(stem: str) -> str:
    """题干清洗：折叠换行与多余空白（换行进 TTS/白板都不友好，归一为空格）。"""
    return re.sub(r"\s+", " ", str(stem or "")).strip()


def _parse_options(raw: Any) -> list[str]:
    """``options`` JSON → 展示用选项文本列表（**不带** "A．" 前缀）。

    实测两种形态（见库里抽样）：
    * ``[{"k": "A", "v": "嫌疑犯A"}, ...]`` → ``["嫌疑犯A", "嫌疑犯B", ...]``
      （字母由前端渲染时自动加：``String.fromCharCode(65+i) + ". " + opt``，
      后端再拼一次会显示成 "A. A．嫌疑犯A"）；
    * ``null`` —— 这类题的选项**内嵌在题干里**（``"…的是(   )\\nA．…\\nB．…"``），
      题干原样保留即可，返回空列表由前端/播报按题干读。
    """
    if not raw:
        return []
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return []
    out: list[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                v = str(item.get("v") or "").strip()
                if v:
                    out.append(v)
            elif str(item).strip():
                out.append(str(item).strip())
    return out


def to_quiz_fields(row: dict[str, Any], *, topic: str = "") -> dict[str, Any]:
    """``exercise_item`` 行 → :meth:`TeacherAgent._build_quiz` 期望的字段 dict。

    题型映射：``choice→choice``、``fill→blank``、``solve/other→short``。
    答案直接取 ``answer_text``（已过滤 ``has_answer=1``，选择题多为 "A"/"D"、填空题为短语，
    都适合 ``evaluate_answer`` 的 sympy/LLM/字符串比对判分）。
    """
    qtype = str(row.get("question_type") or "").strip().lower()
    mapped = {"choice": "choice", "fill": "blank", "solve": "short", "other": "short"}.get(qtype, "short")
    options = _parse_options(row.get("options"))
    focus = topic or str(row.get("grade") or "")
    return {
        "type": mapped,
        "question": _clean_stem(row.get("stem_plain") or ""),
        "options": options,
        "answer": str(row.get("answer_text") or "").strip(),
        "hint": f"别急，先圈出题目里的条件，再想想刚才讲的「{focus}」相关的知识。",
    }


# ---------------------------------------------------------------------------- 测试辅助
def set_picker(fn: Callable[..., dict[str, Any] | None] | None) -> None:
    """注入假题源（签名同 :func:`pick`）；传 ``None`` 复原真实查询。"""
    global _PICKER
    _PICKER = fn


def reset_for_tests() -> None:
    """测试收尾：清空去重 FIFO 与连接（不参与生产路径）。"""
    global _CONN
    _RECENT_IDS.clear()
    if _CONN is not None:
        try:
            _CONN.close()
        except Exception:  # noqa: BLE001
            pass
    _CONN = None
