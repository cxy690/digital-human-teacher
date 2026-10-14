"""L5 治理层：用离线模型找出「说法不同、意思相同」的经验。

为什么字面相似度够不着
----------------------
已经有一套字面判定（``review.SIMILARITY_FOR_REVIEW``，免费、快、每次抽检都跑），
但它有**实测出来的天花板**。云端真实数据里这一对：

    学生常只对一边运算，或把未跨等号的项也变号，需强调只有跨过等号的项才变号
    学生常只对含未知数的一边操作，忘记另一边同步变化，需强调等式如天平两边同动

谁都看得出是一件事，字面相似度只有 **0.348** —— 而彼此无关的条目是 **0.07**，
中间几乎没有可用的判别区间（用词全换了：「一边运算」→「一边操作」、
「变号」→「同步变化」）。这一类只能靠**语义**。本模块补的就是这一段。

为什么是「离线 LLM」而不是 embedding
------------------------------------
embedding 更便宜，但本项目当前环境跑的是**词法检索**（bigram），语义向量服务在
低配 / 无卡时起不来。而**离线 LLM 判定**与项目既有做法一致
（``core.failure_cluster.analyze_cluster`` 的根因分析就是同一种模式）：
只在**被人显式调用**时跑，一次处理一位老师的几十条经验，绝不挂在答疑主链路上。

⚠️ 它只给建议，**不自动停用**
------------------------------
归并错了会丢掉一个独立的教学要点 —— 那是不可逆的信息损失。所以这里只产出
「哪几条像是一件事 + 建议留哪条」，由人在抽检台确认后再 ``retire``。
和 L5 其它闸门一致：**降低人的判断成本，不替人做决定。**

这也决定了它的返回形状：``groups`` 里每一条都带 ``keep`` / ``drop`` / ``why``，
是**给人看的建议**，不是给程序执行的指令。
"""
from __future__ import annotations

from typing import Any

from config.logging import get_logger
from core.llm_json import parse_json_object
from core.skills.experience.store import ExperienceStore, get_experience_store

log = get_logger(__name__)

#: 一次最多送多少条给模型。经验库上限不限（``skill_experience_max_items=0``），
#: 条数多时全塞进去既贵又会稀释注意力 —— 按「最值得看」的顺序截断（见 :func:`_pick`）。
MAX_ITEMS_PER_CALL = 60

_PROMPT = """你在维护一位 AI 教师的教学经验库。下面是它积累的经验，每条前面是编号。

请找出**在讲同一件事**的条目，按组输出。只有满足下面两条之一才算：

* **说法不同、意思相同** —— 例如「学生只对等式一边运算」与「学生只对含未知数的一边操作」；
* **一条的内容被另一条完全覆盖** —— 那么更宽的那条留着，窄的可以去掉。

**只在这两种情况下归组。** 宁可少报、不要多报 —— 每条经验都来自真实课堂的复盘，
误合并会永久丢掉一个独立的教学要点，代价远大于留着一条冗余。

每组的 ``keep`` 要选**信息最全**的那条，``drop`` 列出其余编号。

严格输出 JSON，不要加任何别的文字：

{"groups": [{"ids": ["第一条的id", "第二条的id"], "keep": "要留的那条id",
             "drop": ["要去掉的那条id"], "why": "一句话说明为什么是一件事"}]}

没有发现重复就输出 {"groups": []}
"""


async def find_duplicate_groups(
    teacher_key: str = "", *, store: ExperienceStore | None = None, llm: Any = None,
    limit: int = MAX_ITEMS_PER_CALL,
) -> dict[str, Any]:
    """找出「在讲同一件事」的经验分组（``teacher_key`` 为空则逐位老师各查一次）。

    返回 ``{"ok": True, "groups": [...], "scanned": {...}}``；
    失败返回 ``{"ok": False, "error": ...}`` —— **不抛**，排障工具自己报错会让人
    以为是系统坏了（与 ``analyze_cluster`` 同一取舍）。
    """
    book = store or get_experience_store()
    keys = [teacher_key.strip()] if (teacher_key or "").strip() else book.list_teachers()
    if not keys:
        return {"ok": True, "groups": [], "scanned": {}}

    client = llm
    if client is None:
        try:
            from models.llm import get_llm  # noqa: PLC0415 - 延迟导入，未配置模型时也能 import 本模块

            client = get_llm()
        except Exception as err:  # noqa: BLE001 - 拿不到模型就直说
            return {"ok": False, "error": f"取模型失败：{err}"}

    groups: list[dict[str, Any]] = []
    scanned: dict[str, int] = {}
    for key in keys:
        items = _pick(book, key, limit=limit)
        scanned[key] = len(items)
        if len(items) < 2:
            continue  # 一条无从比较
        try:
            raw = await client.chat(
                [
                    {"role": "system", "content": _PROMPT},
                    {"role": "user", "content": _render(items)},
                ],
                temperature=0.2,
                max_tokens=1200,
                json_mode=True,
            )
        except Exception as err:  # noqa: BLE001 - 一位老师失败不该让整轮失败
            log.warning("经验归并调用失败（{}）：{}", key, err)
            groups.append({"teacher": key, "error": f"{type(err).__name__}: {err}"})
            continue
        groups.extend(_sanitize(parse_json_object(raw), key, items))

    return {"ok": True, "groups": groups, "scanned": scanned}


def _pick(book: ExperienceStore, key: str, *, limit: int) -> list[Any]:
    """挑出要送去比较的经验：只比**生效中**的（已停用的不占名额，归并无意义）。

    按「被注入次数多、被复盘提到多」优先 —— 影响面大的先比，
    条数超限时截掉的是最边缘的那些。
    """
    items = [it for it in book.all_items(key) if it.active]
    items.sort(key=lambda it: (-(it.pulls + it.hits), it.text))
    return items[: max(2, int(limit))]


def _render(items: list[Any]) -> str:
    """把经验渲染成给模型看的清单（带 id，好让它按 id 归组）。"""
    from core.skills.experience.models import KIND_LABELS  # noqa: PLC0415 - 循环导入规避

    lines = []
    for index, item in enumerate(items, 1):
        label = KIND_LABELS.get(item.kind, item.kind)
        lines.append(f"{index}. [{item.id}] （{label}）{item.text}")
    return "\n".join(lines)


def _sanitize(parsed: Any, key: str, items: list[Any]) -> list[dict[str, Any]]:
    """把模型输出收敛成**可执行性为零、可读性满分**的建议。

    四道校验，每一道都对应一种真实的模型失误：

    1. **id 必须真实存在** —— 模型会编 id（把相邻两条的编号拼错）；
    2. **一组至少 2 条** —— 单条不是「重复」；
    3. **``keep`` 必须在组内** —— 否则取第一条，不能让模型把两条都丢掉；
    4. **一条不能出现在两组** —— 否则「留哪条」互相矛盾。
    """
    valid = {it.id: it for it in items}
    groups: list[dict[str, Any]] = []
    used: set[str] = set()
    for raw in (parsed or {}).get("groups") or []:
        if not isinstance(raw, dict):
            continue
        ids = [str(i).strip() for i in (raw.get("ids") or []) if str(i).strip() in valid]
        ids = [i for i in dict.fromkeys(ids) if i not in used]
        if len(ids) < 2:
            continue
        keep = str(raw.get("keep") or "").strip()
        if keep not in ids:
            keep = ids[0]  # 模型指了组外 / 没给：保守取第一条
        drop = [str(d).strip() for d in (raw.get("drop") or [])]
        drop = [d for d in dict.fromkeys(drop) if d in ids and d != keep]
        if not drop:
            drop = [i for i in ids if i != keep]
        used.update(ids)
        groups.append(
            {
                "teacher": key,
                "ids": ids,
                "keep": keep,
                "drop": drop,
                "why": str(raw.get("why") or "").strip()[:160],
                "texts": {i: valid[i].text for i in ids},
            }
        )
    return groups


__all__ = ["MAX_ITEMS_PER_CALL", "find_duplicate_groups"]
