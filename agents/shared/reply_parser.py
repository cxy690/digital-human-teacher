"""LLM 回复的 JSON 归一化：答疑与讲课两条链路共用的解析口。

单独成模块的原因：``TeacherAgent`` 与 ``LecturerAgent`` 都要把模型返回的 JSON 文本
解析成 dict，而两者的契约字段完全不同。解析逻辑只做「把 JSON 抠出来」这一件事，
不感知任何字段 —— 字段解释留在各自的 agent 里，避免解析层跟着契约一起漂移。
"""
from __future__ import annotations

import json
import re
from typing import Any

_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")


def _FIELD_RE(name: str) -> re.Pattern[str]:
    """``"speech": "…"`` 的取值正则（按需编译并缓存）。"""
    cached = _FIELD_CACHE.get(name)
    if cached is None:
        cached = re.compile(r'"' + re.escape(name) + r'"\s*:\s*"((?:[^"\\]|\\.)*)"', re.S)
        _FIELD_CACHE[name] = cached
    return cached


_FIELD_CACHE: dict[str, re.Pattern[str]] = {}


def parse_json_reply(text: str) -> dict[str, Any]:
    """从模型回复中截取第一个 ``{...}`` 并解析；失败返回空 dict（调用方走兜底）。"""
    match = _JSON_BLOCK.search(text or "")
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def looks_like_json(text: str) -> bool:
    """这段回复**看起来**是想给 JSON 契约吗？

    用途：解析失败时区分两种情形 ——
    * 模型正常给了散文（就该按散文兜底）；
    * 模型想给 JSON 但结构写坏了（**绝不能再拿原文当话术**，那会把 JSON 念出来）。
    """
    stripped = (text or "").lstrip()
    return stripped.startswith("{") or '"speech"' in (text or "")


def _read_json_string(text: str, start: int) -> tuple[str, int]:
    """读一个 JSON 字符串（``start`` 指向起始引号），返回 ``(值, 下一个位置)``。"""
    i = start + 1
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == '"':
            raw = text[start:i + 1]
            try:
                return json.loads(raw), i + 1
            except json.JSONDecodeError:
                return raw[1:-1], i + 1
        i += 1
    return text[start + 1:], n          # 引号没闭合：剩下全当值


def _extract_string_list(text: str, name: str) -> list[str]:
    """按字段名抠出一个「字符串数组」，**嵌套数组直接摊平**。

    为什么这样写：实测的坏法就是 ``"steps": ["a", ["b", "c", ...`` ——
    模型多开了一个 ``[`` 又没闭合，后面直接接了下一个字段名。
    扫描时只要发现「某个字符串后面跟着 ``:``」，那它是**下一个字段的键**，
    说明当前数组没闭合，就地收工 —— 于是不用修语法也能把整列步骤捞回来。
    """
    found = _ARRAY_START(name).search(text or "")
    if not found:
        return []
    i = found.end()                             # 已越过字段名、冒号与 '['
    n = len(text)
    depth = 0
    items: list[str] = []
    while i < n:
        ch = text[i]
        if ch == "[":
            depth += 1
            i += 1
            continue
        if ch == "]":
            if depth == 0:
                break
            depth -= 1
            i += 1
            continue
        if ch == '"':
            value, i = _read_json_string(text, i)
            j = i
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] == ":":        # 这是下一个字段的键名 -> 数组没闭合
                break
            if value.strip():
                items.append(value.strip())
            continue
        i += 1
    return items


def _ARRAY_START(name: str) -> re.Pattern[str]:      # noqa: N802 - 与下面的缓存配套，读起来像常量
    """``"steps": [`` 的开头正则（按需编译并缓存）。"""
    cached = _ARRAY_CACHE.get(name)
    if cached is None:
        cached = re.compile(r'"' + re.escape(name) + r'"\s*:\s*\[', re.S)
        _ARRAY_CACHE[name] = cached
    return cached


_ARRAY_CACHE: dict[str, re.Pattern[str]] = {}

#: 契约里的字符串标量字段（抢救时按顺序找）
_SCALAR_FIELDS: tuple[str, ...] = ("speech", "title", "equation", "answer")
#: 契约里的「字符串数组」字段（嵌套数组会被摊平）
_LIST_FIELDS: tuple[str, ...] = ("steps", "knowledge_points", "pitfalls")


def salvage_object(text: str) -> dict[str, Any]:
    """从**结构损坏**的 JSON 里按字段名把内容抠出来（确定性抢救，不再花模型调用）。

    只认「字符串标量」与「字符串数组」两类字段 —— 实测坏的就是数组括号；
    ``followup`` 那种嵌套对象不在这里还原（没有它只是不追问，不影响讲解）。

    为什么优先用它而不是「让模型重发一遍合法 JSON」：重发同样是 1000+ 字的长输出，
    实测**照样会写坏**（日志里出现过「修复后 1023 字仍然损坏」），
    而这里的扫描是纯本地的、必然成功。
    """
    out: dict[str, Any] = {}
    for name in _SCALAR_FIELDS:
        value = salvage_text_field(text, name)
        if value:
            out[name] = value
    for name in _LIST_FIELDS:
        items = _extract_string_list(text, name)
        if items:
            out[name] = items
    return out


def salvage_text_field(text: str, name: str = "speech") -> str:
    """从**结构已坏**的 JSON 里抢救出一个字符串字段。

    实测触发场景：长篇讲解里模型把 ``"steps": ["a", ["b", …`` 多写了一个 ``[``，
    ``json.loads`` 直接失败。此时至少把 ``speech`` 这一句捞回来 —— 学生能听到一句
    正常的话，而不是一段 ``{"speech": …`` 的原文。
    """
    match = _FIELD_RE(name).search(text or "")
    if not match:
        return ""
    raw = match.group(1)
    try:
        return str(json.loads(f'"{raw}"')).strip()
    except json.JSONDecodeError:      # 连转义都坏了：手工还原最常见的两种
        return raw.replace("\\n", " ").replace('\\"', '"').strip()
