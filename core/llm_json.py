"""从模型输出里抠出一个 JSON 对象 —— 一处实现，多处方用。

为什么值得单独一个模块
----------------------
「让模型输出 JSON，然后从它的回复里把 JSON 抠出来」在这类项目里会出现很多次
（根因分析、经验归并……）。每处各写一遍的后果不是「多几行代码」，而是
**各处对边界的容忍度不一样**：一处能处理 markdown 代码块、另一处不能，
于是同一个模型在 A 功能里正常、在 B 功能里报「没给出可用的 JSON」。
这种分叉很难查 —— 因为两处各自跑起来都不报错。

所以：解析逻辑只此一份，两个失败模式都在这里兜住。
"""
from __future__ import annotations

from typing import Any

__all__ = ["parse_json_object"]


def parse_json_object(raw: Any) -> dict[str, Any] | None:
    """抠出**第一个完整的 JSON 对象**；抠不到返回 ``None``（不抛）。

    容忍三种常见的模型输出噪声：

    * 外面套 ``markdown`` 代码块（``​```json … ```​``）；
    * 前后夹带解释文字（「好的，我的分析如下：{…}」）；
    * JSON 之后又跟了一段示例 —— 这时「第一个 ``{`` 到最后一个 ``}``」会把两段
      一起圈进去导致解析失败，所以要退到**括号配对**去找第一个配平的对象。

    先整体试、失败再配对扫，是为了**兼容既有行为**：绝大多数情况下模型是
    ``json_mode=True`` 直接给纯 JSON，第一段就成功，行为与改动前完全一致。
    """
    text = str(raw or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        import json  # noqa: PLC0415 - 只在需要时导入

        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    return _first_balanced_object(text)


def _first_balanced_object(text: str) -> dict[str, Any] | None:
    """括号配对扫一遍，返回**第一个配平**的 JSON 对象。

    必须跳过字符串字面量里的花括号 —— 教学话术里出现 ``{"text": "用 {x} 表示"}``
    这种内容很常见，不跳的话计数会错位，抠出来的片段不是合法 JSON。
    """
    import json  # noqa: PLC0415

    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        parsed = None
                    if isinstance(parsed, dict):
                        return parsed
                    break  # 这个起点抠不出合法对象，换下一个起点
        start = text.find("{", start + 1)
    return None
