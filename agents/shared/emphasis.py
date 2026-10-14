"""重点标记：讲解文字里「哪几个字要标红」的统一约定与解析。

为什么用「文字内标记」而不是另开一个字段
--------------------------------------
重点是**片段**，不是整句属性。模型写「移项要变号」时，只有「变号」两字该标红；
若另开 ``key_points: [...]`` 字段，前端没法知道它落在句子里的哪个位置，只能整行变红
—— 整行红等于没有重点。

标记语法用 Markdown 粗体 ``**...**``：

* 模型最熟悉（几乎不用教就会写），token 最省；
* 失败代价极小：没写标记 = 不标红，文字照常可读；写错了也只多两个星号；
* 不必给每个输出字段都加结构，前端一处解析就够。

两个必须守住的性质
------------------
1. **绝不进语音**：``**`` 被 TTS 读出来会很怪，所以播报/聊天文本一律先
   :func:`strip_emphasis` 再下发（见 ``Supervisor._spoken_text`` 与 ``TeacherAgent``）；
2. **绝不进 HTML**：前端解析时只拼 DOM 节点（``createTextNode`` / ``<span>``），
   不用 innerHTML —— 模型产出属于不可信文本。
"""
from __future__ import annotations

import re

#: 重点标记（与前端 ``renderText`` 里的解析保持一致）
KEY_MARK = "**"

_KEY_RE = re.compile(r"\*\*(.+?)\*\*", re.S)


def strip_emphasis(text: str) -> str:
    """去掉重点标记，只留文字（语音 / 聊天文本用）。

    未配对的多余 ``**`` 也一并去掉：模型偶尔只写一个，留在白板上就是两个显眼的星号。
    """
    value = str(text or "")
    if KEY_MARK not in value:
        return value
    cleaned = _KEY_RE.sub(lambda m: m.group(1), value)
    return cleaned.replace(KEY_MARK, "")


def split_emphasis(text: str) -> list[tuple[str, bool]]:
    """把文字切成 ``[(片段, 是否重点)]``，供测试与前端对齐（前端用同一套规则解析）。"""
    value = str(text or "")
    if KEY_MARK not in value:
        return [(value, False)] if value else []
    out: list[tuple[str, bool]] = []
    cursor = 0
    for match in _KEY_RE.finditer(value):
        if match.start() > cursor:
            plain = value[cursor:match.start()].replace(KEY_MARK, "")
            if plain:
                out.append((plain, False))
        out.append((match.group(1), True))
        cursor = match.end()
    tail = value[cursor:].replace(KEY_MARK, "")
    if tail:
        out.append((tail, False))
    return out


def has_emphasis(text: str) -> bool:
    """这段文字里是否含有重点标记（排障用：可确认模型到底有没有按契约标注）。"""
    return bool(_KEY_RE.search(str(text or "")))
