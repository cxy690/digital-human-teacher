# -*- coding: utf-8 -*-
"""查询改写（链路第 1 步，P4 两级实现）。

在检索前把用户原问题规范化/补全为一条更适合检索的 **独立 query**：

  1. **规则级（默认开，不联网）**：
     - 错别字/异体字词典归一（口径同 ``tools/transcript_normalize`` 的 ``*_ALIASES``：变体→规范；
       online 不 import tools 以守住"在线只读三存储"边界，故内置一份面向检索查询的小词典）；
     - 空白/标点归一（NFKC 统一全角数字/字母/空格、折叠多余空白、折叠重复句读）。
  2. **LLM 级（``rewrite.llm_enabled`` 可配开关，默认关，需 Ollama）**：
     指代消解与补全（"它怎么算" + 历史 → "有理数的加法怎么算"），输出改写后的独立 query。
  3. **语义漂移护栏**：改写前（原 query）后（候选）各编码一次，bge 余弦 < ``drift_threshold``
     即判为漂移，放弃改写回退原 query。护栏只在**文本确有变化**时触发编码——纯直通不加载模型，
     保证链路单测无需网络/模型即可跑（无 LLM key 时 fixed/refuse/template 全程可用）。

改写对（原 query / 改写后 / 级别 / 是否漂移）随 :class:`RewriteResult` 返回，由 pipeline 记入
统一结构的 ``rewrite`` 日志字段；检索用 ``rewritten``，而策略判定与生成仍以**原 query**为准
（见 rag.online.pipeline.ask），改写只影响召回、不改变对用户问题的作答语境。
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

from . import config as _cfg, embedder

# 异体/繁体 → 规范简体（逐字；NFKC 不做繁简转换，故显式建表）。
# 只收录检索查询里高频、且不会误伤简体文本的字形。
_VARIANT_CHARS = {
    "麼": "么", "數": "数", "類": "类", "應": "应", "當": "当", "與": "与",
    "從": "从", "這": "这", "個": "个", "們": "们", "說": "说", "問": "问", "請": "请",
    "則": "则", "於": "于", "為": "为", "學": "学", "習": "习", "題": "题", "難": "难",
    "對": "对", "時": "时", "後": "后", "過": "过", "係": "系", "號": "号",
    "決": "决", "據": "据", "單": "单", "項": "项", "點": "点",
}
# 常见同音/形近错别字（词级，最长优先替换；只放无歧义的数学场景修复）。
_TYPO_WORDS = {
    "相返数": "相反数", "倒数": "倒数", "有理树": "有理数", "方程": "方程",
}
_RE_SPACE = re.compile(r"\s+")
_RE_DUP_PUNCT = re.compile(r"([。！？?!，,、；;：:])\1+")


def _fix_typos(text: str) -> str:
    """词级错别字（最长优先）后接逐字异体归一。"""
    for bad in sorted(_TYPO_WORDS, key=len, reverse=True):
        good = _TYPO_WORDS[bad]
        if bad != good and bad in text:
            text = text.replace(bad, good)
    return "".join(_VARIANT_CHARS.get(ch, ch) for ch in text)


def _norm_punct(text: str) -> str:
    """NFKC 统一全角/半角，折叠空白与重复句读，去首尾空白。"""
    text = unicodedata.normalize("NFKC", text)
    text = _RE_SPACE.sub(" ", text)
    text = _RE_DUP_PUNCT.sub(r"\1", text)
    return text.strip()


def _rule_rewrite(query: str, rc: _cfg.RewriteConfig) -> str:
    out = query
    if rc.typo_fix:
        out = _fix_typos(out)
    if rc.punct_norm:
        out = _norm_punct(out)
    return out


def _cosine(a: str, b: str) -> float | None:
    """bge 余弦（向量已归一化 → 点积即余弦）。模型不可用时返回 None（不误伤改写）。"""
    try:
        va, vb = embedder.embed_one(a), embedder.embed_one(b)
    except Exception:  # noqa: BLE001 - 嵌入不可用不得打断改写，按"不漂移"处理
        return None
    return sum(x * y for x, y in zip(va, vb))


def semantic_similarity(a: str, b: str) -> float | None:
    """两段文本的 bge 余弦（供 P6 主题切换检测复用；嵌入不可用时返回 None）。"""
    if not a or not b:
        return None
    return _cosine(a, b)


_LLM_PROMPT = (
    "你是查询改写助手。请结合对话历史，把学生最新的问题改写成一个**不依赖上下文、可独立检索**"
    "的完整问题：补全其中的代词（如「它/这个/那种」）与省略的对象，只补全指代，不得改变或扩展原意，"
    "不得作答。只输出改写后的问题本身，不要引号、前缀或解释。\n"
    "对话历史（可能为空）：\n{history}\n最新问题：{query}\n改写后的独立问题："
)


def _llm_rewrite(query: str, history, cfg, backend) -> str | None:
    """LLM 级指代消解/补全；无 backend/异常/空输出均返回 None（视为不改写）。"""
    if backend is None:
        try:
            from . import generation  # 延迟导入，避免链路耦合与非必要依赖

            backend = generation.build_backend(cfg)
        except Exception:  # noqa: BLE001
            return None
    hist = ""
    if history:
        hist = "\n".join(f"- {h}" for h in list(history)[-4:])
    prompt = _LLM_PROMPT.replace("{history}", hist or "（无）").replace("{query}", query)
    try:
        raw = backend.complete(prompt)
    except Exception:  # noqa: BLE001 - LLM 不可用时降级为不改写，绝不阻断主链
        return None
    for line in (raw or "").splitlines():
        cand = line.strip().strip("\"'“”‘’ ").strip()
        if cand:
            return cand
    return None


@dataclass
class RewriteResult:
    """一次改写的全部留痕（供 pipeline 记入统一结构的 rewrite 日志）。"""
    original: str
    rewritten: str
    level: str = "none"        # none | rule | llm
    changed: bool = False       # 是否最终改变了 query
    drift: bool = False         # 是否触发语义漂移护栏（触发即回退原 query）
    cosine: float | None = None # 原↔候选的 bge 余弦（未编码时 None）

    def as_dict(self) -> dict:
        d = asdict(self)
        if d["cosine"] is not None:
            d["cosine"] = round(d["cosine"], 4)
        return d


def rewrite(query: str, *, history=None, cfg: _cfg.Config | None = None,
            backend=None) -> RewriteResult:
    """两级改写 + 漂移护栏，返回带完整留痕的 :class:`RewriteResult`。

    ``history`` 为 P6 多轮预留（LLM 级开启时用于指代消解）；``backend`` 为可注入的
    LLM 后端（单测/评测可传 fake，缺省且 llm_enabled 时才惰性建真后端）。
    """
    cfg = cfg or _cfg.get_config()
    rc = cfg.rewrite
    original = (query or "").strip()
    if not rc.enabled:
        return RewriteResult(original, original, "none", False)

    candidate = _rule_rewrite(original, rc)
    level = "rule" if candidate != original else "none"

    if rc.llm_enabled:
        cand2 = _llm_rewrite(candidate, history, cfg, backend)
        if cand2:
            candidate = cand2
            level = "llm"

    changed = candidate != original
    if not changed:
        return RewriteResult(original, original, "none", False)

    cosine = _cosine(original, candidate)
    if cosine is not None and cosine < rc.drift_threshold:   # 语义漂移→放弃改写
        return RewriteResult(original, original, "none", False, drift=True, cosine=cosine)
    return RewriteResult(original, candidate, level, True, cosine=cosine)
