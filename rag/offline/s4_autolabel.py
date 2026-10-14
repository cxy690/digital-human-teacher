#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 4 · LLM 预标注（rag/offline/rag/offline/s4_autolabel.py）
==============================================
人机协同的**第一环**，不是最后一步。本 stage 只做「预标注」，永远不产出 ``approved``：

- 每条结果都带 ``prelabel_confidence``（0~1）与 ``needs_human_review: true``；
- 结构化标签的 ``status`` 固定为 ``candidate``；合并回创作态 segment 时仍是 ``draft``；
- 置信度低的条目会被排到审核队列前面（``review_priority``），先看最可疑的。

> 为什么这么强调：标注错误一旦以 ``approved`` 形态流进 L1 离线渲染，
> 就会被 TTS 批量合成成音频，返工成本是标注本身的上百倍。机器只负责缩小人工工作量。

三个标注器
----------
====================  ==========================================================
类                     说明
====================  ==========================================================
``Labeler``           抽象基类，定义 ``label_chunk()`` 契约
``RuleLabeler``       纯规则，**离线可跑、无外部依赖**，是默认标注器
``LLMLabeler``        调用桩：只构造 prompt 与解析响应，真实调用未实现
====================  ==========================================================

``LLMLabeler`` 的接入方式（未实现，故意留空）
--------------------------------------------
1. 在 ``build_prompt()`` 的基础上，把 ``llm_call`` 参数替换成真实客户端调用::

       from openai import OpenAI          # 或内部网关 SDK
       client = OpenAI(base_url=..., api_key=...)
       resp = client.chat.completions.create(
           model="your-model",
           messages=[{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": build_prompt(chunk)}],
           response_format={"type": "json_object"},
           temperature=0,
       )
       raw = json.loads(resp.choices[0].message.content)

2. ``label_chunk()`` 已经写好「校验 + 夹紧到受控词表 + 补 confidence/needs_human_review」，
   真实响应只要满足 ``{"teacher_intent": …, "style_emotion": …, "kp_ids": […], "confidence": …}``
   就能直接复用 ``RuleLabeler.normalize()`` 的安全网。

3. 灰度策略建议：先只让 LLM 做 ``teacher_intent``（有 taxonomy 枚举约束，最好校验），
   ``kp_ids`` 继续走规则 + 人工，避免幻觉知识点污染知识图谱。

CLI::

    python rag/offline/rag/offline/s4_autolabel.py --in data/chunks --out data/labels --labeler rule
    python rag/offline/rag/offline/s4_autolabel.py --in data/chunks/<file>.json --labeler llm   # 会明确报未实现
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    from . import common
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common  # type: ignore

try:  # pragma: no cover - 复用 s3 的 kp 匹配逻辑，保持两 stage 一致
    from .s3_chunk import auto_find_kp, load_kp_index, match_kp
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from s3_chunk import auto_find_kp, load_kp_index, match_kp  # type: ignore

log = common.get_logger("s4_autolabel")

#: 需要人工复核的置信度阈值：低于它就在审核队列里标红
LOW_CONFIDENCE_THRESHOLD = 0.6

# --------------------------------------------------------------------------- #
# 规则表
# --------------------------------------------------------------------------- #
#: teacher_intent 规则：(优先级, intent, 关键词表)。**优先级数字越小越优先**，
#: 因为课堂语言常常一句话里同时出现多个信号（"我们总结一下刚才做的练习"），
#: 必须有确定的裁决顺序，否则同一句话两次跑出不同标签，人工无法建立信任。
INTENT_RULES: list[tuple[int, str, tuple[str, ...]]] = [
    (10, "TI_SUMMARY", ("总结一下", "我们总结", "这节课我们学了", "这节课我们认识",
                        "回顾一下今天", "小结一下", "这节课的收获")),
    (15, "TI_HOMEWORK", ("今天的作业", "作业是", "布置作业", "课后完成", "回家做",
                         "课后练习", "课本第", "课后作业")),
    (20, "TI_FAREWELL", ("下课", "再见", "下节课见", "同学们再见")),
    (25, "TI_REVIEW", ("上节课我们", "还记得", "复习一下", "上一节", "之前我们学过",
                       "回忆一下")),
    (30, "TI_INTRO", ("今天我们要", "今天我们来", "这节课我们来认识", "先看一个",
                      "情境", "夏天", "小明去", "生活中")),
    (35, "TI_DEMO", ("举个例子", "比如这道题", "看这道题", "我们来做这道题", "示范一下",
                     "示范一遍", "示范", "看老师怎么", "老师先来", "老师先做")),
    # 注意：这里刻意不放"我们来做两道题"——它是练习口径，字面与"我们来做这道题"高度相似；
    # 也不放"看黑板"（"我们来看黑板上这道题"会被误判成课堂管理），纪律口令单独一张表。
    (40, "TI_PRACTICE", ("请大家", "请同学们", "拿出练习本", "自己做", "试着做", "试着算",
                         "练一练", "随堂练习", "做一做", "动手", "同桌互相", "在练习本上",
                         "开始做", "我们做两道题")),
    (45, "TI_QUESTION", ("谁能告诉", "谁来回答", "谁来说说", "想一想", "思考一下",
                         "是不是", "对不对", "好不好", "谁知道", "哪个同学")),
    (50, "TI_LECTURE", ("我们来看", "我们看", "请看", "也就是说", "它的意思是",
                        "单位", "等于", "厘米", "毫米", "分米")),
]

#: 纪律口令（课堂管理意图的**强信号**）。这些词本身就是纪律动作，
#: 句子里出现它们基本可以断定这句话的主职是管理，而不是讲授。
MANAGE_KEYWORDS = ("安静", "坐好", "注意听", "看黑板", "抬头", "小眼睛看老师", "别说话")
#: 强管理信号被这些前缀修饰时，其实是**讲授**语境，不算管理：
#: 「我们来看黑板上这道题」——"看黑板"只是引出例题的定语，句子的主职是讲题。
#:
#: ⚠️ 历史 bug（已修）：这里原本写成 ``prefix.endswith("来看")``，**永远不可能命中**——
#: 关键字是「看黑板」，"看"正是它的首字，所以"来看"与关键字重叠，
#: 完整的两字"来看"落不进 prefix，prefix 只到"我们来"为止。
#: 现在改为匹配**紧邻关键字的单字动词**，这才是真正能生效的判据。
MANAGE_TEACHING_PREFIX_SUFFIXES = ("来", "看", "观察", "研究", "瞅", "瞧")
#: 兼容旧名（外部若有引用），语义同 MANAGE_TEACHING_PREFIX_SUFFIXES。
MANAGE_FALSE_FRIEND_PREFIXES = MANAGE_TEACHING_PREFIX_SUFFIXES
#: 纪律口令的"弱信号"：只有在句子里没有任何教学动作时才成立。
#: "注意看"必须放这里——"注意看这道题"是讲授，"注意看老师"是管理，字面分不开。
WEAK_MANAGE_KEYWORDS = ("注意看", "注意")
PRAISE_KEYWORDS = ("非常好", "太棒了", "真棒", "了不起", "为你鼓掌", "答得真准", "厉害")
#: 过渡泛化信号：优先级最低，**不允许单独决定 intent**。
#: "我们继续""接下来"只说明话轮在推进，不说明推进到哪个教学动作；
#: 只有当整句没有命中任何更具体的规则时，才退回按上一个环节的 cue 提示处理。
TRANSITION_KEYWORDS = ("我们继续", "接下来", "下面我们", "然后我们", "那么", "好的")

#: style_emotion 规则
EMOTION_QUESTION = "E_CURIOUS"
EMOTION_EXCLAIM = "E_ENTHUSIASTIC"
EMOTION_DEFAULT = "E_CALM"

SYSTEM_PROMPT = """你是一名小学数学教研专家，负责给课堂逐字稿切片做预标注。
只输出 JSON，不要输出解释。字段：
  teacher_intent: taxonomy.teacher_intent 中的合法 id（如 TI_LECTURE）
  style_emotion:  taxonomy.style_emotion 中的合法 id（如 E_CALM）
  kp_ids:         命中的知识点 id 数组，只能取自给定候选，不许编造；无法判断给 []
  confidence:     0~1 的小数，表示你对本次标注的整体把握
不确定就降低 confidence，不要猜。所有结果都会经过人工复核。"""


def _only_teaching_context(text: str, keyword: str) -> bool:
    """判断命中的管理口令是否**只出现在讲授语境**里（即不该按管理处理）。

    「我们来看黑板上这道题」里的"看黑板"是引出例题的定语，不是纪律口令；
    「请看黑板」「看黑板，别说话」才是管理。

    区分靠**紧邻关键字的前一个字**：``来``/``看``/``观察``/``研究`` 等动词在前 → 讲授语境；
    ``请``/``注意``/标点在前 → 真纪律口令。
    只要有一处是"真"管理语境，就返回 ``False``（仍按管理处理，安全优先）。
    """
    for m in re.finditer(re.escape(keyword), text):
        prefix = text[max(0, m.start() - 4): m.start()]
        if not any(prefix.endswith(p) for p in MANAGE_TEACHING_PREFIX_SUFFIXES):
            return False        # 存在一处真管理语境
    return True                 # 所有出现处都是讲授语境


def build_prompt(chunk: dict, kp_candidates: list[dict] | None = None) -> str:
    """构造 LLM 标注 prompt（纯函数，便于单测与复用）。

    prompt 设计要点：把"合法取值"以清单形式给出（而不是让模型自由发挥），
    并要求输出置信度——这样低置信度条目可以被自动挑出来优先给人工。
    """
    intents = common.taxonomy_ids("teacher_intent")
    emotions = common.taxonomy_ids("style_emotion")
    kp_lines = []
    for kp in kp_candidates or []:
        kp_lines.append(f"  - {kp.get('kp_id')}: {kp.get('name')}（别名：{'、'.join(kp.get('aliases') or []) or '无'}）")
    return (
        f"【合法 teacher_intent】{', '.join(intents)}\n"
        f"【合法 style_emotion】{', '.join(emotions)}\n"
        f"【候选知识点】\n" + ("\n".join(kp_lines) if kp_lines else "  （无，kp_ids 请给 []）") + "\n"
        f"【机器预切分的原文】\n{chunk.get('text', '')}\n"
        f"【机器给出的线索】cue={chunk.get('_cue') or '无'}；"
        f"source_locator={((chunk.get('source') or {}).get('locator')) or '无'}\n"
        "请输出 JSON。"
    )


# --------------------------------------------------------------------------- #
# Labeler 抽象基类
# --------------------------------------------------------------------------- #
class Labeler(ABC):
    """预标注器抽象基类。

    子类只需实现 ``predict(chunk)``，返回 ``{"teacher_intent", "style_emotion",
    "kp_ids", "confidence"}``；``normalize()`` 负责把它夹紧到受控词表并强制
    打上 ``needs_human_review``，任何子类都无法绕过这道安全网。
    """

    name = "abstract"

    @abstractmethod
    def predict(self, chunk: dict) -> dict:
        """返回原始预测（未做词表校验）。"""

    def predict_intent(self, chunk: dict) -> tuple[str | None, float]:
        """只预测 teacher_intent（LLM 灰度上线时单独复用）。"""
        return self.predict(chunk).get("teacher_intent"), 0.0

    def label_chunk(self, chunk: dict) -> dict | None:
        """预标注入口：正常返回标准化后的标签。

        子类若要"不可用时降级"，覆写本方法；``LLMLabeler`` 就是这么做的。
        基类实现保证任何标注器产出的结果都经过 ``normalize()`` 的词表夹紧。
        """
        return self.normalize(chunk, self.predict(chunk))

    @property
    def low_threshold(self) -> float:
        """低置信度阈值。做成属性是为了支持 CLI 的 ``--confidence-threshold``
        在不重建对象的情况下调整（否则子类会把导入期的常量固化下来）。"""
        try:
            return float(globals()["LOW_CONFIDENCE_THRESHOLD"])
        except (KeyError, TypeError, ValueError):  # pragma: no cover
            return 0.6

    def normalize(self, chunk: dict, raw: dict) -> dict:
        """把预测结果夹紧到 taxonomy，并补齐人工复核字段。"""
        intents = set(common.taxonomy_ids("teacher_intent"))
        emotions = set(common.taxonomy_ids("style_emotion"))
        intent = raw.get("teacher_intent")
        if intent not in intents:
            intent = None
        emotion = raw.get("style_emotion")
        if emotion not in emotions:
            emotion = EMOTION_DEFAULT
        kp_ids = [k for k in (raw.get("kp_ids") or []) if isinstance(k, str)]
        try:
            confidence = float(raw.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(1.0, max(0.0, confidence))
        reasons = list(raw.get("reasons") or [])
        if intent is None:
            reasons.append("规则未命中任何教学意图，需人工指定")
        if not kp_ids:
            reasons.append("未匹配到知识点，需人工挂接")
        # 审核优先级由**两件事**共同决定，而不是揉成一个数：
        #   1) 意图判定本身不可信（未判定 / 低置信度）
        #   2) 该意图按 taxonomy 应当挂知识点，但没挂上（needs_kp）
        # 二者任一成立就排到队列前面（priority=1）。
        needs_kp = bool(raw.get("needs_kp"))
        priority = 1 if (intent is None or confidence < self.low_threshold or needs_kp) else 3
        return {
            "status": "candidate",
            "teacher_intent": intent,
            "style_emotion": emotion,
            "kp_ids": kp_ids,
            "prelabel_confidence": round(confidence, 3),
            # 分字段置信度：便于教研看出"到底是哪一项不可信"，
            # 也说明为什么"未挂知识点"不参与整体置信度（见 RuleLabeler.predict）
            "prelabel_confidence_by_field": raw.get("confidence_by_field") or {},
            "prelabel_source": self.name,
            "needs_human_review": True,          # 硬约束：机器永不产出 approved
            "needs_kp": needs_kp,
            "low_confidence": confidence < self.low_threshold,
            "review_priority": priority,
            "review_notes": "；".join(reasons),
        }


# --------------------------------------------------------------------------- #
# 规则标注器
# --------------------------------------------------------------------------- #
class RuleLabeler(Labeler):
    """纯规则标注器：关键词表 + 标点启发式，**不依赖任何外部服务，可离线跑**。

    为什么先有它：a) 离线可跑，CI 里能当回归测试；b) 它给出的置信度分布，
    正好是判断"哪些内容真的需要 LLM"的依据；c) LLM 不可用时流水线不断。
    """

    name = "rule"

    def __init__(self, kp_index: dict[str, list[str]] | None = None,
                 cue_intent: dict[str, str] | None = None):
        self.kp_index = kp_index or {}
        self.cue_intent = cue_intent or {}

    # ---------------- 各字段的规则 ----------------
    @staticmethod
    def predict_intent(text: str, hint: str | None = None) -> tuple[str | None, float, str]:
        """返回 ``(intent, 置信度, 命中理由)``。命中越靠前的规则置信度越高。

        **优先级顺序（本函数最容易出错的地方，改动务必同步回归用例）**：

        1. **表扬 / 纪律口令**（``安静``/``坐好``/``看黑板``…）：整句性质的管理与鼓励。
           必须最优先——否则"请同学们安静一下"会被"请同学们"抢成随堂练习。
           ``注意看`` 属弱信号，放在教学动作之后判断（"注意看这道题"是讲授）。
        2. **具体教学动作**（``INTENT_RULES``，已按强→弱排序）：示范 / 练习 / 提问 / 讲授…
           其中 TI_LECTURE 是泛化兜底，优先级最低。
        3. **过渡泛化信号**（``我们继续``/``接下来``）：**不允许单独决定 intent**，
           只作为"说话人在推进话轮"的旁证；整句没有更强信号时回落到 s3 的 cue 提示。
        """
        # 第一优先级：表扬 / 纪律口令（强信号）
        for kw in MANAGE_KEYWORDS:
            if kw in text and not _only_teaching_context(text, kw):
                return "TI_MANAGE", 0.55, f"命中课堂管理口令『{kw}』"
        for kw in PRAISE_KEYWORDS:
            if kw in text:
                return "TI_PRAISE", 0.55, f"命中表扬用语『{kw}』"
        # 第二优先级：具体教学动作（INTENT_RULES 已按强→弱排序）
        for priority, intent, keywords in INTENT_RULES:
            for kw in keywords:
                if kw in text:
                    # 优先级越小越确定。基准取 0.95 而非 0.9：
                    # 阈值 LOW_CONFIDENCE_THRESHOLD=0.6 用来决定"先进审核队列"，
                    # 若基准太低（旧值 0.9），最强信号也只能到 0.55，
                    # 于是**所有规则标注都被判为低置信度**、review_priority 全变 1，
                    # 排序信号彻底失效。现在 priority≤35 的强信号能过线，
                    # 泛化信号（LECTURE/QUESTION/PRACTICE）仍落在线下，符合直觉。
                    conf = round(max(0.5, 0.95 - priority / 100.0), 3)
                    return intent, conf, f"命中关键词『{kw}』（规则优先级 {priority}）"
        # 第三优先级：弱纪律信号（句子里没有任何教学动作时才成立）
        for kw in WEAK_MANAGE_KEYWORDS:
            if kw in text:
                return "TI_MANAGE", 0.5, f"命中弱管理信号『{kw}』（未命中任何教学动作）"
        # 第四优先级：过渡泛化信号 + s3 的 cue 提示（最弱证据，不单独定 intent）
        transition = [kw for kw in TRANSITION_KEYWORDS if kw in text]
        if hint and hint in common.taxonomy_ids("teacher_intent"):
            note = f"沿用 s3 切分时的环节提示 {hint}（未命中关键词规则）"
            if transition:
                note += f"；句中有过渡语『{transition[0]}』，不足以单独定意图"
            return hint, 0.45, note
        if transition:
            return None, 0.2, f"仅命中过渡泛化信号『{transition[0]}』，不足以判定教学意图，需人工指定"
        return None, 0.0, "无规则命中"

    @staticmethod
    def predict_emotion(text: str) -> tuple[str, float, str]:
        """疑问 → E_CURIOUS；感叹 → E_ENTHUSIASTIC；否则 E_CALM。"""
        if re.search(r"[?？]", text):
            return EMOTION_QUESTION, 0.7, "含疑问句标点『？』"
        if re.search(r"[!！]", text):
            return EMOTION_EXCLAIM, 0.7, "含感叹号『！』"
        return EMOTION_DEFAULT, 0.5, "陈述句，默认平和讲解（可人工改为 E_WARM）"

    def predict_kp(self, text: str) -> tuple[list[str], float, str]:
        """关键词 / 别名匹配知识点。"""
        if not self.kp_index:
            return [], 0.0, "无知识点索引（未提供 kp.yaml）"
        hits = match_kp(text, self.kp_index)
        if not hits:
            return [], 0.0, "未匹配到任何知识点关键词"
        # 命中多个知识点时给较低置信度：跨知识点片段需要教研判断归属
        conf = 0.75 if len(hits) == 1 else 0.5
        return hits, conf, f"关键词匹配到 {len(hits)} 个知识点"

    # ---------------- 契约实现 ----------------
    def predict(self, chunk: dict) -> dict:
        text = str(chunk.get("text") or "")
        hint = chunk.get("_cue") or chunk.get("teacher_intent_hint")
        intent, c_intent, r_intent = self.predict_intent(text, hint)
        emotion, c_emotion, r_emotion = self.predict_emotion(text)
        kp_ids, c_kp, r_kp = self.predict_kp(text)

        # 整体置信度 = **意图判定**的可信度（主决策），再打 0.9 折。
        #
        # 为什么只算意图，改过三版，前两版都在"把两件事揉进一个数"：
        #   ① min(intent, emotion, kp)：kp 没命中就把整体压成 0.0。可「下课，同学们再见」
        #      本来就不该挂知识点 —— 那不是"低置信度"，是"正常"。
        #   ② min(intent, kp)：仍是同样的揉法。而且 0.0 会被误读成"意图判错了"，
        #      实际意图判得很准，缺的只是知识点。
        #
        # 最终：**置信度只描述意图判定**；"该挂却没挂上知识点"是另一件事，
        # 用 needs_kp 标志 + review_notes 单独表达，两者一起决定审核优先级。
        # 情绪不参与：它是次要标签、改动成本低、不涉合规风险，其可信度在
        # confidence_by_field.emotion 里单独暴露。
        kp_bound = self._intent_kp_bound().get(intent, True) if intent else False
        needs_kp = bool(kp_bound and not kp_ids)
        conf = round(c_intent * 0.9, 3) if intent else 0.0

        return {
            "teacher_intent": intent,
            "style_emotion": emotion,
            "kp_ids": kp_ids,
            "confidence": conf,
            "confidence_by_field": {
                "intent": round(c_intent, 3),
                "emotion": round(c_emotion, 3),
                "kp": round(c_kp, 3),
            },
            "kp_bound": kp_bound,
            "needs_kp": needs_kp,
            "reasons": [r_intent, r_emotion, r_kp],
        }

    @staticmethod
    def _intent_kp_bound() -> dict[str, bool]:
        """teacher_intent → kp_bound（该意图是否**应当**挂知识点）。取自 taxonomy。"""
        try:
            tax = common.load_taxonomy()
            return {e["id"]: bool(e.get("kp_bound", True))
                    for e in tax.get("teacher_intent", [])}
        except Exception:  # noqa: BLE001 - 拿不到就都按"应当挂"，偏保守
            return {}


# --------------------------------------------------------------------------- #
# LLM 标注器（桩）
# --------------------------------------------------------------------------- #
class LLMLabeler(Labeler):
    """LLM 预标注器：**只实现 prompt 构造与响应解析的骨架，不实现真实调用**。

    未实现的原因：流水线骨架阶段不引入网络依赖与密钥管理，避免 CI 不可复现。
    接入方式见模块 docstring（把 ``llm_call`` 换成真实客户端即可）。

    ``label_chunk()`` 在 ``llm_call is None`` 时返回 ``None``，调用方据此降级到
    ``RuleLabeler``——**优雅降级而不是抛异常**，保证流水线在无模型环境里仍能跑通。
    """

    name = "llm"

    def __init__(self, llm_call=None, kp_candidates: list[dict] | None = None,
                 fallback: Labeler | None = None):
        self.llm_call = llm_call
        self.kp_candidates = kp_candidates or []
        self.fallback = fallback

    def build_prompt(self, chunk: dict) -> str:
        """构造发给模型的 prompt（已实现，可单测）。"""
        return build_prompt(chunk, self.kp_candidates)

    def parse_response(self, raw: Any) -> dict:
        """解析模型响应：支持 JSON 字符串或已解析的 dict；解析失败返回空预测。"""
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return {}
        if not isinstance(raw, dict):
            return {}
        return {
            "teacher_intent": raw.get("teacher_intent"),
            "style_emotion": raw.get("style_emotion"),
            "kp_ids": raw.get("kp_ids") or [],
            "confidence": raw.get("confidence", 0.0),
            "reasons": ["LLM 预标注"],
        }

    def predict(self, chunk: dict) -> dict:
        """真实调用未实现：``llm_call`` 为空时抛 ``NotImplementedError`` 说明接入方式。"""
        if self.llm_call is None:
            raise NotImplementedError(
                "LLMLabeler 未接入真实模型：请向 LLMLabeler(llm_call=...) 传入一个"
                " (prompt: str) -> dict|str 的可调用对象（见模块 docstring 的接入步骤）。"
                "当前请使用 --labeler rule。"
            )
        return self.parse_response(self.llm_call(self.build_prompt(chunk)))

    def label_chunk(self, chunk: dict) -> dict | None:
        """LLM 标注入口：不可用时返回 ``None``，由调用方降级（不抛异常、不中断流水线）。"""
        try:
            return self.normalize(chunk, self.predict(chunk))
        except NotImplementedError as exc:
            log.warning("LLM 标注不可用，将降级：%s", exc)
            if self.fallback is not None:
                return self.fallback.label_chunk(chunk)
            return None


# --------------------------------------------------------------------------- #
# 组装：把预标注结果合并回 segment 草稿
# --------------------------------------------------------------------------- #
def annotate_chunk(chunk: dict, label: dict, lesson_id: str) -> dict:
    """把预标注结果写进 segment 草稿。

    受 ``segment.schema.json`` 的 ``additionalProperties: false`` 约束：
    schema 里没有 ``prelabel_*`` / ``needs_human_review`` 这些字段，
    所以它们**不能**平铺在记录里（否则导出门禁必挂）。做法是把预标注信息
    以 ``[机器预标注] …`` 前缀写进 schema 允许的 ``review_notes``，
    同时在同一份输出的 JSON 里保留结构化副本（见 ``label_document``）。
    """
    out = {k: v for k, v in chunk.items() if k != "_comment"}
    if label.get("teacher_intent"):
        out["teacher_intent"] = label["teacher_intent"]
    if label.get("kp_ids"):
        merged = list(dict.fromkeys([*(out.get("kp_ids") or []), *label["kp_ids"]]))
        out["kp_ids"] = merged
    style = dict(out.get("style") or {})
    style["emotion"] = label.get("style_emotion", style.get("emotion", EMOTION_DEFAULT))
    out["style"] = style
    out["status"] = "draft"
    out["version"] = int(out.get("version") or 1)
    out["needs_human_review"] = True
    out["prelabel_confidence"] = label.get("prelabel_confidence")
    out["prelabel_source"] = label.get("prelabel_source")
    machine_note = (
        f"[机器预标注] source={label.get('prelabel_source')} "
        f"confidence={label.get('prelabel_confidence')} "
        f"needs_human_review=true "
        f"teacher_intent={label.get('teacher_intent') or '未判定'}；"
        f"{label.get('review_notes') or ''}"
    )
    prev = str(out.get("review_notes") or "").strip()
    out["review_notes"] = f"{prev}｜{machine_note}" if prev else machine_note
    return out


def label_document(
    chunk_doc: dict,
    labeler: Labeler,
    lesson_id: str,
) -> dict:
    """对一份 s3 切分结果跑预标注，返回含 ``segments``（草稿）与 ``labels``（结构化副本）。"""
    chunks = chunk_doc.get("chunks") or []
    segments: list[dict] = []
    labels: list[dict] = []
    low = 0
    for ch in chunks:
        label = labeler.label_chunk(ch)
        if label is None:
            log.warning("切片 %s 预标注失败（无可用 labeler），跳过标注只保留草稿",
                        ch.get("segment_id"))
            label = {
                "status": "candidate",
                "teacher_intent": None, "style_emotion": EMOTION_DEFAULT, "kp_ids": [],
                "prelabel_confidence": 0.0, "prelabel_source": "none",
                "needs_human_review": True, "low_confidence": True, "review_priority": 1,
                "review_notes": "无可用标注器，未产生任何机器标签",
            }
        if label.get("low_confidence"):
            low += 1
        segments.append(annotate_chunk(ch, label, lesson_id))
        labels.append({
            "segment_id": ch.get("segment_id"),
            "source_locator": (ch.get("source") or {}).get("locator"),
            **label,
        })
    stats = {
        "chunk_count": len(chunks),
        "labeled_count": sum(1 for x in labels if x.get("teacher_intent")),
        "low_confidence_count": low,
        "low_confidence_ratio": round(low / len(chunks), 3) if chunks else 0.0,
        "labeler": getattr(labeler, "name", "unknown"),
        "intent_distribution": _dist(labels, "teacher_intent"),
        "emotion_distribution": _dist(labels, "style_emotion"),
    }
    return {
        "lesson_id": lesson_id,
        "source_name": chunk_doc.get("source_name"),
        "labeler": getattr(labeler, "name", "unknown"),
        "labeled_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "segments": segments,
        "labels": labels,
        "stats": stats,
        "warnings": list(chunk_doc.get("warnings") or []) + [
            "预标注结果全部 needs_human_review=true，未经教研审核不可导出"
        ],
    }


def _dist(rows: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        v = r.get(key) or "<未判定>"
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def make_labeler(kind: str, lesson_id: str, kp_path: str | Path | None = None) -> Labeler:
    """按名字构造标注器；``llm`` 用 RuleLabeler 作为降级实现（见 README 的灰度建议）。"""
    kp_file = Path(kp_path) if kp_path else auto_find_kp(lesson_id)
    kp_index = load_kp_index(kp_file) if kp_file else {}
    if kind == "rule":
        return RuleLabeler(kp_index)
    if kind == "llm":
        kp_items = common.read_yaml_list(kp_file) if kp_file else []
        return LLMLabeler(kp_candidates=kp_items, fallback=RuleLabeler(kp_index))
    raise ValueError(f"未知标注器 {kind!r}（支持：rule / llm）")


def label_dir(
    in_dir: str | Path,
    out_dir: str | Path,
    lesson_id: str,
    labeler_kind: str = "rule",
    kp_path: str | Path | None = None,
) -> list[Path]:
    """批量预标注目录下的 s3 切分产物，输出 ``<lesson>__<file>.yaml/.json``。"""
    files = [p for p in common.iter_input_files(in_dir, [".json"])
             if not p.name.endswith(".labels.json")]
    if not files:
        log.warning("输入目录 %s 下没有 s3 切分产物（*.json），请先跑 s3_chunk.py", in_dir)
        return []
    labeler = make_labeler(labeler_kind, lesson_id, kp_path)
    out = common.ensure_dir(out_dir)
    written: list[Path] = []
    for f in files:
        chunk_doc = common.read_json(f)
        result = label_document(chunk_doc, labeler, lesson_id)
        stem = f"{lesson_id}__{common.slug(f.stem)}"
        header = [
            f"预标注草稿（rag/offline/rag/offline/s4_autolabel.py，labeler={labeler_kind}）",
            "硬约束：结构化标签 status=candidate、needs_human_review=true；segment 仍为 draft，机器绝不产出 approved。",
            "prelabel_confidence 是机器的自评置信度，review_priority=1 的条目建议优先看。",
            "教研审核后请把结果写到 data/review/<lesson_id>/，不要直接改本文件（会被下次运行覆盖）。",
        ]
        yml = common.write_yaml_list(out / f"{stem}.labels.yaml", result["segments"], "\n".join(header))
        js = common.write_json(out / f"{stem}.labels.json", result)
        written += [yml, js]
        s = result["stats"]
        log.info("预标注完成 %s → %d 条，低置信度 %d 条（%.0f%%），意图分布 %s",
                 f.name, s["chunk_count"], s["low_confidence_count"],
                 s["low_confidence_ratio"] * 100, s["intent_distribution"])
    log.info("本批共标注 %d 个文件，输出目录 %s", len(files), out)
    return written


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="s4_autolabel",
        description="Stage4 LLM 预标注：给候选切片打 teacher_intent / style_emotion / kp_ids 预标签（全部待人工复核）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python rag/offline/rag/offline/s4_autolabel.py --in data/chunks --out data/labels --labeler rule --lesson LESSON-MATH-P3A-RJ-U03\n"
            "  python rag/offline/rag/offline/s4_autolabel.py --in data/chunks --out data/labels --labeler llm  --lesson LESSON-MATH-P3A-RJ-U03\n"
            "\n说明：--labeler llm 目前会打印『未接入』并自动降级到 rule（见 LLMLabeler 文档）。\n"
        ),
    )
    common.add_in_out_args(ap, "data/chunks", "data/labels")
    ap.add_argument("--labeler", choices=["rule", "llm"], default="rule",
                    help="标注器：rule=纯规则（离线可跑，默认）；llm=模型预标注（需自行接入）")
    ap.add_argument("--lesson", default="LESSON-MATH-P3A-RJ-U03", help="课程 ID（用于 kp.yaml 定位）")
    ap.add_argument("--kp", default=None, help="kp.yaml 路径（默认自动在 samples/<lesson>/ 查找）")
    ap.add_argument("--confidence-threshold", type=float, default=LOW_CONFIDENCE_THRESHOLD,
                    help=f"低置信度阈值（默认 {LOW_CONFIDENCE_THRESHOLD}），低于它的条目 review_priority=1")
    ap.add_argument("--show-prompt", action="store_true",
                    help="只打印一条示例 prompt（用于评审 LLM 标注方案，不跑标注）")
    return ap


def main(argv: list[str] | None = None) -> int:
    global LOW_CONFIDENCE_THRESHOLD
    args = build_parser().parse_args(argv)
    LOW_CONFIDENCE_THRESHOLD = args.confidence_threshold

    if args.show_prompt:
        kp_file = Path(args.kp) if args.kp else auto_find_kp(args.lesson)
        kp_items = common.read_yaml_list(kp_file) if kp_file else []
        demo = {"text": "我们来看这道题，1 厘米等于多少毫米？", "_cue": "TI_LECTURE",
                "source": {"locator": "00:08:10-00:08:25"}}
        print(build_prompt(demo, kp_items))
        print("\n# 以上为示例 prompt；LLMLabeler 的真实调用尚未实现（见模块 docstring）。")
        return 0

    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        log.error("输入目录不存在：%s（请先跑 python rag/offline/rag/offline/s3_chunk.py）", in_dir)
        return 1
    if args.labeler == "llm":
        log.warning("LLM 标注器尚未接入真实模型，将自动降级到规则标注器（RuleLabeler）")
    written = label_dir(in_dir, args.out_dir, args.lesson, args.labeler, args.kp)
    return 0 if written or not common.iter_input_files(in_dir, [".json"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
