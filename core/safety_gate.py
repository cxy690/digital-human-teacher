"""安全网关（对象形态）：把 :mod:`core.guardrail` 的教学护栏变成**必经挂载点**。

**为什么是网关，而不是第 5 个平级 Agent**：
如果它只是「总控最后调一下的子 Agent」，那它永远只能提建议 —— 那正是改造前的现状
（``guardrail.filter_output`` 调用点零散、危机信号被当普通敏感词打码）。
挂成网关后提供三个入口，且都不依赖总控「记得调用」：

* :meth:`SafetyGate.screen_student_text` —— 学生输入侧，危机信号优先于一切教学逻辑；
* :meth:`SafetyGate.screen_response` —— AI 输出侧，覆盖话术 / 白板文字 / 分步解析 / 追问；
* :meth:`SafetyGate.screen_text` —— 不走 ``AIResponse`` 的裸文本通道（流式话术、板书）。

处置口径与 :mod:`core.guardrail` 保持一致（三级）：

====================  ================================================
危机信号              顶替为关怀话术 + 上报（**不是**打码后继续讲课）
不当用语              「*」遮蔽
教学内容敏感          只标记，不改文本
====================  ================================================

与 ``core/a2a/safety.py``（云端分叉）的关系：本文件是它的**对象形态部分**；
事件形态（``screen_outbound`` / ``screen_inbound``，供 A2A 网关使用）依赖
``core/a2a/protocol`` 那套进程内总线，随 A2A 编排层一起延后，未纳入本次融合。
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from config.logging import get_logger
from core.guardrail import GuardrailVerdict, TeachingGuardrail
from core.state.models import AIResponse

log = get_logger("core.safety_gate")


class SafetyGate:
    """安全网关：输入侧危机专线 + 输出侧三级过滤。"""

    #: 危机专线话术（与 guardrail 同源，避免两处硬编码）。
    CRISIS_SPEECH = TeachingGuardrail.CRISIS_SPEECH

    def __init__(
        self,
        guardrail: TeachingGuardrail | None = None,
        *,
        reporter: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.guardrail = guardrail if guardrail is not None else TeachingGuardrail()
        #: 事件上报回调（危机/敏感命中时调用）。由 ``Supervisor`` 注入，
        #: 推到前端 ``safety_alert`` 并落盘审计；接人工工单是后续的事。
        self.reporter = reporter

    # ============================================================ 对象形态
    def screen_student_text(self, text: str) -> GuardrailVerdict:
        """学生输入侧：危机信号优先，命中即**不该继续答疑**。

        只做危机判定，不做不当用语遮蔽 —— 学生说了什么要原样保留，
        它是学情分析与人工介入的证据，不该被改写。
        """
        verdict = self.guardrail.screen_crisis(text or "")
        if verdict.crisis:
            # 必须带上**原文**：人工复核的第一件事就是「学生到底说了什么」，
            # 只记命中词等于把最关键的证据丢了。
            self._report("crisis", verdict.crisis, verdict.speech, source="student", text=text or "")
        return verdict

    def screen_response(self, response: AIResponse) -> GuardrailVerdict:
        """AI 输出侧：就地遮蔽不当用语、顶替危机内容、标记敏感内容。

        覆盖全部**会到达学生眼睛/耳朵**的文本通道：

        * ``response.speech`` —— 话术，同时是 TTS 合成源（覆盖语音通道）；
        * ``response.board_ops[].text`` —— 白板文字（覆盖板书通道）；
        * ``response.meta['steps']`` —— 分步解析，会被 ``_spoken_text`` 念出来；
        * ``response.meta['followup']['question']`` —— 追问题目，同样会被念出来。

        危机一旦命中：整段作废（清空板书与追问），只保留关怀话术 —— fail-closed。
        """
        verdict = GuardrailVerdict()
        response.speech = self._filter(response.speech or "", verdict)
        for op in response.board_ops:
            if op.text:
                op.text = self._filter(op.text, verdict)

        steps = response.meta.get("steps")
        if isinstance(steps, list):
            response.meta["steps"] = [self._filter(str(s), verdict) for s in steps]

        followup = response.meta.get("followup")
        if isinstance(followup, dict) and followup.get("question"):
            followup["question"] = self._filter(str(followup["question"]), verdict)

        if verdict.crisis:
            response.board_ops = []
            response.meta.pop("steps", None)
            response.meta.pop("followup", None)
            self._report("crisis", verdict.crisis, verdict.speech, source="model", text=response.speech)

        if verdict.hit:
            response.meta["safety"] = {
                "crisis": verdict.crisis,
                "unsafe": verdict.unsafe,
                "flagged": verdict.flagged,
            }
        return verdict

    def screen_text(self, text: str) -> str:
        """过滤单段文本，返回安全文本（用于不走 ``AIResponse`` 的通道）。

        目前调用点：**流式话术**（句子直接送 TTS/渲染，不经过 ``_speak``）。
        危机命中时整段顶替为关怀话术。
        """
        safe, verdict = self.guardrail.filter_output(text)
        if verdict.crisis:
            self._report("crisis", verdict.crisis, safe, source="stream", text=text)
        return safe

    def screen_board_ops(self, ops: list[Any]) -> GuardrailVerdict:
        """白板指令流过滤（就地改写文本；危机命中即清空整批指令）。

        单独开一个入口，是因为**板书有旁路**：流式答疑与课件连讲会直接
        ``session.push_board(...)``，不经过 ``_speak``。不在这里收口，
        板书就是一条不经过滤的通道（而板书恰恰是学生一定会看到的内容）。
        """
        verdict = GuardrailVerdict()
        for op in ops:
            text = getattr(op, "text", "")
            if text:
                op.text = self._filter(text, verdict)
        if verdict.crisis:
            ops.clear()
            self._report("crisis", verdict.crisis, verdict.speech, source="board")
        return verdict

    # ============================================================ 内部
    def _filter(self, text: str, verdict: GuardrailVerdict) -> str:
        """过滤单段文本并合并判定（供 :meth:`screen_response` 复用）。"""
        safe, extra = self.guardrail.filter_output(text)
        self._merge(verdict, extra)
        return safe

    @staticmethod
    def _merge(target: GuardrailVerdict, extra: GuardrailVerdict) -> None:
        """把一次判定的结果并入累计判定（去重，保持先命中的顺序）。"""
        target.allowed = target.allowed and extra.allowed
        for field_name in ("crisis", "unsafe", "flagged"):
            bucket: list[str] = getattr(target, field_name)
            for item in getattr(extra, field_name):
                if item not in bucket:
                    bucket.append(item)
        if extra.speech:
            target.speech = extra.speech

    def _report(self, kind: str, hits: list[str], speech: str, *, source: str, text: str = "") -> None:
        """上报安全事件（日志 + 注入的回调）。上报失败绝不能影响主链路。

        ``text`` 是**触发命中的原文**：人工复核的第一件事就是「学生/模型到底说了什么」，
        只记命中到的词等于把最关键的证据丢了。
        """
        log.warning("安全事件 kind={} source={} hits={} text={!r}", kind, source, hits, text[:60])
        if self.reporter is None:
            return
        payload = {"kind": kind, "source": source, "hits": list(hits), "speech": speech}
        if text:
            payload["text"] = text
        try:
            self.reporter(payload)
        except Exception as err:  # noqa: BLE001 - 上报链路故障不得掀翻答疑
            log.warning("安全事件上报失败：{}", err)


__all__ = ["SafetyGate"]
