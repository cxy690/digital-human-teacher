"""core.intent：意图识别 / 分流决策层（纯决策，不是 Agent）。

由 ``agents/router/`` 迁移而来：它不持有会话状态、不调用其他执行者、不参与 ``AgentBus``，
只是「一次学生输入 → ``IntentDecision``」的确定性决策器，因此不属于 agents 层
（``agents/`` 只放真正干活的执行者）。

* :class:`IntentAnalyzer` —— 规则直判 + 复杂/多意图标注 + 执行计划；同时透出
  ``IntentRouter`` 的 ``route`` / ``route_async`` / ``_looks_like_question`` 接口，
  因此可原地替换 ``Supervisor`` 里的 router；
* :class:`IntentLlmClassifier` —— 二级小模型复核，只在规则判不准 / 复杂·多意图时调用，
  超时或输出不合法一律回退规则结果。
"""
from core.intent.analyzer import IntentAnalyzer, IntentDecision, IntentSegment
from core.intent.classifier import IntentLlmClassifier

__all__ = [
    "IntentAnalyzer",
    "IntentDecision",
    "IntentSegment",
    "IntentLlmClassifier",
]
