"""回合规划层：让模型在开口之前先产出「目标 + 步骤 + 建议工具」的执行计划。

与 ``core/intent`` 的分工：

* ``core.intent.IntentAnalyzer`` 回答「学生这句是什么意图、要不要转交」——它是**闸门**，
  规则优先、零成本，判的是**类别**；
* ``core.planning.TurnPlanner`` 回答「这一回合打算怎么做」——它在闸门之后、答疑之前运行，
  由模型给出一份**可执行计划**（先拆条件还是先查教材、要不要先澄清条件不足、
  该不该用工具），作为上下文注入答疑链路。

计划**不是**硬指令：真正调用哪些工具仍由模型在 ReAct 闭环里现场决定
（``agents/teacher/react.py``）。计划的价值在于把「这道题该怎么讲」这件本该由模型
自己拿主意的事，从写死在代码里的决策树挪回模型手里，并且留下可回溯的痕迹
（``turn.meta["turn_plan"]``）。
"""
from core.planning.planner import PlanStep, TurnPlan, TurnPlanner, get_turn_planner

__all__ = ["PlanStep", "TurnPlan", "TurnPlanner", "get_turn_planner"]
