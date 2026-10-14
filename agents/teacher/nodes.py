"""AI 教师 Agent（``TeacherAgent``）负责的流程节点。

教师承担三条链路，各自的节点与连接顺序如下（声明见 :data:`ANSWER_SPECS` /
:data:`JUDGE_SPECS` / :data:`QUIZ_SPECS`）：

| 图 | 节点（按连接顺序） | 说明 |
|---|---|---|
| ``teacher.answer`` | ``retrieve`` → ``generate`` → ``validate`` → ``degrade`` → ``solve`` | 答疑流水线：RAG 检索 → 生成讲解/白板 → 校验 → 丢弃非法白板指令 → sympy 验算 |
| ``teacher.judge`` | ``evaluate`` → ``decide`` → ``announce`` | 判题流水线：判定作答 → 决定反馈/重试/公布答案 → 播报 + 推送 ai_check |
| ``teacher.quiz`` | ``generate`` → ``publish`` → ``announce`` | 出题流水线：生成模拟题 → 写入待答状态并推 ai_quiz → 播报 |

``teacher.answer`` 与 ``core/langgraph_flow.py`` 的 LangGraph 版答疑子图**节点同名同序**
（两套编排行为由 ``tests/test_graph_parity.py`` 强绑定），因此这里既是自研（TaskGraph）
编排的事实源，也是两套编排对照的基准。

节点所需配置与外部能力逐条标在 ``requires`` 里（如 ``settings.course_quiz_enable``、
``models.llm.get_llm``）：缺配置时的降级路径写在节点 handler 内，保证「少一个开关
只是少一步，不是整节课挂掉」。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from config.constants import IntentType
from config.logging import get_logger
from config.settings import settings
from core import math_solver
from core.graph import NodeSpec, TaskGraph
from core.state.models import AIResponse, QuizItem, StudentTurn
from core.validator import drop_invalid_board_ops

if TYPE_CHECKING:  # 仅类型标注：handler 挂在总控实例上，运行期无需导入
    from agents.supervisor import Supervisor

log = get_logger("agents.teacher.nodes")

#: 答疑流水线图名
ANSWER_GRAPH = "teacher.answer"
#: 判题流水线图名
JUDGE_GRAPH = "teacher.judge"
#: 出题流水线图名
QUIZ_GRAPH = "teacher.quiz"


# ------------------------------------------------------------------ 答疑流水线
ANSWER_SPECS: tuple[NodeSpec, ...] = (
    NodeSpec(
        name="retrieve",
        summary="RAG 检索教材/题库上下文；失败或无检索器时退化为纯 LLM",
        impl="Supervisor.retriever.retrieve",
        inputs=("turn", "notes"),
        outputs=("notes",),
        deps=(),
        requires=("Supervisor.retriever（可选：core/harness.py 装配）",),
    ),
    NodeSpec(
        name="generate",
        summary="生成讲解与白板内容：ReAct 闭环（模型自主调工具复核）失败则退回单次生成"
                "（图片题走 answer_image，文字/语音走 answer）",
        impl="TeacherAgent.answer / TeacherAgent.answer_image",
        inputs=("turn", "intent", "hint_level", "ocr_text", "notes"),
        outputs=("response",),
        deps=("retrieve",),
        requires=(
            "models.llm.get_llm",
            "core.hint_homework.HintLevel（作业模式分层提示）",
            "settings.agent_react_enable / settings.agent_tools_enable（关闭即退回单次生成）",
        ),
    ),
    NodeSpec(
        name="validate",
        summary="校验模型产物（话术长度、白板坐标、作业提示层级）",
        impl="Supervisor.validator.validate",
        inputs=("response", "hint_level"),
        outputs=("validation",),
        deps=("generate",),
        requires=("core.validator.ResponseValidator",),
    ),
    NodeSpec(
        name="degrade",
        summary="校验不通过时只丢弃有问题的白板指令（不整块清屏，避免「讲了半天板上一片白」）",
        impl="core.validator.drop_invalid_board_ops",
        inputs=("response", "validation"),
        outputs=("response",),
        deps=("validate",),
        requires=("core.validator.drop_invalid_board_ops",),
    ),
    NodeSpec(
        name="solve",
        summary="独立求解 / 验算：sympy 复核含等号的等式题，必要时追加更正话术",
        impl="core.math_solver.apply_verification",
        inputs=("response",),
        outputs=("response",),
        deps=("degrade",),
        requires=("core.math_solver.available",),
    ),
)


def build_answer_graph(supervisor: "Supervisor") -> TaskGraph:
    """构建答疑流水线图（内建编排；LangGraph 版见 ``core/langgraph_flow.py``）。"""

    async def retrieve(ctx: dict[str, Any]) -> dict[str, Any]:
        notes = list(ctx.get("notes") or [])
        if supervisor.retriever is None:
            return {"notes": notes}
        notes += await supervisor.retriever.retrieve(ctx["turn"].text)
        return {"notes": notes}

    async def generate(ctx: dict[str, Any]) -> dict[str, Any]:
        turn = ctx["turn"]
        hint_level = ctx.get("hint_level")
        # 工具集由总控持有（带本会话的检索器 / OCR / 白板）：模型在 ReAct 闭环里
        # 自主决定调哪些工具，工具看到的也是这节课的真实上下文
        toolkit = supervisor.toolkit_for(supervisor.session)
        if turn.intent is IntentType.IMAGE_QUESTION:
            response = await supervisor.teacher.answer_image(
                turn, supervisor.session,
                ocr_text=ctx.get("ocr_text", ""), hint_level=hint_level,
                toolkit=toolkit,
            )
        else:
            response = await supervisor.teacher.answer(
                turn, supervisor.session,
                hint_level=hint_level, context_notes=ctx.get("notes"),
                toolkit=toolkit,
            )
        return {"response": response}

    def validate(ctx: dict[str, Any]) -> dict[str, Any]:
        return {"validation": supervisor.validator.validate(
            ctx["response"], hint_level=ctx.get("hint_level")
        )}

    def degrade(ctx: dict[str, Any]) -> dict[str, Any]:
        response: AIResponse = ctx["response"]
        result = ctx.get("validation")
        if result is None or result.ok:
            return {}
        log.warning("response validation issues: {}", result.reasons)
        # 只丢弃有问题的白板指令，不要把整块板清空
        # （整块清空 = 学生看到「AI 讲了半天、白板一片空白」）
        dropped = drop_invalid_board_ops(response)
        if dropped:
            log.warning("dropped {} invalid board op(s), kept {}",
                        dropped, len(response.board_ops))
        return {"response": response}

    def solve(ctx: dict[str, Any]) -> dict[str, Any]:
        response: AIResponse = ctx["response"]
        # 仅存在「含等号的等式」时生效；require_answer 避免把概念公式（如 ax+b=0）
        # 误判为「解不唯一」而污染讲解话术
        math_solver.apply_verification(response, supervisor.teacher.board, require_answer=True)
        return {"response": response}

    return TaskGraph.from_specs(
        ANSWER_GRAPH,
        ANSWER_SPECS,
        {"retrieve": retrieve, "generate": generate, "validate": validate,
         "degrade": degrade, "solve": solve},
        session=getattr(supervisor, "session", None),   # 活动日志按会话归组（见 core/activity_log.py）
    )


# ------------------------------------------------------------------ 判题流水线
JUDGE_SPECS: tuple[NodeSpec, ...] = (
    NodeSpec(
        name="evaluate",
        summary="判定学生对追问/模拟题的作答（等式题 sympy 精确判定，其余交 LLM，失败退字符串比较）",
        impl="TeacherAgent.evaluate_answer",
        inputs=("turn", "pending_question"),
        outputs=("verdict",),
        deps=(),
        requires=("settings.interaction_max_retries", "core.math_solver.available"),
    ),
    NodeSpec(
        name="decide",
        summary="按判定结果决定反馈话术：答对收起题目 / 答错可重试（带提示）/ 超限公布答案；"
                "首次答错顺带写一条学情记忆（错题）",
        impl="Supervisor._ask_followup 的判题分支",
        inputs=("verdict", "pending_question"),
        outputs=("response", "allow_retry", "expected"),
        deps=("evaluate",),
        requires=("settings.interaction_max_retries", "settings.memory_enable（学情记忆）"),
    ),
    NodeSpec(
        name="announce",
        summary="播报反馈并推送 ai_check 事件（含对错、是否可重试、标准答案）",
        impl="Supervisor._speak_async + SessionState.push_event",
        inputs=("response", "verdict", "allow_retry", "expected"),
        outputs=(),
        deps=("decide",),
        requires=("models.tts.get_tts",),
    ),
)


def build_judge_graph(supervisor: "Supervisor") -> TaskGraph:
    """构建判题流水线图（由 ``Supervisor._turn_answer_check`` 调用）。"""

    async def evaluate(ctx: dict[str, Any]) -> dict[str, Any]:
        pq = ctx.get("pending_question") or {}
        verdict = await supervisor.teacher.evaluate_answer(
            str(pq.get("question") or ""),
            str(pq.get("expected") or ""),
            ctx["turn"].text,
            options=pq.get("options"),
        )
        if not verdict.get("correct"):
            expected = str(pq.get("expected") or "").strip()
            explanation = await supervisor.teacher.answer(
                StudentTurn(text=(
                    f"请直接讲解这道题，不要再次提问学生。题目：{pq.get('question') or ''}。"
                    f"标准答案：{expected}。学生回答：{ctx['turn'].text}。"
                    "先公布答案，再逐步解释依据和计算过程。不要生成追问。"
                )),
                supervisor.session,
            )
            verdict = dict(verdict)
            verdict["explanation"] = explanation
        return {"verdict": verdict}

    def decide(ctx: dict[str, Any]) -> dict[str, Any]:
        pq = dict(ctx.get("pending_question") or {})
        verdict = dict(ctx.get("verdict") or {})
        expected = str(pq.get("expected") or "")
        correct = bool(verdict.get("correct"))
        feedback = str(verdict.get("feedback") or "").strip()
        hint = str(verdict.get("hint") or "").strip() or str(pq.get("hint") or "").strip()
        # 互动题只核对一次：第一次答错就立即公布答案并完成讲解，
        # 不把学生困在“再试一次”的等待状态里。
        allow_retry = False
        if correct:
            speech = feedback or "答对啦，真棒！"
            board_ops: list[Any] = []
            supervisor.session.pending_question = None
        else:
            explanation = hint or "我们把题目中的条件代回去检查，就能得到这个答案。"
            explanation_response = verdict.get("explanation")
            prefix = f"这题的正确答案是：{expected}。解题思路是："
            if isinstance(explanation_response, AIResponse) and explanation_response.speech:
                speech = prefix + explanation_response.speech
                board_ops = explanation_response.board_ops
            else:
                speech = prefix + explanation
                board_ops = []
            supervisor.session.pending_question = None
            supervisor._record_mistake_memory(
                str(pq.get("question") or ""), expected, str(ctx["turn"].text or ""),
            )
        if correct or not allow_retry:
            supervisor.session.quiz = None      # 答对或公布答案后收起当前题目
            supervisor.session.scratch["suppress_idle_until_input"] = True
        # 数字人手势 + 形象状态编排：答对点头并表扬、答错摇头（可重试）或摊手
        gesture = "nod" if correct else ("shake" if allow_retry else "shrug")
        mood = "praising" if correct else "thinking"
        return {
            "response": AIResponse(
                speech=speech,
                board_ops=board_ops,
                meta={"gesture": gesture, "mood": mood},
            ),
            "allow_retry": allow_retry,
            "expected": expected,
        }

    async def announce(ctx: dict[str, Any]) -> dict[str, Any]:
        response: AIResponse = ctx["response"]
        await supervisor._speak_async(response)  # noqa: SLF001 - 复用同一播报实现
        supervisor.session.push_event(
            "ai_check",
            {
                "correct": bool((ctx.get("verdict") or {}).get("correct")),
                "feedback": response.speech,
                "expected": ctx.get("expected") or "",
                "allow_retry": bool(ctx.get("allow_retry")),
            },
        )
        return {}

    return TaskGraph.from_specs(
        JUDGE_GRAPH,
        JUDGE_SPECS,
        {"evaluate": evaluate, "decide": decide, "announce": announce},
        session=getattr(supervisor, "session", None),
    )


# ------------------------------------------------------------------ 出题流水线
QUIZ_SPECS: tuple[NodeSpec, ...] = (
    NodeSpec(
        name="generate",
        summary="按学科 + 知识点（默认上一题主题）出题：真题库（exercise_item）优先，"
                "无题可用回退 LLM，再不可用回退规则题",
        impl="TeacherAgent.generate_quiz",
        inputs=("subject", "topic", "grade"),
        outputs=("quiz_item",),
        deps=(),
        requires=("settings.course_quiz_enable", "models.llm.get_llm"),
    ),
    NodeSpec(
        name="publish",
        summary="写入待答状态（pending_question/quiz/quiz_count）并推进出题节奏、推 ai_quiz；"
                "播报带过渡语（Supervisor._quiz_lead，先说明为何做题）",
        impl="Supervisor.on_quiz_request 的入库分支",
        inputs=("quiz_item",),
        outputs=("response",),
        deps=("generate",),
        requires=("settings.course_quiz_interval_minutes",),
    ),
    NodeSpec(
        name="announce",
        summary="播报题目（mood=waiting, gesture=point），进入等待作答",
        impl="Supervisor._speak_async",
        inputs=("response",),
        outputs=(),
        deps=("publish",),
        requires=("models.tts.get_tts",),
    ),
)


def build_quiz_graph(supervisor: "Supervisor") -> TaskGraph:
    """构建出题流水线图（由 ``Supervisor.on_quiz_request`` 调用）。"""

    async def generate(ctx: dict[str, Any]) -> dict[str, Any]:
        quiz = await supervisor.teacher.generate_quiz(
            str(ctx.get("subject") or ""),
            str(ctx.get("topic") or ""),
            grade=str(ctx.get("grade") or ""),
        )
        return {"quiz_item": quiz}

    def publish(ctx: dict[str, Any]) -> dict[str, Any]:
        quiz: QuizItem = ctx["quiz_item"]
        supervisor.session.quiz = quiz.model_dump()
        supervisor.session.quiz_count += 1
        supervisor._mark_quiz_done()  # noqa: SLF001 - 出题节奏：本次出题时间入账
        supervisor.session.pending_question = {
            "question": quiz.question,
            "expected": quiz.answer,
            "hint": quiz.hint,
            "options": list(quiz.options),
            "attempts": 0,
            "source_title": quiz.topic,
            "kind": "quiz",
        }
        supervisor.session.followup_asked += 1
        supervisor.session.push_event(
            "ai_quiz",
            {
                "qid": quiz.qid,
                "type": quiz.type,
                "question": quiz.question,
                "options": list(quiz.options),
                "index": supervisor.session.quiz_count,
                "subject": supervisor.session.subject,
                "topic": quiz.topic,
            },
        )
        return {
            "response": AIResponse(
                # 过渡语（「我们刚讲了…，来做一道题检验一下」）先说清为什么现在做题，
                # 避免讲解中途突然甩一道题显得莫名其妙（见 Supervisor._quiz_lead）。
                speech=f"{supervisor._quiz_lead(quiz)}：{quiz.question}",  # noqa: SLF001
                meta={"mood": "waiting", "gesture": "point"},
            )
        }

    async def announce(ctx: dict[str, Any]) -> dict[str, Any]:
        await supervisor._speak_async(ctx["response"])  # noqa: SLF001 - 复用同一播报实现
        return {}

    return TaskGraph.from_specs(
        QUIZ_GRAPH,
        QUIZ_SPECS,
        {"generate": generate, "publish": publish, "announce": announce},
        session=getattr(supervisor, "session", None),
    )


#: 本 agent 的图名 → 节点声明（供 ``agents.nodes`` 注册表汇总）
GRAPH_SPECS: dict[str, tuple[NodeSpec, ...]] = {
    ANSWER_GRAPH: ANSWER_SPECS,
    JUDGE_GRAPH: JUDGE_SPECS,
    QUIZ_GRAPH: QUIZ_SPECS,
}
