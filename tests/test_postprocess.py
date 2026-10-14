"""P3 测试：答疑产物收尾的唯一实现（``core/postprocess``）与总控的「恰好一次」。
这块的重点不是「能不能验算」，而是三件容易做错的事：

1. **只丢非法白板指令，不整块清空** —— 历史 bug：一处坐标越界就把白板清成空白；
2. **每回合恰好收尾一次** —— 收敛前流式链路会跑两遍，更正说明被重复追加；
3. **收尾出任何错都不能中断播报** —— 「校验挂了」绝不能变成「老师不说话了」。

> 来源：本文件由云端分叉的 ``tests/test_postprocess.py`` 移植而来。
> 云端用 ``agents.teaching.agent`` 的 ``solve`` 节点作为收尾入口（A2A 形态）；
> 本仓库的收尾入口是 ``Supervisor._finalize_streamed_response``（直连形态），
> 因此与 A2A 相关的那几条用例改写成对本地入口的等价断言。
"""
from __future__ import annotations

from agents.supervisor import Supervisor
from agents.teacher.agent import TeacherAgent
from config.constants import BoardOp
from config.settings import settings
from core.hint_homework import HintLevel
from core.postprocess import postprocess_response
from core.safety_audit import SafetyAudit
from core.state.models import AIResponse, StudentTurn, WhiteboardOp
from core.state.session import SessionState
from models.llm import MockLLM
from models.tts import MockTTS


def _board_op(text: str, x: float = 0.5, y: float = 0.5) -> WhiteboardOp:
    return WhiteboardOp(op=BoardOp.TEXT, text=text, x=x, y=y)


# ============================================================ postprocess 单元
def test_only_invalid_board_ops_are_dropped() -> None:
    """回归历史 bug：一处坐标越界不能把整块白板清空。"""
    response = AIResponse(
        speech="我们看这道题",
        board_ops=[_board_op("合法的一行"), _board_op("越界的一行", x=3.0)],
    )
    postprocess_response(response)
    texts = [op.text for op in response.board_ops]
    assert texts == ["合法的一行"], "只该丢非法那条，其余必须保留"


def test_verification_correction_is_appended() -> None:
    """模型给了错答案（x=9），sympy 必须纠正成正确解 x=5。"""
    response = AIResponse(
        speech="答案是 x=9。",
        meta={"equation": "3x+5=20", "answer": "x=9"},
    )
    postprocess_response(response)
    blob = response.speech + " " + " ".join(str(s) for s in (response.meta.get("steps") or []))
    assert "x=5" in blob, "错误答案必须被 sympy 纠正"
    assert response.meta.get("verified") is False


def test_postprocess_never_raises() -> None:
    """校验/验算出任何错都只能降级，不能把异常抛给播报链路。"""

    class _BoomValidator:
        def validate(self, *a, **kw):
            raise RuntimeError("validator boom")

    response = AIResponse(speech="你好")
    postprocess_response(response, validator=_BoomValidator())  # type: ignore[arg-type]
    assert response.speech == "你好"


def test_postprocess_is_idempotent_for_clean_response() -> None:
    """干净的产物跑两遍应该完全不变（这正是不该重复跑两遍的原因）。"""
    response = AIResponse(speech="分数的分子和分母同时乘 2，大小不变。")
    postprocess_response(response)
    once = response.speech
    postprocess_response(response)
    assert response.speech == once


# ============================================================ 总控收尾入口
def _supervisor(monkeypatch, tmp_path) -> Supervisor:
    monkeypatch.setattr(settings, "digital_human_enable", False)
    monkeypatch.setattr(settings, "board_first_mode", "off")
    state = SessionState(session_id="p3", student_id="stu")
    sup = Supervisor(state, teacher=TeacherAgent(llm=MockLLM()))
    sup.tts = MockTTS()
    sup.safety_audit = SafetyAudit(tmp_path / "safety.jsonl")
    return sup


def test_finalize_uses_the_shared_postprocess(monkeypatch, tmp_path) -> None:
    """流式收尾必须复用 ``core/postprocess``，而不是各写一份。"""
    sup = _supervisor(monkeypatch, tmp_path)
    response = AIResponse(
        speech="答案是 x=9。",
        board_ops=[_board_op("第一步：移项"), _board_op("越界", y=2.0)],
        meta={"equation": "3x+5=20", "answer": "x=9"},
    )
    sup._finalize_streamed_response(response, HintLevel.CONCEPT)  # noqa: SLF001

    # 非法白板指令被丢、合法那条保留（与 postprocess_response 单元口径一致）
    texts = [op.text for op in response.board_ops]
    assert "第一步：移项" in texts, "合法白板指令必须保留"
    assert "越界" not in texts, "非法白板指令必须被丢弃"
    blob = response.speech + " ".join(response.meta.get("steps") or [])
    assert "x=5" in blob, "总控收尾也必须做验算"


def test_finalize_calls_postprocess_exactly_once(monkeypatch, tmp_path) -> None:
    """收尾入口对同一个产物只调用一次共享收尾 —— 收敛前会跑两遍，更正说明被重复追加。"""
    sup = _supervisor(monkeypatch, tmp_path)
    calls: list[str] = []

    import agents.supervisor.agent as sup_module

    original = sup_module.postprocess_response

    def _counting(response, **kwargs):
        calls.append(str(getattr(response, "speech", "")))
        return original(response, **kwargs)

    monkeypatch.setattr(sup_module, "postprocess_response", _counting)

    response = AIResponse(speech="答案是 x=9。", meta={"equation": "3x+5=20", "answer": "x=9"})
    sup._finalize_streamed_response(response, HintLevel.CONCEPT)  # noqa: SLF001
    assert len(calls) == 1, f"收尾被调用了 {len(calls)} 次，应当恰好 1 次"
    # 更正说明只追加一次，不重复
    assert response.speech.count("x=5") <= 1


async def test_postprocess_still_runs_on_plain_answer_path(monkeypatch, tmp_path) -> None:
    """回退直连时收尾也必须发生（否则校验与验算就悄悄没了）。"""
    sup = _supervisor(monkeypatch, tmp_path)
    response = AIResponse(
        speech="答案是 x=9。", meta={"equation": "3x+5=20", "answer": "x=9"},
    )
    sup._finalize_streamed_response(response, HintLevel.CONCEPT)  # noqa: SLF001
    blob = response.speech + " ".join(response.meta.get("steps") or [])
    assert "x=5" in blob, "直连路径也必须做验算"
