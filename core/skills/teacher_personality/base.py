"""教师人格 Skill 基础接口。

设计原则（见《九科独立教师 Personality Skill 修改方案》）：

* 每个学科各自拥有 **独立** 的教师人格 Skill，互不共享人格内容；
* 本模块只定义「接口」与「平台底线规则」，不定义任何具体学科人格；
* ``GLOBAL_TEACHER_RULES`` 是平台底线，不属于人格，每位教师都必须遵守。

每个具体教师的提示词由 ``build_system_prompt()`` 自行返回，注册表只做查找，
TeacherAgent 只做通用调用（不写 ``if subject == "math"`` 这类分支）。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - 仅类型标注
    from core.state.session import SessionState

# 平台底线：所有教师共享（通用教师规则 ≠ 人格 Skill）
GLOBAL_TEACHER_RULES = """通用教师规则（平台底线，任何教师人格都不得违反）：
1. 面向中小学生解释，语言通俗、口语化，适合直接朗读；
2. 不训斥、不羞辱、不贬低学生，保持鼓励与尊重；
3. 不编造事实，不确定时明确说明并给出可验证的思路；
4. 不鼓励学生自行尝试危险操作，涉及实验安全必须提醒；
5. 只围绕当前学科边界作答；问题明显超出本学科时温和说明，并欢迎学生回到本学科提问；
6. 保持既有 JSON 输出格式与字段约定，不因为角色扮演而违反系统规则。"""


class TeacherPersonalitySkill:
    """教师人格 Skill 抽象接口（每个学科分别实现独立人格）。"""

    #: 全局唯一技能标识，如 ``teacher.math.yanli``
    skill_id: str = ""
    #: 对应学科 id，需与 ``data/curriculum/subjects.json`` 的 ``id`` 一致
    subject_id: str = ""
    #: 学科名称，如「数学」
    subject_name: str = ""
    #: 教师形象名，如「严谨的数学老师」
    display_name: str = ""
    #: 该教师的人格关键词（日志 / 调试用）
    persona: str = ""
    #: 语气与节奏（日志 / 调试用）
    tone: str = ""

    # ------------------------------------------------------------ 必选人格行为
    def build_system_prompt(self) -> str:
        """返回该教师 **独立** 的人格与教学行为提示词。"""
        raise NotImplementedError

    def opening_style(self) -> str:
        """定义教师的开场方式。"""
        raise NotImplementedError

    def correction_style(self) -> str:
        """定义教师的纠错方式。"""
        raise NotImplementedError

    def encouragement_style(self) -> str:
        """定义教师的鼓励方式。"""
        raise NotImplementedError

    def confused_student_strategy(self) -> str:
        """定义学生说「我不会 / 听不懂」时的处理方式。"""
        raise NotImplementedError

    # ------------------------------------------------------------ 二期行为（可选覆盖）
    def questioning_style(self) -> str:
        """定义教师的追问方式。"""
        return "用一个学生能直接回答的小问题推进，一次只问一件事。"

    def praise_style(self) -> str:
        """定义学生答对时的反馈方式。"""
        return "具体指出学生答对的部分，再顺势推进到下一步。"

    def repeated_error_strategy(self) -> str:
        """定义学生连续答错时的处理方式。"""
        return "降低难度，改为示范一个小步骤，先让学生重获成功体验。"

    def whiteboard_style(self) -> str:
        """定义白板内容的组织方式。"""
        return "按步骤分行板书，每行是一句完整、适合朗读的话。"

    def tone_rhythm(self) -> str:
        """定义语气与节奏要求。"""
        return "语速平稳，一句一个意思，长句拆短。"

    # ------------------------------------------------------------ 开场语
    def greeting(self, where: str = "") -> str:
        """进页面打招呼的口播稿（各学科可覆盖为更有个人色彩的版本）。"""
        target = f"{where}" if where else "今天的课程"
        return f"同学你好！我是你的{self.display_name}，今天我们一起学{target}，有不会的题随时发给我～"

    # ------------------------------------------------------------ 汇总
    def behaviors(self) -> dict[str, str]:
        """汇总全部人格行为（注册表自检 / 单测用）。"""
        return {
            "opening_style": self.opening_style(),
            "questioning_style": self.questioning_style(),
            "encouragement_style": self.encouragement_style(),
            "praise_style": self.praise_style(),
            "correction_style": self.correction_style(),
            "confused_student_strategy": self.confused_student_strategy(),
            "repeated_error_strategy": self.repeated_error_strategy(),
            "whiteboard_style": self.whiteboard_style(),
            "tone_rhythm": self.tone_rhythm(),
        }

    # ------------------------------------------------------------ 便利方法
    def describe(self) -> dict[str, Any]:
        """教师身份摘要（下发前端 / 日志）。"""
        return {
            "skill_id": self.skill_id,
            "subject_id": self.subject_id,
            "subject_name": self.subject_name,
            "display_name": self.display_name,
            "persona": self.persona,
            "tone": self.tone,
        }

    @classmethod
    def session_scope(cls, session: "SessionState") -> str:
        """从会话状态中提取「学段 + 学科」文本（供子类组装提示词）。"""
        scratch = getattr(session, "scratch", {}) or {}
        grade = str(scratch.get("grade", "") or "").strip() or "中小学"
        subject = str(scratch.get("subject", "") or "").strip()
        return f"{grade}{subject}" if subject else grade
