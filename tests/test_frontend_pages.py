"""任务5/6 专项测试：前端两页的「结构契约」（静态断言，无需浏览器）。

前端是原生 HTML + 内联 JS，没有构建与框架，因此用结构契约测试守住关键锚点：
* 任务5 学科选择页：学科卡片、后端学科清单、跳转时透传学科参数；
* 任务6 1v1 辅导页：学科/阶段/倒计时/结束按钮、教师形象状态、协议事件、模拟题作答，
  且**已删除的体态检测功能不得回流**。
"""
from __future__ import annotations

from pathlib import Path

CLIENTS = Path(__file__).resolve().parent.parent / "clients"
SELECT_PAGE = CLIENTS / "subject_select" / "index.html"
TUTOR_PAGE = CLIENTS / "student_tablet" / "index.html"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------- 任务5：学科选择页
def test_subject_select_page_contract() -> None:
    assert SELECT_PAGE.is_file(), "学科选择页缺失"
    html = _read(SELECT_PAGE)

    assert "/api/subjects" in html, "学科清单应来自后端接口"
    assert "FALLBACK" in html, "应保留后端不可用时的兜底清单，避免白屏"
    assert "开始辅导" in html

    # 卡片 -> 1v1 辅导页，并透传学科参数
    assert "student_tablet/index.html?subject=" in html
    assert "encodeURIComponent" in html


# ---------------------------------------------------------------- 任务6：1v1 辅导页
def test_subject_page_cards_have_dark_readable_background() -> None:
    """回归保护：卡片必须是深色底。

    历史问题：用「双背景 + background-clip: padding-box, border-box」做渐变发丝边框时，
    内层填充是半透明的 → 外层浅灰渐变从整张卡片透出，卡片变浅、浅灰文字直接看不见。
    """
    html = _read(SELECT_PAGE)
    assert "background-clip:padding-box, border-box" not in html, "不要再用双背景裁剪做渐变边框（曾导致文字不可读）"
    assert "background-origin:border-box" not in html
    assert "rgba(26,36,62,.95)" in html, "缺少深色卡底"
    assert "--muted:#a3b4cc" in html, "说明文字颜色应保持足够亮度"


def test_tutor_page_course_hud_contract() -> None:
    """1v1 辅导页的课程 HUD 契约。

    **注意**：阶段 chip / 倒计时 / 「结束课程」按钮 / 阶段条**已被有意停用**
    （页面里保留了「已停用…恢复方式…」注释），原 ``course-bar`` 整行也已删除、
    高度让给白板。所以这里**不再**断言它们存在——恢复那套 HUD 时，
    请把这些断言一并加回来。
    """
    assert TUTOR_PAGE.is_file(), "1v1 辅导页缺失"
    html = _read(TUTOR_PAGE)

    # 当前生效：学科徽章（已从 course-bar 移入顶栏）
    assert "chip-subject" in html, "辅导页缺少学科徽章"

    # 教师形象状态（打招呼/讲解中/思考中/表扬/等待回答）；数字人画面只有口型层与眨眼层
    for marker in ("status-mood", "MOOD_LABELS", "greeting", "praising", "waiting", "idle-blink"):
        assert marker in html, f"辅导页缺少教师状态相关：{marker}"


def test_tutor_page_protocol_and_quiz_contract() -> None:
    html = _read(TUTOR_PAGE)

    # 课程策略与上行事件
    assert "/api/course/policy" in html
    assert "course_start" in html, "缺少开课上行事件（用于触发老师主动打招呼）"
    # course_phase（阶段上报）随倒计时 HUD 一起停用，不再断言
    assert "init.subject = SUBJECT" in html, "init 应带上学科参数"

    # 下行事件处理
    for marker in ("ai_quiz", "course_state"):
        assert marker in html, f"辅导页未处理下行事件：{marker}"

    # 模拟题作答：复用既有作答卡片与提交链路
    assert "renderQuizCard" in html
    assert "submitQuestion" in html
    assert 'id = "ask-card"' in html or 'id="ask-card"' in html


def test_tutor_page_start_gate_contract() -> None:
    """开课闸门：解决浏览器自动播放拦截 + 未选学科时的「一片安静」。"""
    html = _read(TUTOR_PAGE)
    for marker in ("start-gate", "btn-start", "start-title", "start-desc"):
        assert marker in html, f"辅导页缺少开课闸门元素：{marker}"
    # 闸门逻辑：等 ready + 策略就绪后弹出；点击才开始课程并解锁音频
    assert "maybeShowStartGate" in html
    assert "showStartGate" in html
    assert "wsReady" in html
    assert "audioCtx.resume()" in html
    # 未选学科时应引导回学科选择页
    assert "去选择学科" in html
    assert "subject_select/index.html" in html


def test_pages_degrade_gracefully_without_https() -> None:
    """分享给他人时：HTTP 访问（非安全上下文）下浏览器禁用麦克风/摄像头。

    页面必须显式告知并禁用相关控件，而不是「点了没反应」。
    """
    tutor = _read(TUTOR_PAGE)
    assert "isSecureContext" in tutor
    assert "net-banner" in tutor
    assert "applyInsecureMode" in tutor
    assert "语音需 HTTPS" in tutor
    assert 'classList.add("locked")' in tutor
    # 非安全上下文下不应再徒劳调用摄像头
    assert "if (SECURE_CTX) {" in tutor

    select = _read(SELECT_PAGE)
    assert "https-hint" in select
    assert "isSecureContext" in select


def test_tutor_page_has_no_removed_features() -> None:
    """回归保护：上一轮删除的体态检测/学段选择器不得回流。"""
    html = _read(TUTOR_PAGE)
    for marker in ("btn-vision", "posture-hud", "status-posture", "video_frame", "cfg-stage", "cfg-grade", "cfg-subject"):
        assert marker not in html, f"已删除的功能回流到辅导页：{marker}"
