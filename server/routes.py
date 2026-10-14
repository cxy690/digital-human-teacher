"""REST 路由：健康检查、工具清单、数字人源信息等。"""
from __future__ import annotations

import asyncio
import re
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from config.settings import settings
from config.subjects import SUBJECTS, SUBJECT_BY_KEY, SUBJECT_BY_NAME
from core import course_session
from core.auth.deps import current_user, require_developer
from core.auth.security import encode_token
from core.auth.service import AuthError, authenticate, register_user
from core.avatar_source import get_avatar_source_info
from core.safety_audit import safety_audit
from document_parsing.pipeline import QuestionParser
from mcp_tools.registry import build_default_registry

router = APIRouter()


def _personality_info(subject: str, *, detail: bool = False) -> dict | None:
    """学科 -> 教师人格技能（``core/skills/teacher_personality``）信息。

    ``subject`` 可传中文名（数学）或英文 key（math）。技能关闭、学科未知或加载
    异常时一律返回 ``None``，调用方按「没有人格」处理，不影响接口可用性。
    ``detail=True`` 时额外带上九个行为维度与完整 system prompt（调试/演示用）。
    """
    if not settings.teacher_personality_enable:
        return None
    try:
        from core.skills.teacher_personality import get_skill_registry

        raw = (subject or "").strip()
        meta = SUBJECT_BY_NAME.get(raw) or SUBJECT_BY_KEY.get(raw)
        subject_id = str((meta or {}).get("key") or raw)
        skill = get_skill_registry().get_for_subject(subject_id)
        if skill is None:
            return None
        info = dict(skill.describe())
        if detail:
            info["behaviors"] = skill.behaviors()
            info["system_prompt"] = skill.build_system_prompt()
        return info
    except Exception:  # noqa: BLE001 - 人格技能异常不应影响接口
        return None


@router.get("/health")
async def health() -> dict:
    """健康检查（附带 RAG 检索后端：一眼确认向量检索是否生效）。"""
    info: dict = {"status": "ok", "app": settings.app_name, "env": settings.env}
    try:
        from server.websocket import harness  # noqa: PLC0415 - 延迟导入，避免模块级循环依赖

        retriever = getattr(harness, "retriever", None)
        if retriever is not None:
            info["rag"] = str(getattr(retriever, "backend_name", "on"))
    except Exception:  # noqa: BLE001 - 附加信息失败不应影响健康检查
        pass
    return info


#: 诊断窗口的合法范围（防呆：接口是公开的，不能让一次请求去扫几年的数据）
_DIAG_DAYS_MIN, _DIAG_DAYS_MAX = 1, 90


@router.get("/api/health/diagnosis")
async def health_diagnosis(days: int = 7, baseline_days: int = 0) -> dict:
    """L1 健康诊断：四维画像 + 退化结论 + 建议（实现见 ``core/health.py``）。

    **与 ``/health`` 不是一回事，别互相替代**：

    * ``/health`` 回答「进程还活着吗」—— 给探活/负载均衡用，永远是 ok 或连不上；
    * 本接口回答「**系统在变好还是变坏**」—— 给监控台与运维用，
      刻意不放进 ``/health``：健康检查必须快而稳，这里要读裁决日志、经验文件、
      评测基线，慢且会随数据量增长，混在一起会让探活变脆。

    参数：``days`` 当前窗口天数（默认 7，限 1~90）；
    ``baseline_days`` 基线窗口天数（0 = 与当前窗口等长，即与紧邻的前一个窗口比）。
    """
    days = max(_DIAG_DAYS_MIN, min(_DIAG_DAYS_MAX, int(days)))
    base = max(0, min(_DIAG_DAYS_MAX, int(baseline_days))) or None
    try:
        from core.health import as_dict, diagnose  # noqa: PLC0415 - 延迟导入

        # 诊断要读文件（裁决 / 经验 / 评测基线），放线程池避免阻塞事件循环 ——
        # 数据量小的时候无所谓，但它是公开接口，不能被一次慢 IO 拖住整个应用。
        diag = await asyncio.to_thread(diagnose, days=days, baseline_days=base)
        return as_dict(diag)
    except Exception as err:  # noqa: BLE001 - 诊断失败不该 500，前端要能降级显示
        return {
            "error": f"{type(err).__name__}: {err}",
            "level": "unknown",
            "window": {"days": days, "baseline_days": base or days},
            "findings": [],
        }


@router.get("/api/health/failures")
async def health_failures(days: int = 7, top: int = 5) -> dict:
    """L3 失败聚类：把「一批失败」聚成**可追的缺陷**（实现见 ``core/failure_cluster.py``）。

    与 ``/api/health/diagnosis`` 的分工：

    * ``diagnosis`` 回答「**系统在变好还是变坏**」（趋势，L1）；
    * 本接口回答「**坏了什么、该去改哪儿**」（定位，L3）——
      它把「失败原因 + 卡住的那一步 + 偏向哪个场景/学科」压成一个簇，
      而不是给四张互不相干的频次表让人自己拼。

    默认**不调模型**（纯统计，快且免费）；需要根因结论时用命令行
    ``python scripts/failure_report.py --analyze``。
    """
    days = max(_DIAG_DAYS_MIN, min(_DIAG_DAYS_MAX, int(days)))
    top = max(1, min(20, int(top)))
    try:
        from core.failure_cluster import report  # noqa: PLC0415 - 延迟导入

        # 要读裁决文件，放线程池避免阻塞事件循环（与 diagnosis 同一理由）
        return await asyncio.to_thread(report, days, top=top)
    except Exception as err:  # noqa: BLE001 - 排障接口失败不该 500
        return {"error": f"{type(err).__name__}: {err}", "window_days": days, "clusters": []}


# --------------------------------------------------------------------------- 身份认证
class RegisterBody(BaseModel):
    """注册请求体。"""

    username: str
    password: str
    role: str = "student"
    display_name: str = ""
    grade: str = ""


class LoginBody(BaseModel):
    """登录请求体。"""

    username: str
    password: str


def _issue_token(user: dict) -> str:
    """签发登录令牌（HS256 JWT，主体为 ``users.id``）。"""
    return encode_token(
        {"sub": user["id"], "role": user["role"], "username": user["username"]},
        settings.auth_secret,
        ttl_seconds=settings.auth_token_ttl_minutes * 60,
    )


@router.post("/api/auth/register")
async def auth_register(body: RegisterBody) -> dict:
    """注册并直接签发令牌（``AUTH_ALLOW_REGISTER=false`` 时关闭）。"""
    if not settings.auth_enable:
        raise HTTPException(status_code=404, detail="身份体系未启用")
    if not settings.auth_allow_register:
        raise HTTPException(status_code=403, detail="当前未开放注册")
    try:
        user = await asyncio.to_thread(
            register_user,
            body.username,
            body.password,
            role=body.role,
            display_name=body.display_name,
            grade=body.grade,
        )
    except AuthError as err:
        raise HTTPException(status_code=err.code, detail=err.message) from err
    return {"user": user, "token": _issue_token(user)}


@router.post("/api/auth/login")
async def auth_login(body: LoginBody) -> dict:
    """账号密码登录，返回令牌与用户信息。"""
    if not settings.auth_enable:
        raise HTTPException(status_code=404, detail="身份体系未启用")
    try:
        user = await asyncio.to_thread(authenticate, body.username, body.password)
    except AuthError as err:
        raise HTTPException(status_code=err.code, detail=err.message) from err
    return {"user": user, "token": _issue_token(user)}


@router.get("/api/auth/me")
async def auth_me(user: dict = Depends(current_user)) -> dict:
    """返回当前登录用户（需 ``Authorization: Bearer <token>``）。"""
    return {"user": user}


@router.get("/api/subjects")
async def list_subjects() -> dict:
    """学科清单：学科选择页据此渲染卡片（新增学科只改 config/subjects.py）。

    每个学科额外带上 ``personality``（该学科教师人格摘要），前端可据此展示
    「严谨的数学老师」这类标签；人格未启用时为 ``None``。
    """
    subjects = [
        {**item, "personality": _personality_info(item["key"])}
        for item in SUBJECTS
    ]
    return {"subjects": subjects}


@router.get("/api/personality")
async def personality_info(subject: str = "", detail: bool = False) -> dict:
    """按学科返回教师人格技能信息（detail=true 时含行为维度与完整提示词）。

    供前端展示教师身份标签、以及调试「这一科到底加载了哪个人格」。
    """
    info = _personality_info(subject, detail=detail)
    return {
        "subject": subject,
        "enabled": settings.teacher_personality_enable,
        "personality": info,
    }


@router.get("/api/course/policy")
async def course_policy() -> dict:
    """1v1 辅导课策略：总时长与阶段划分（前端倒计时 / 阶段条据此显示）。"""
    return {
        "enable": settings.course_enable,
        "duration_minutes": settings.course_duration_minutes,
        "duration_seconds": course_session.duration_seconds(),
        "quiz_enable": settings.course_quiz_enable,
        "phases": course_session.tutorial_plan(),
    }


@router.get("/api/courseware")
async def list_courseware(reload: bool = False, subject: str = "") -> dict:
    """课件清单（PPT 讲解用）：前端据此渲染「选课件」下拉。

    ``subject``：按学科过滤 —— **只返回该学科的课件**（物理课就只看到物理）。
    可传中文名（物理）或英文 key（physics），内部统一归一化后比较；
    不传则返回全部（首页/未选学科时的兜底）。

    ``reload=true`` 重新扫描磁盘 —— 往 ``data/courseware/`` 放了新 PPT 之后
    调一次即可让新课件可见，无需重启服务。默认走进程内缓存，避免每次
    打开下拉都重新解析一遍所有 pptx（10 份课件要好几秒）。
    """
    from core.courseware.loader import get_store

    store = get_store()
    # 扫描是**同步磁盘活**：231 份课件全量解析要 40~60 秒。必须丢到线程里，
    # 否则事件循环会被卡死——期间所有请求排队、WebSocket 心跳超时直接断连，
    # 学生看到的就是「一点课件，整个页面僵住」。
    if reload:
        await asyncio.to_thread(store.reload)
    decks = await asyncio.to_thread(store.list_decks)
    raw = (subject or "").strip()
    if raw:
        meta = SUBJECT_BY_NAME.get(raw) or SUBJECT_BY_KEY.get(raw)
        want = str((meta or {}).get("name") or raw)
        # 只认「标了该学科」的课件：subject 为空的通用课件不出现在学科课堂里，
        # 否则数学课可能冒出语文 PPT（宁可少显示，也不要串台）。
        decks = [d for d in decks if str(d.get("subject") or "").strip() in (want, raw)]

    return {"decks": decks, "root": str(store.root), "subject": raw}


@router.get("/api/courseware/{deck_id}")
async def courseware_detail(deck_id: str) -> dict:
    """课件详情：全部页（含页图片可访问 URL 与可标注元素框）。

    前端可以只用它做预览；真正的翻页由 WebSocket 下发 ``slide`` 指令驱动。
    """
    from core.courseware.loader import get_store

    store = get_store()
    # 同上：冷启动时 get_deck 会触发全量扫描，必须放线程，别卡住事件循环
    deck = await asyncio.to_thread(store.get_deck, deck_id)
    if deck is None:
        return {"ok": False, "error": "课件不存在"}
    slides = [
        {**slide.model_dump(), "image_url": store.page_image_url(deck.deck_id, slide.image)}
        for slide in deck.slides
    ]
    return {"ok": True, "deck": {**deck.model_dump(), "slides": slides}}


@router.get("/api/tools")
async def list_tools() -> dict:
    """列出可用工具（MCP 风格）。"""
    registry = build_default_registry(parser=QuestionParser())
    return {"tools": registry.specs()}


@router.get("/api/safety/events")
async def safety_events(limit: int = 50) -> dict:
    """安全事件（危机 / 敏感命中）最近记录，供教师端监控台复核。

    数据来自 ``data/raw/safety_events.jsonl``（见 :mod:`core.safety_audit`）。
    读文件是阻塞 IO，放线程里跑，避免拖住事件循环。
    """
    size = max(1, min(int(limit), 500))
    events = await asyncio.to_thread(safety_audit.recent, size)
    return {
        "enabled": settings.safety_audit_enable,
        "total": len(events),
        "events": events,
    }


# ---------------------------------------------------------------- 教师经验（Skill 自我更新）
def _experience_store():
    """进程级经验表（延迟导入，技能关闭时也不影响其它接口）。"""
    from core.skills.experience import get_experience_store

    return get_experience_store()


@router.get("/api/skills/experience")
async def list_skill_experience(teacher: str = "") -> dict:
    """查看某位老师沉淀的经验（含已停用条目），或全局概况。

    自我更新是自动的，这个接口是给人**看一眼和止损**用的：
    ``teacher`` 为空返回全局统计与老师列表；给了老师 key 则返回其全部条目。
    """
    store = _experience_store()
    if not (teacher or "").strip():
        return {"ok": True, **store.stats(), "teachers_list": store.list_teachers()}
    key = teacher.strip()
    return {
        "ok": True,
        "teacher": key,
        "active": [it.to_dict() for it in store.active(key)],
        "all": [it.to_dict() for it in store.all_items(key)],
    }


@router.get("/api/skills/experience/review")
async def review_skill_experience(teacher: str = "", limit: int = 10) -> dict:
    """L5 人工抽检台：按**风险**排出「最该看一眼」的经验。

    经验是系统自己学的、立刻就用，人只能事后 ``retire`` —— 所以整个治理环节里
    最稀缺的是**人的注意力**。按时间列「最近新增的」会把这几分钟花在绝大多数没问题
    的条目上；按风险排才能落在最可能出问题的地方。

    **它不阻断注入**（新经验照旧立刻可用）：把自动学习改成「先审后用」等于退化成
    人肉录入。这里做的是降低抽检成本，不是把学习停下来。

    风险信号与对应动作见 ``core/skills/experience/review.py``。
    """
    from core.skills.experience import review_queue  # noqa: PLC0415 - 延迟导入

    rows = await asyncio.to_thread(
        review_queue, teacher, limit=max(1, min(50, int(limit)))
    )
    return {"ok": True, "teacher": teacher or "(全部)", "count": len(rows), "items": rows}


class _ExperiencePatch(BaseModel):
    teacher: str
    id: str
    status: str = "retired"      # retired 停用 / active 恢复


@router.post("/api/skills/experience/status")
async def set_skill_experience_status(payload: _ExperiencePatch) -> dict:
    """停用 / 恢复一条经验（**回滚**入口）：坏经验能立刻止损，也能再恢复回来。"""
    store = _experience_store()
    if payload.status not in ("retired", "active"):
        return {"ok": False, "error": "status 只能是 retired / active"}
    changed = (
        store.restore(payload.teacher, payload.id)
        if payload.status == "active"
        else store.retire(payload.teacher, payload.id)
    )
    return {"ok": changed, "teacher": payload.teacher, "id": payload.id, "status": payload.status}


# ---------------------------------------------------------------- Agent 技能（模型可自主调用）
# 与上面的「教师经验」同一条治理思路：技能是模型自己决定调用的，人只能事后止损，
# 因此这里只提供「看一眼」与「停用/恢复」，不提供「先审后用」的闸门。


@router.get("/api/skills/agent")
async def list_agent_skills() -> dict:
    """列出全部 Agent 技能及其调用战绩（排障 / 抽检 / 前端调试面板用）。"""
    from core.skills.agent_skills import get_skill_registry, get_skill_usage_store  # noqa: PLC0415

    registry = get_skill_registry()
    usage = get_skill_usage_store()
    return {
        "ok": True,
        "enabled": settings.agent_skills_enable,
        "skills": [
            {**spec, "usage": usage.load(spec["name"]).to_dict()}
            for spec in registry.specs()
        ],
    }


@router.get("/api/skills/agent/review")
async def review_agent_skills(limit: int = 10) -> dict:
    """技能战绩抽检台：按**风险**排出「最该人工看一眼」的技能。

    与经验抽检台同一立场：**默认不阻断调用**（战绩差的技能照旧可用，
    打开 ``AGENT_SKILL_AUTO_RETIRE`` 才会自动停用）—— 自动学到的东西不该被
    系统自己静悄悄丢掉。风险信号与对应动作见
    :meth:`core.skills.agent_skills.usage.SkillUsageStore.review_queue`。
    """
    from core.skills.agent_skills import get_skill_usage_store  # noqa: PLC0415

    rows = await asyncio.to_thread(
        get_skill_usage_store().review_queue, limit=max(1, min(50, int(limit)))
    )
    return {"ok": True, "count": len(rows), "items": rows}


class _AgentSkillStatus(BaseModel):
    name: str
    status: str = "disabled"      # disabled 停用 / active 恢复


@router.post("/api/skills/agent/status")
async def set_agent_skill_status(payload: _AgentSkillStatus) -> dict:
    """停用 / 恢复一个 Agent 技能（**止损**入口：坏技能立刻不再暴露给模型）。"""
    from core.skills.agent_skills import get_skill_usage_store  # noqa: PLC0415

    if payload.status not in ("disabled", "active"):
        return {"ok": False, "error": "status 只能是 disabled / active"}
    changed = await asyncio.to_thread(
        get_skill_usage_store().set_status, payload.name.strip(), payload.status
    )
    return {"ok": changed, "name": payload.name, "status": payload.status}


# ---------------------------------------------------------------- 技能管理平台（仅开发人员）
# 事实源 = .agents/skills/<学科key>/SKILL.md（见 config/teacher_skills/md.py）。
# 门禁 = require_developer（登录账号 role=admin）：注册接口只开放 student/parent/teacher，
# admin 是内部账号（seed 预置/运维改库），天然区分「开发人员」与普通用户。
# 保存后清解析缓存（mtime 失效）→ 下一节课立即用新技能，不必重启进程。

class SkillFileBody(BaseModel):
    """保存技能文件请求体。"""

    path: str = "SKILL.md"
    text: str = ""


def _skill_dir(key: str) -> Path:
    """技能目录；key 只接受已存在的学科目录名（拒绝分隔符/..，防目录穿越）。"""
    from config.teacher_skills import md as skills_md

    k = str(key or "").strip()
    if not k or "/" in k or "\\" in k or ".." in k:
        raise HTTPException(status_code=400, detail="非法技能 key")
    d = skills_md.SKILLS_DIR / k
    if not d.is_dir():
        raise HTTPException(status_code=404, detail=f"技能不存在：{k}")
    return d


def _skill_file_path(key: str, rel: str) -> Path:
    """技能内文件路径：只允许 ``SKILL.md`` 或 ``references/<name>.md``。"""
    from config.teacher_skills import md as skills_md

    r = str(rel or skills_md.SKILL_FILENAME).strip().replace("\\", "/")
    if r == skills_md.SKILL_FILENAME:
        return _skill_dir(key) / skills_md.SKILL_FILENAME
    if re.fullmatch(r"references/[\w\-.]+\.md", r):
        return _skill_dir(key) / r
    raise HTTPException(status_code=400, detail="只允许 SKILL.md 与 references/*.md")


def _validate_skill_md(text: str) -> list[str]:
    """SKILL.md 保存前校验：返回问题清单（空 = 通过）。

    为什么拒收：这份 md 是运行期事实源，写进去立刻影响上课；半成品（缺必填字段 /
    缺九项行为小节）会被加载器判定「不完整」并静默回退默认人格 —— 与其线上才发现，
    不如保存时就拦住，并把缺什么列清楚。
    """
    from config.teacher_skills import md as skills_md

    try:
        front, sections = skills_md.parse_skill_md(text)
    except Exception as err:  # noqa: BLE001 - 解析失败原样返回原因
        return [f"解析失败：{err}"]
    problems = [
        f"frontmatter 缺少必填字段：{field}"
        for field in skills_md.REQUIRED_FRONT
        if not str(front.get(field) or "").strip()
    ]
    required = [*skills_md.STYLE_SECTIONS, skills_md.PROMPT_SECTION,
                skills_md.GREETING_SECTION, *skills_md.BEHAVIOR_SECTIONS]
    missing = [t for t in required if not str(sections.get(t) or "").strip()]
    if missing:
        problems.append("缺少必备小节：" + "、".join(missing))
    return problems


@router.get("/api/admin/skills")
async def admin_list_skills(user: dict = Depends(require_developer)) -> dict:
    """技能清单（开发人员）：各科技的 SKILL.md 状态与可编辑文件列表。"""
    from config.teacher_skills import md as skills_md

    items = []
    for key in skills_md.available_keys():
        doc = skills_md.load_doc(key)
        path = skills_md.skill_path(key)
        files = [skills_md.SKILL_FILENAME]
        refs = skills_md.SKILLS_DIR / key / "references"
        if refs.is_dir():
            files += [f"references/{p.name}" for p in sorted(refs.glob("*.md"))]
        items.append({
            "key": key,
            "skill_id": doc.skill_id if doc else "",
            "display_name": doc.display_name if doc else "",
            "persona": doc.persona if doc else "",
            "depth": doc.depth if doc else "",
            "speech_rate": doc.speech_rate if doc else None,
            "complete": bool(doc),          # False = 解析失败/必填缺失（加载器会跳过）
            "files": files,
            "updated_at": path.stat().st_mtime if path.is_file() else 0.0,
        })
    return {"ok": True, "skills": items, "operator": user.get("username") or ""}


@router.get("/api/admin/skills/{key}/file")
async def admin_read_skill_file(
    key: str, path: str = "SKILL.md", user: dict = Depends(require_developer),
) -> dict:
    """读取技能文件原文（开发人员）。"""
    fp = _skill_file_path(key, path)
    if not fp.is_file():
        raise HTTPException(status_code=404, detail=f"文件不存在：{path}")
    return {"ok": True, "key": key, "path": str(path), "text": fp.read_text(encoding="utf-8")}


@router.put("/api/admin/skills/{key}/file")
async def admin_write_skill_file(
    key: str, body: SkillFileBody, user: dict = Depends(require_developer),
) -> dict:
    """保存技能文件（开发人员）。

    * SKILL.md：先结构校验（frontmatter 必填 + 必备小节），不通过拒收并列出问题；
    * 写前同目录留一份 ``<文件名>.bak``（手滑保护，只保留最新一份；回滚走 git）；
    * 保存后清解析缓存 —— 热生效，不必重启。
    """
    from config.teacher_skills import md as skills_md

    rel = str(body.path or skills_md.SKILL_FILENAME).strip().replace("\\", "/")
    fp = _skill_file_path(key, rel)
    text = str(body.text or "")
    if rel == skills_md.SKILL_FILENAME:
        problems = _validate_skill_md(text)
        if problems:
            raise HTTPException(status_code=400, detail="；".join(problems))
    if fp.is_file():
        try:
            fp.with_name(fp.name + ".bak").write_text(fp.read_text(encoding="utf-8"), encoding="utf-8")
        except OSError:
            pass                                # 备份失败不阻断保存（git 仍是兜底）
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(text, encoding="utf-8")
    if rel == skills_md.SKILL_FILENAME:
        skills_md.reload_docs()                 # 清解析缓存：下一节课立即生效
    return {"ok": True, "key": key, "path": rel, "bytes": len(text.encode("utf-8"))}


def _skill_loop_config() -> dict:
    """自我更新的「触发条件」快照：前端只展示，不在页面里再抄一份阈值。"""
    return {
        "self_update_enable": settings.skill_self_update_enable,
        "reflect_max_items": settings.skill_reflect_max_items,
        "reflect_min_confidence": settings.skill_reflect_min_confidence,
        "reflect_min_interval_seconds": settings.skill_reflect_min_interval_seconds,
        "inject_limit": settings.skill_experience_inject_limit,
        "bandit_enable": settings.skill_experience_bandit_enable,
        "feedback_enable": settings.skill_experience_feedback_enable,
        "experience_max_items": settings.skill_experience_max_items,
        "experience_dir": settings.skill_experience_dir,
        "agent_skills_enable": settings.agent_skills_enable,
        "agent_skill_auto_retire": settings.agent_skill_auto_retire,
        "agent_skill_min_pulls": settings.agent_skill_min_pulls,
        "agent_skill_min_gain_rate": settings.agent_skill_min_gain_rate,
        "agent_skill_learned_max": settings.agent_skill_learned_max,
        "agent_skill_usage_dir": settings.agent_skill_usage_dir,
    }


@router.get("/api/admin/skills/loop")
async def admin_skill_loop(limit: int = 500, user: dict = Depends(require_developer)) -> dict:
    """Skill 自我更新看板：把 append-only JSONL 还原成更新事件时间线（开发人员）。

    **只读接口**：数据源是 ``data/teacher_experience/*.jsonl``（教师经验）与
    ``data/skill_usage/*.jsonl``（Agent 技能战绩），不写任何文件、不改变学习链路。
    事件类型（学习 / 命中强化 / 战绩回写 / 停用 / 恢复 / 记录损坏）由同一 ``id``
    相邻两次快照的字段差分推断，前后对比与结果三态（ok / warn / fail）也在这里算好，
    页面只负责画出来。重建规则与取舍见 ``core/skills/loop_timeline.py``。
    """
    from core.skills.loop_timeline import MAX_LIMIT, build_loop_timeline  # noqa: PLC0415 - 延迟导入

    data = await asyncio.to_thread(
        build_loop_timeline,
        settings.skill_experience_dir,
        settings.agent_skill_usage_dir,
        limit=max(1, min(int(limit), MAX_LIMIT)),
    )
    return {
        "ok": True,
        "operator": user.get("username") or "",
        "config": _skill_loop_config(),
        **data,
    }


# ---------------------------------------------------------------- 学情记忆（长期记忆）
def _memory_store():
    """进程级学情记忆表（延迟导入，记忆功能关闭时也不影响其它接口）。"""
    from core.memory import get_memory_store

    return get_memory_store()


@router.get("/api/memory")
async def list_memory_students() -> dict:
    """学情记忆全局概况（哪些学生有记忆、各有多少条）。"""
    store = _memory_store()
    return {"ok": True, **store.stats(), "students_list": store.list_students()}


@router.get("/api/memory/{student_id}")
async def list_student_memory(student_id: str, subject: str = "") -> dict:
    """查看某位学生的学情记忆（含已停用条目）。

    与教师经验接口对称：自动写入是默认行为，这个接口是给人**看一眼和止损**用的。
    """
    store = _memory_store()
    return {
        "ok": True,
        "student_id": student_id,
        "active": [it.to_dict() for it in store.active(student_id, subject=subject)],
        "all": [it.to_dict() for it in store.load(student_id)],
    }


class _MemoryPatch(BaseModel):
    student_id: str
    id: str
    status: str = "retired"      # retired 停用 / active 恢复


@router.post("/api/memory/status")
async def set_memory_status(payload: _MemoryPatch) -> dict:
    """停用 / 恢复一条学情记忆（记错时的止损与回滚入口）。"""
    store = _memory_store()
    if payload.status not in ("retired", "active"):
        return {"ok": False, "error": "status 只能是 retired / active"}
    changed = (
        store.restore(payload.student_id, payload.id)
        if payload.status == "active"
        else store.retire(payload.student_id, payload.id)
    )
    return {
        "ok": changed,
        "student_id": payload.student_id,
        "id": payload.id,
        "status": payload.status,
    }


@router.delete("/api/memory/{student_id}")
async def forget_student_memory(student_id: str) -> dict:
    """**物理删除**某位学生的全部学情记忆（家长要求删除 / 数据清理）。

    面向未成年人的合规要求：删除请求不能只是「标记停用」——标记后数据仍在盘上。
    """
    removed = _memory_store().forget(student_id)
    return {"ok": True, "student_id": student_id, "removed": removed}


@router.get("/api/avatar/source")
async def avatar_source() -> dict:
    """暴露当前生效的 AI 教师驱动视频源（含 video_hash）。

    供监控与“换源触发下游重跑”判定：`video_hash` 一变即代表源已更换，
    人脸缓存会随之失效重建。
    """
    return get_avatar_source_info().model_dump()


@router.get("/api/avatar/idle")
async def avatar_idle(subject: str = "", teacher: str = "") -> dict:
    """静音待机素材：与讲解视频同源，供前端替换待机画面，使切换无跳变。

    返回 ``{"video": url|None, "still": url|None, "blink": url|None}``：
    * ``still`` 为从静音视频抽出的静态帧（推荐，嘴完全不动）；
    * ``blink`` 为该老师真实闭眼帧合成的整帧图（待机眨眼用）；
    * ``video`` 为静音渲染视频（仍会有轻微嘴部/头部微动）。

    ``subject``/``teacher`` 指定学科（多老师）：按该学科的老师底片生成**专属**素材。
    首次调用会触发一次静音渲染（数秒），之后走缓存直接返回；引擎不可用时均为 null。
    """
    from core.avatar_idle import ensure_idle_assets  # 局部导入避免模块级循环依赖
    from config.teachers import teacher_for_subject, teacher_key_for_subject

    # 走 teacher_for_subject：它会校验「老师 key 是否属于该学科」，
    # 避免 ?subject=物理&teacher=math_strict 把数学老师的待机素材串过来。
    profile = teacher_for_subject(subject, teacher)
    key = profile.key if profile else teacher_key_for_subject(subject)
    return await ensure_idle_assets(key)


@router.get("/api/teachers")
async def list_teachers(subject: str = "") -> dict:
    """某学科的可选老师列表（同科多老师 / 风格选择）。

    学科选择页点开一张学科卡后调用：返回该学科下全部老师（默认老师排第一），
    前端据此弹出「选风格」浮层。**只配置了一位老师的学科返回长度为 1 的列表**，
    前端据此跳过选择、直接进课堂（保持旧行为）。
    """
    from config.teachers import default_teacher_key, teacher_variants
    from core.avatar_idle import ensure_teacher_portrait

    variants = teacher_variants(subject)
    # 学科级教师人格（同科各变体共享同一个人格，这里挂在学科上，也挂到每位老师上）
    personality = _personality_info(subject)
    # 每位老师的形象缩略图（抽底片待机帧，毫秒级；失败为 None，前端退化为 emoji）
    for item in variants:
        item["portrait_url"] = await ensure_teacher_portrait(item["key"])
        item["personality"] = personality
    return {
        "subject": subject,
        "default": default_teacher_key(subject),
        "multi": len(variants) > 1,
        "personality": personality,
        "teachers": variants,
    }


@router.get("/api/teacher")
async def teacher_info(subject: str = "", teacher: str = "") -> dict:
    """按学科返回该学科「老师」的形象信息（同科多老师：由 ``teacher`` 指定）。

    前端开课时调用，据此加载**该老师专属**的待机画面（静态帧 + 闭眼帧），
    从而保证「语文课只出现语文老师」「数学严谨型只出现那位老师」。
    ``teacher`` 为空（或该 key 不存在）时回退学科默认老师；
    学科没有独立档案时返回 ``key=""``，前端沿用旧的单老师形象。
    """
    from config.teachers import teacher_for_subject, teacher_key_for_subject
    from core.avatar_idle import ensure_idle_assets

    profile = teacher_for_subject(subject, teacher)
    key = profile.key if profile else teacher_key_for_subject(subject)
    idle = await ensure_idle_assets(key)
    return {
        "key": key,
        "subject": profile.subject if profile else "",
        "name": profile.name if profile else "",
        "display_name": profile.display_name() if profile else "",
        "variant_name": profile.variant_name if profile else "",
        # 性别：前端在服务端音频不可用、改用浏览器 SpeechSynthesis 朗读时，
        # 据此挑男声/女声（否则系统默认女声，男老师开口就是女声）。
        "gender": profile.gender if profile else "",
        # 该学科加载的教师人格技能（身份 / 人格 / 语气；未启用时为 None）
        "personality": _personality_info(profile.subject if profile else subject),
        "idle": idle,
        # 眼位（像素，基于底片坐标系）：前端做眨眼/口型遮罩时可参考
        "eye_points": [list(p) for p in profile.eye_points] if profile else [],
        "eye_r": list(profile.eye_r) if profile else [],
        # 底片真实分辨率：眼位坐标基于它，前端据此映射到显示尺寸。
        # 换形象后底片尺寸可能变化（如英语老师新肖像为 910x720），
        # 这里写死 1280x720 会让眨眼贴图错位。
        "video_size": _video_size(profile.talk_path if profile else ""),
    }


@lru_cache(maxsize=32)
def _video_size(talk_path: "Path | str") -> list[int]:
    """读底片分辨率（ffmpeg 探测）；失败时回退 1280x720。"""
    import re as _re
    import subprocess

    from core.audio_utils import _resolve_ffmpeg

    path = str(talk_path or "")
    if not path or not Path(path).is_file():
        return [1280, 720]
    ff = _resolve_ffmpeg()
    if not ff:
        return [1280, 720]
    try:
        out = subprocess.run([ff, "-i", path], capture_output=True, text=True, timeout=20)
        m = _re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", out.stderr or "")
        if m:
            return [int(m.group(1)), int(m.group(2))]
    except Exception:  # noqa: BLE001 - 探测失败不影响接口
        pass
    return [1280, 720]


@router.get("/api/idle/policy")
async def idle_policy() -> dict:
    """空闲主动问候策略：前端据此决定静默多久后发送 idle 事件。"""
    return {
        "enable": settings.idle_nudge_enable,
        "seconds": settings.idle_nudge_seconds,
        "max": settings.idle_nudge_max,
    }


# ---------------------------------------------------------------- 课程课时与计费
class CourseStartReq(BaseModel):
    student_id: str = "anonymous"
    subject: str = ""
    # 同科多老师：学生在选择界面挑中的老师档案 key（可选，留空=学科默认老师）
    teacher: str = ""


class CourseIdReq(BaseModel):
    course_id: str = ""


def _course_policy() -> dict:
    """课程策略（前端据此显示倒计时、提示时机与计费文案）。"""
    minutes = int(settings.course_duration_minutes)
    return {
        "duration_minutes": minutes,
        "duration_seconds": minutes * 60,
        "renew_prompt_seconds": int(settings.course_renew_prompt_seconds),
        "free_seconds": int(settings.course_free_seconds),
        "price": float(settings.course_price),
        "currency": settings.course_currency,
    }


@router.post("/api/course/start")
async def course_start(req: CourseStartReq) -> dict:
    """开一节课：返回课时信息。

    课时由服务端持有 —— 学生中途退出页面，倒计时**不会停**，到点自动结算关闭。
    同一学生可同时存在多门课（互不影响）。
    """
    from core.course_billing import get_course_manager

    course = get_course_manager().start(req.student_id, req.subject)
    return {"course": course.to_dict(), "policy": _course_policy()}


@router.get("/api/course/status")
async def course_status(student_id: str = "") -> dict:
    """进行中的课程列表（前端轮询：刷新倒计时、判断是否要弹续费提示）。"""
    from core.course_billing import get_course_manager

    return {"courses": get_course_manager().snapshot(student_id), "policy": _course_policy()}


@router.post("/api/course/renew")
async def course_renew(req: CourseIdReq) -> dict:
    """续费：在**剩余时间上再加一节课时长**。"""
    from core.course_billing import get_course_manager

    course = get_course_manager().renew(req.course_id)
    if course is None:
        return {"ok": False, "error": "课程不存在或已结束"}
    return {"ok": True, "course": course.to_dict()}


@router.post("/api/course/ack")
async def course_ack(req: CourseIdReq) -> dict:
    """学生点了「知道了」：本次不再重复弹窗。"""
    from core.course_billing import get_course_manager

    return {"ok": get_course_manager().ack_prompt(req.course_id)}


@router.post("/api/course/close")
async def course_close(req: CourseIdReq) -> dict:
    """结束课程并结算：已用时长 ≤ 免费时长不收费，超过则收整节课费用。"""
    from core.course_billing import get_course_manager

    result = get_course_manager().close(req.course_id)
    if result is None:
        return {"ok": False, "error": "课程不存在"}
    return {"ok": True, "course": result}


@router.get("/api/course/bills")
async def course_bills(student_id: str = "", limit: int = 50) -> dict:
    """账单记录（按学生）。"""
    from core.course_billing import get_course_manager

    return {
        "bills": get_course_manager().bills(student_id, limit),
        "currency": settings.course_currency,
    }
