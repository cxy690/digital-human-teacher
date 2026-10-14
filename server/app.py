"""FastAPI 应用入口：挂载 REST 路由、WebSocket 端点与学生端静态页面。"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# 项目根目录（education/）加入模块搜索路径，并固定为工作目录。
# 这样 `python server/app.py`、`python -m server.app`、PyCharm 直接运行三种方式都能跑，
# 也能正确读取 .env 与 data/，避免出现 RAG 0 条、AI 回退 Mock、配置丢失的问题。
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config.logging import get_logger, setup_logging
from config.settings import settings
from core.avatar_source import resolve_teacher_video
import asyncio
from contextlib import suppress

from a2a.routes import router as a2a_router
from mcp_tools.server import http_router as mcp_http_router
from server.routes import router
from server.websocket import websocket_endpoint

setup_logging()
log = get_logger("server.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时校验 AI 教师驱动视频源。

    校验不通过时自动回退到 ``AI_TEACHER_FALLBACK_PATH``（默认同主底片），
    并打印 WARNING；任何异常都不阻断启动（保持开箱即跑）。
    """
    try:
        path, used_fallback = resolve_teacher_video()
        if used_fallback:
            log.warning("启动完成：AI 教师数字人使用【回退旧素材】{}", path)
        else:
            log.info("启动完成：AI 教师数字人使用【新素材】{}", path)
    except Exception as err:  # noqa: BLE001 - 校验异常不影响服务启动
        log.warning("AI 教师驱动视频校验异常（忽略，不影响启动）：{}", err)

    # 身份体系：建库建表（MySQL）+ 预置演示账号。连不上数据库只记 WARNING，不阻断启动
    # （登录/注册接口会各自返回 5xx/错误，其余教学链路完全不受影响）。
    if settings.auth_enable:
        try:
            from core.auth.service import seed_demo_users
            from db.base import init_db

            await asyncio.to_thread(init_db)
            if settings.auth_seed_demo:
                created = await asyncio.to_thread(seed_demo_users)
                if created:
                    log.info("已预置演示账号（密码 123456）：{}", "、".join(created))
            log.info(
                "身份体系就绪：auth_required={}，令牌有效期 {} 分钟",
                settings.auth_required,
                settings.auth_token_ttl_minutes,
            )
        except Exception as err:  # noqa: BLE001 - 数据库异常不应阻断教育服务
            log.warning("身份体系初始化失败（登录/注册将不可用）：{}", err)

    # 数字人子服务（可选，合并自 education_fusion）：AVATAR_ENABLE=true 时先探活，
    # 未运行且 AVATAR_AUTOSTART=true 则 detached 拉起 digital_human/（:8010，WebRTC 推流）。
    # 默认关闭（settings.avatar_enable=False），探活与拉起都放到线程里，失败只记日志。
    if getattr(settings, "avatar_enable", False):
        try:
            from core.avatar_launcher import ensure_avatar_running

            ready = await asyncio.to_thread(ensure_avatar_running)
            log.info("数字人子服务：{}", "就绪" if ready else "未就绪（前端将自动降级）")
        except Exception as err:  # noqa: BLE001 - 托管失败不影响教育服务
            log.warning("数字人子服务托管异常（忽略）：{}", err)

    # 课件仓库预热（后台线程，不阻塞启动）。
    # 首次扫描要解析全部 pptx（实测 228 份课件约 100 秒）。原来这 100 秒是由
    # 「第一个打开课件菜单的学生」承担的，而且是在**事件循环里同步跑**——
    # 会把同一台机器上所有人的会话（含数字人视频流）一起冻住。
    warm_task = asyncio.create_task(_warm_courseware())

    # 课程巡检：学生退出页面后课时仍在跑，靠这里「到点自动关闭并结算」
    course_task = asyncio.create_task(_course_watchdog())
    try:
        yield
    finally:
        warm_task.cancel()
        course_task.cancel()
        with suppress(asyncio.CancelledError):
            await warm_task
        with suppress(asyncio.CancelledError):
            await course_task


async def _warm_courseware() -> None:
    """后台预热课件**索引**（不是全部解析）。

    课件解析最贵的是 pptx + 每页页图吸附，全量 228 份要一两分钟；以前启动就
    跑到这里全量解析，重启后头一两分钟谁点「课件」都在转圈。现在只建轻量索引
    （读各目录的 deck.json，几百毫秒），具体学科留到第一次被打开时按需解析。
    预热失败只记日志，不能影响服务启动。
    """
    try:
        from core.courseware.loader import get_store

        count = await asyncio.to_thread(get_store().prewarm)
        log.info("课件索引预热完成：{} 份（按学科懒解析，不再全量扫 pptx）", count)
    except Exception as err:  # noqa: BLE001 - 预热失败不影响服务
        log.warning("课件索引预热失败（不影响启动）：{}", err)


async def _course_watchdog(interval: float = 10.0) -> None:
    """周期性巡检课程课时：到时间自动结算关闭（见 core/course_billing.py）。"""
    from core.course_billing import get_course_manager

    while True:
        try:
            for item in get_course_manager().tick():
                log.info("课程到点自动关闭：{}（用时 {}s，{}）",
                         item["course_id"], item["elapsed_seconds"],
                         "免费" if item.get("charge", 0) <= 0 else f"收费 {item['charge']}")
        except Exception as err:  # noqa: BLE001 - 单次巡检异常不应终止任务
            log.warning("课程巡检异常：{}", err)
        await asyncio.sleep(interval)


app = FastAPI(title="AI 教师 1 对 1 实时答疑系统", version="0.1.0", lifespan=lifespan)

_cors_origins = [
    origin.strip() for origin in settings.cors_allow_origins.split(",") if origin.strip()
] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
# A2A 协议端点（Agent Card 发现 + JSON-RPC）：供外部 agent 调用答疑能力。
# 必须在这里注册 —— 下面的 app.mount("/") 静态兜底会吃掉未显式注册的路径。
app.include_router(a2a_router)
# MCP 协议端点（Streamable HTTP，单端点 POST /mcp）：把工具集暴露给标准 MCP 客户端。
# 与 A2A 同样必须在 app.mount("/") 静态兜底之前注册。
app.include_router(mcp_http_router())
app.add_api_websocket_route(settings.websocket_path, websocket_endpoint)

# 挂载学生端 / 教师端静态页面
_clients_dir = Path(__file__).resolve().parent.parent / "clients"
# 入口流程：学科选择页 -> 1v1 辅导页（student_tablet）
_subject_page = _clients_dir / "subject_select" / "index.html"
_tutor_page = _clients_dir / "student_tablet" / "index.html"
_entry_page = _subject_page if _subject_page.exists() else _tutor_page


class NoCacheStaticFiles(StaticFiles):
    """禁用浏览器缓存的静态文件服务。

    前端 JS 内联在页面里，页面被缓存会导致「改了代码却还是旧版本」的困惑
    （页面 no-store 已覆盖根路径入口，这里补齐 /app 静态挂载）。
    """

    async def get_response(self, path: str, scope):  # type: ignore[override]
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        return response


if _clients_dir.exists():
    app.mount("/app", NoCacheStaticFiles(directory=str(_clients_dir), html=True), name="clients")

# 注意：旧的 /media/ai_teacher 静态挂载已移除。
# 教师形象一律来自 data/teachers/<学科key>/，由 /api/teacher 按学科下发（见 server/routes.py）。

# 实时口型视频：后端落盘后由前端按 URL 流式加载，避免 1~3MB base64 经 WebSocket 传输造成卡顿
_avatar_stream_dir = _clients_dir.parent / "data" / "avatar_stream"
try:
    _avatar_stream_dir.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/media/avatar_stream",
        StaticFiles(directory=str(_avatar_stream_dir)),
        name="avatar_stream_media",
    )
except OSError:
    pass

# 课件页图片：PPT 导出的页面图按课件分目录存放，供白板作为背景加载
_courseware_dir = _clients_dir.parent / "data" / "courseware"
try:
    _courseware_dir.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/media/courseware",
        StaticFiles(directory=str(_courseware_dir)),
        name="courseware_media",
    )
except OSError:
    pass

if _entry_page.exists():

    # 前端 JS 内联在页面里，改完必须刷新才能生效；no-store 避免浏览器用启发式缓存返回旧页面
    _NO_STORE = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    }

    @app.get("/", include_in_schema=False)
    async def _root() -> FileResponse:
        """根路径进入学科选择页（选择学科后跳转 1v1 辅导页）。"""
        return FileResponse(str(_entry_page), headers=_NO_STORE)

    @app.get("/student_tablet/index.html", include_in_schema=False)
    async def _tutor() -> FileResponse:
        """1v1 辅导页也禁用缓存，避免「改了页面却还是旧版本」的困惑。"""
        return FileResponse(str(_tutor_page), headers=_NO_STORE)

    # 兜底：客户端静态资源也能从根路径直接访问（如 /teacher_web/index.html）。
    # 此挂载最后注册，/health、/api、/ws 以及上面两个显式路由优先匹配。
    # 必须用 NoCacheStaticFiles：否则直接访问 /student_tablet/（不带 index.html）时走的是
    # 默认静态服务，浏览器会用「启发式缓存」返回旧页面，表现为「改了代码但前端还是旧版」。
    app.mount("/", NoCacheStaticFiles(directory=str(_clients_dir), html=True), name="root")


def main() -> None:
    """命令行入口：``education-server``。"""
    import uvicorn

    # 启动后打印一个可直接点击的本地访问地址（0.0.0.0 是监听地址，浏览器打不开）
    print("=" * 66, flush=True)
    print(f"  AI 教师学生端已启动，请点击打开： http://127.0.0.1:{settings.port}", flush=True)
    print("=" * 66, flush=True)

    uvicorn.run(
        "server.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
