"""HTTP/信令服务（aiohttp）：WebRTC offer、播报控制、状态查询。

接口：
    POST /offer          WebRTC 信令（{"sdp": ..., "type": "offer"} → 返回应答与 sessionid）
    POST /api/echo       播报文本（{"text": ..., 可选 voice/expression/avatar}；
                         expression=表情素材目录名；avatar=要切换到的激活素材目录名）
    POST /api/avatar     切换激活素材（{"avatar_id": "ai4"}）：待机动画与默认播报同源，
                         由教育端按学生所选科目下发（数学 → ai4）
    GET  /api/status     播报状态（{"speaking": bool, "playing": bool, "avatar": str}）
                         speaking=有播报任务（含 TTS/推理/预滚准备阶段）；
                         playing =口型已就绪、音频真正开播（显示端据此切换实时画面）
    POST /api/interrupt  打断当前播报
    GET  /               服务信息
"""
import asyncio
import logging
import os
import time
import uuid

from aiohttp import web
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)

from server.webrtc import AudioTrack, VideoTrack

logger = logging.getLogger("digital_human.http")


def _build_rtc_configuration(cfg):
    """WebRTC ICE 配置：AutoDL 等无公网 IP 的算力云，媒体流必须走 TURN 中继，
    否则 aiortc 只收集 host 候选、浏览器连不上。
    优先读环境变量 RTC_STUN_URL / RTC_TURN_URL / RTC_TURN_USERNAME / RTC_TURN_CREDENTIAL，
    其次读 config.yaml 的 stun_url / turn_url / turn_username / turn_credential；
    都为空则返回 None（仅 host 候选，本机/局域网演示用）。
    """
    c = cfg or {}
    stun = os.environ.get("RTC_STUN_URL") or c.get("stun_url") or ""
    turn = os.environ.get("RTC_TURN_URL") or c.get("turn_url") or ""
    servers = []
    if stun:
        servers.append(RTCIceServer(urls=[stun]))
    if turn:
        servers.append(RTCIceServer(
            urls=[u.strip() for u in turn.split(",") if u.strip()],
            username=os.environ.get("RTC_TURN_USERNAME") or c.get("turn_username") or "",
            credential=os.environ.get("RTC_TURN_CREDENTIAL") or c.get("turn_credential") or "",
        ))
    if not servers:
        return None
    return RTCConfiguration(iceServers=servers)


def create_app(engine, cfg):
    app = web.Application(middlewares=[_cors])
    # 单活跃会话：engine 是全局单播报流（_latest_frame/_play_frames/_audio_pcm 唯一），
    # 多个 pc 同时连会各自读到同一份状态 → 第二个客户端静默串音串画。故同一时刻
    # 只保留一个活跃会话，新 /offer 顶替旧会话（关旧 pc），把“一卡一路”隐含约束变成显式保护。
    session = {"pc": None, "sid": None}
    session_lock = asyncio.Lock()

    async def index(request):
        return web.json_response({"service": "digital-human", "avatar": cfg["avatar_id"]})

    async def offer(request):
        try:
            params = await request.json()
        except Exception:
            return web.json_response({"error": "invalid json"}, status=400)
        sdp = params.get("sdp")
        if not sdp:
            return web.json_response({"error": "missing sdp"}, status=400)

        pc = RTCPeerConnection(configuration=_build_rtc_configuration(cfg))
        sid = uuid.uuid4().hex[:12]

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info("会话 %s 状态：%s", sid, pc.connectionState)
            if pc.connectionState in ("failed", "closed", "disconnected"):
                # 仅当自己仍是活跃会话才清空占位，避免误清后来的顶替者
                async with session_lock:
                    if session["pc"] is pc:
                        session["pc"], session["sid"] = None, None
                await pc.close()

        pc.addTrack(VideoTrack(engine))
        pc.addTrack(AudioTrack(engine))

        # 顶替旧会话：锁内原子换占位，锁外关旧 pc。
        # close 会触发旧 pc 的 connectionstatechange 回调、回调又要抢同一把锁，
        # 故 close 必须放锁外，避免“持锁等待被自己触发的回调”式互等。
        async with session_lock:
            old_pc, old_sid = session["pc"], session["sid"]
            session["pc"], session["sid"] = pc, sid
        if old_pc is not None:
            logger.info("顶替旧会话 %s（新会话 %s 接入）", old_sid, sid)
            try:
                await old_pc.close()
            except Exception:  # noqa: BLE001 - 关闭失败不阻断新会话
                logger.exception("关闭旧会话 %s 失败", old_sid)

        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=params.get("type", "offer")))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        # 等待 ICE 收集完成（非 trickle 模式，应答 SDP 需包含候选）
        deadline = time.time() + 5.0
        while pc.iceGatheringState != "complete" and time.time() < deadline:
            await asyncio.sleep(0.05)

        logger.info("WebRTC 会话建立：%s", sid)
        return web.json_response({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "sessionid": sid,
        })

    async def echo(request):
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid json"}, status=400)
        text = (data.get("text") or "").strip()
        if not text:
            return web.json_response({"ok": False, "error": "empty text"}, status=400)
        engine.echo(text, voice=data.get("voice"), expression=data.get("expression"),
                    avatar=data.get("avatar"))
        return web.json_response({"ok": True})

    async def switch_avatar(request):
        """切换激活素材：待机帧与后续播报一起换成该素材（教育端按科目下发）。"""
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid json"}, status=400)
        avatar_id = (data.get("avatar_id") or "").strip()
        if not avatar_id:
            return web.json_response({"ok": False, "error": "empty avatar_id"}, status=400)
        engine.set_avatar(avatar_id)
        logger.info("切换激活素材请求：%s", avatar_id)
        return web.json_response({"ok": True, "avatar": avatar_id})

    async def status(request):
        return web.json_response({
            "speaking": engine.is_speaking(),
            "playing": engine.is_playing(),
            "avatar": engine.active_avatar(),
        })

    async def interrupt(request):
        engine.interrupt()
        return web.json_response({"ok": True})

    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.router.add_post("/api/echo", echo)
    app.router.add_post("/api/avatar", switch_avatar)
    app.router.add_get("/api/status", status)
    app.router.add_post("/api/interrupt", interrupt)

    async def on_shutdown(_app):
        async with session_lock:
            pc = session["pc"]
            session["pc"], session["sid"] = None, None
        if pc is not None:
            await pc.close()

    app.on_shutdown.append(on_shutdown)
    return app


@web.middleware
async def _cors(request, handler):
    """CORS：允许学生端页面跨域访问本服务。"""
    if request.method == "OPTIONS":
        resp = web.Response(status=204)
    else:
        resp = await handler(request)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp
