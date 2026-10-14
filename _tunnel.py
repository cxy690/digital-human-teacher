"""临时：本地 -> 算力云的 SSH 端口转发（把云端引擎映射到本地同名端口）。

转发端口：
    8012  MuseTalk 数字人渲染引擎
    8090  CosyVoice 克隆 TTS 引擎（本地要用老师本人音色，必须映射）

断线自愈（为什么需要）：AutoDL 的 SSH 网关会**静默掐掉长连接**。旧版隧道的现象是
进程还活着、本地端口也还在监听，但 SSH transport 早就失效——转发请求被无声丢弃，
本地主应用调引擎/TTS 全部超时且无任何报错，表现就是「数字人不动、也不出声」。
现在守护线程每 10 秒主动探测一次（往云端真开一条 TCP channel，只查
``is_active()`` 发现不了半开连接），失效立即整体重连；转发 handler 每次
动态取当前连接——不用再手动重启本脚本。

凭据全部从环境变量读取（与 scripts/db_reverse_tunnel.py 同风格）。
AutoDL 实例重启/迁移后 SSH 入口会变，届时只需 setx 更新变量，无需改代码：
    TUNNEL_SSH_HOST      SSH 主机（必填，如 <your-cloud-host>）
    TUNNEL_SSH_PORT      SSH 端口（默认 22）
    TUNNEL_SSH_USER      SSH 用户（默认 root）
    TUNNEL_SSH_PASSWORD  SSH 密码（必填）

用法：
    setx TUNNEL_SSH_PASSWORD "你的密码"   # 一次性写入用户环境变量
    python _tunnel.py                     # 前台运行，Ctrl+C 停止（需新开终端）
"""
from __future__ import annotations

import os
import socketserver
import threading
import time

import paramiko

HOST = os.environ.get("TUNNEL_SSH_HOST", "").strip()
PORT = int(os.environ.get("TUNNEL_SSH_PORT", "22") or 22)
USER = os.environ.get("TUNNEL_SSH_USER", "root").strip()
PWD = os.environ.get("TUNNEL_SSH_PASSWORD", "").strip()

# (本地端口, 云端目标端口)
# 8001 = 云端主应用：转发它，本机浏览器直接开云端页面（渲染/音色在云端内部调用，无需转发）
# 8012 / 8090 = 云端引擎：只有「本地跑主应用、借云端 GPU」时才需要（见下方 TUNNEL_MODE）
# 模式：TUNNEL_MODE=engine（默认）转发 8012/8090；TUNNEL_MODE=app 转发 8001。
# 注意：本机若已跑主应用，转发 8001 会端口冲突，此时用 engine 模式。
TUNNEL_MODE = os.environ.get("TUNNEL_MODE", "engine").strip().lower()
if TUNNEL_MODE == "app":
    FORWARDS = [(8001, 8001)]
else:
    FORWARDS = [(8012, 8012), (8090, 8090)]

PROBE_PORT = 8001            # 主动探测目标：云端主应用常驻端口
PROBE_INTERVAL = 10.0        # 探测周期（秒）

#: 当前 SSH 连接（全局唯一；守护线程发现失效时整体替换）
_CLIENT: paramiko.SSHClient | None = None
_CLIENT_LOCK = threading.Lock()


def _connect() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30,
                   banner_timeout=30, auth_timeout=30)
    return client


def _active_transport() -> paramiko.Transport | None:
    """取当前活跃的 SSH transport；没有 / 已失效时返回 ``None``。"""
    with _CLIENT_LOCK:
        client = _CLIENT
        if client is None:
            return None
        transport = client.get_transport()
        if transport is not None and transport.is_active():
            return transport
    return None


def _reconnect() -> None:
    """整体重建 SSH 连接（旧连接直接丢弃，重连成功后新请求自动指到新连接）。"""
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is not None:
            try:
                _CLIENT.close()
            except Exception:  # noqa: BLE001
                pass
            _CLIENT = None
        _CLIENT = _connect()


def _probe(transport: paramiko.Transport) -> bool:
    """主动探测链路：往云端开一条到主应用端口的 channel，能开即视为在线。

    只查 ``is_active()`` 不够——网关掐断时本端可能仍以为连接活着（半开），
    必须真开一条 channel 才能发现。
    """
    try:
        chan = transport.open_channel(
            "direct-tcpip", ("127.0.0.1", PROBE_PORT), ("127.0.0.1", 0))
        chan.close()
        return True
    except Exception:  # noqa: BLE001
        return False


def _supervise() -> None:
    """守护线程：周期探测，链路失效立即重连（云端不可达时持续重试）。"""
    while True:
        transport = _active_transport()
        if transport is None or not _probe(transport):
            print("[tunnel] 链路失效，正在重连…", flush=True)
            try:
                _reconnect()
                print("[tunnel] 重连成功", flush=True)
            except Exception as err:  # noqa: BLE001
                print(f"[tunnel] 重连失败（{type(err).__name__}: {err}），"
                      f"{PROBE_INTERVAL:.0f}s 后重试", flush=True)
        time.sleep(PROBE_INTERVAL)


class ForwardHandler(socketserver.BaseRequestHandler):
    remote: tuple[str, int] = ("127.0.0.1", 8012)

    def handle(self) -> None:
        # 每次都取当前连接：SSH 断线重连后自动指向新 transport，无需重启本脚本
        transport = _active_transport()
        if transport is None:
            print("[tunnel] SSH 未连接，丢弃一条本地请求（守护线程正在重连）", flush=True)
            return
        try:
            chan = transport.open_channel(
                "direct-tcpip", self.remote, self.request.getpeername())
        except Exception as exc:                       # noqa: BLE001
            print(f"[tunnel] open_channel failed: {exc}", flush=True)
            return

        def pump(src, dst):
            try:
                while True:
                    data = src.recv(16384)
                    if not data:
                        break
                    dst.sendall(data)
            except Exception:
                pass
            finally:
                try:
                    dst.shutdown(2)
                except Exception:
                    pass

        t = threading.Thread(target=pump, args=(self.request, chan), daemon=True)
        t.start()
        pump(chan, self.request)
        t.join(timeout=5)
        chan.close()


class ForwardServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    if not HOST:
        print("缺少环境变量 TUNNEL_SSH_HOST：setx TUNNEL_SSH_HOST \"云主机地址\" 后重开终端再运行")
        raise SystemExit(2)
    if not PWD:
        print("缺少环境变量 TUNNEL_SSH_PASSWORD：setx TUNNEL_SSH_PASSWORD \"密码\" 后重开终端再运行")
        raise SystemExit(2)

    try:
        _reconnect()
        print(f"[tunnel] 已连接 {USER}@{HOST}:{PORT}", flush=True)
    except Exception as err:  # noqa: BLE001 - 首次连不上也起服务，守护线程会持续重试
        print(f"[tunnel] 首次连接失败（{type(err).__name__}: {err}），"
              f"服务照常启动，守护线程会自动重试", flush=True)

    for local_port, remote_port in FORWARDS:
        handler = type("H", (ForwardHandler,), {"remote": ("127.0.0.1", remote_port)})
        server = ForwardServer(("127.0.0.1", local_port), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"[tunnel] 127.0.0.1:{local_port} -> 云端 127.0.0.1:{remote_port}  ready", flush=True)

    threading.Thread(target=_supervise, daemon=True).start()
    print("[tunnel] 全部就绪（断线自动重连），Ctrl+C 停止", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        with _CLIENT_LOCK:
            if _CLIENT is not None:
                try:
                    _CLIENT.close()
                except Exception:  # noqa: BLE001
                    pass


if __name__ == "__main__":
    main()
