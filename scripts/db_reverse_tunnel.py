"""数据库反向隧道：让**云端应用**能访问内网的 MySQL / Milvus。

背景：云端是公网机器，数据库在内网 ``10.224.10.174``，云端直连会 timeout
（实测：云端 -> 10.224.10.174:3306 与 :19530 均 TimeoutError）。
本脚本在**能访问数据库的机器**上运行，通过 SSH 把「云端端口」反向映射到数据库：

    云端 127.0.0.1:13306  ──SSH──►  内网 10.224.10.174:3306    (MySQL)
    云端 127.0.0.1:19531  ──SSH──►  内网 10.224.10.174:19530   (Milvus)

于是云端应用只需把地址配成 ``127.0.0.1:13306`` / ``127.0.0.1:19531`` 即可。

与 ``_tunnel.py``（正向：把云端服务映射到本机，供本机浏览器访问）方向相反，两者互不影响。

**凭据不写进代码**，全部从环境变量读取：

    DB_TUNNEL_SSH_HOST      SSH 主机（必填）
    DB_TUNNEL_SSH_PORT      SSH 端口（默认 22）
    DB_TUNNEL_SSH_USER      SSH 用户（默认 root）
    DB_TUNNEL_SSH_PASSWORD  SSH 密码（必填；也可用 DB_TUNNEL_SSH_KEY 指定私钥）
    DB_TUNNEL_TARGET_HOST   目标库主机（默认 10.224.10.174）
    DB_TUNNEL_MAP           映射表，格式 ``云端端口:目标端口``，逗号分隔
                            （默认 ``13306:3306,19531:19530``）

用法：

    python scripts/db_reverse_tunnel.py

设计要点（吸取 ``_tunnel.py`` 的教训）：
* 每个映射**各用一条独立 SSH 连接** —— 一条连接上多个远程转发时，
  paramiko 无法区分进来的连接属于哪个端口，分开最稳妥；
* ``set_keepalive`` 主动保活 + **断线自动重连（指数退避）**，
  不会像 ``_tunnel.py`` 那样「进程还在、通道已死、却静默不恢复」。
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import time

try:
    import paramiko
except ImportError:  # pragma: no cover
    print("缺少 paramiko：pip install paramiko")
    raise SystemExit(1)

SSH_HOST = os.environ.get("DB_TUNNEL_SSH_HOST", "").strip()
SSH_PORT = int(os.environ.get("DB_TUNNEL_SSH_PORT", "22") or 22)
SSH_USER = os.environ.get("DB_TUNNEL_SSH_USER", "root").strip()
SSH_PWD = os.environ.get("DB_TUNNEL_SSH_PASSWORD", "")
SSH_KEY = os.environ.get("DB_TUNNEL_SSH_KEY", "").strip()

TARGET_HOST = os.environ.get("DB_TUNNEL_TARGET_HOST", "10.224.10.174").strip()
MAP_SPEC = os.environ.get("DB_TUNNEL_MAP", "13306:3306,19531:19530").strip()

KEEPALIVE_SECONDS = 15
_RECONNECT_MAX = 30.0


def _log(message: str) -> None:
    print(f"[db-tunnel {time.strftime('%H:%M:%S')}] {message}", flush=True)


def _parse_map(spec: str) -> list[tuple[int, int]]:
    """``"13306:3306,19531:19530"`` -> ``[(13306, 3306), (19531, 19530)]``。"""
    pairs: list[tuple[int, int]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        cloud, _, target = item.partition(":")
        if not cloud.isdigit() or not target.isdigit():
            raise ValueError(f"映射格式错误：{item!r}（应为 云端端口:目标端口）")
        pairs.append((int(cloud), int(target)))
    if not pairs:
        raise ValueError("DB_TUNNEL_MAP 为空")
    return pairs


class Forwarder(threading.Thread):
    """单个端口的反向转发：云端 ``cloud_port`` -> 本侧 ``target_host:target_port``。

    独立线程 + 独立 SSH 连接：断线只影响自己，重连逻辑也无需区分端口。
    """

    def __init__(self, cloud_port: int, target_port: int) -> None:
        super().__init__(daemon=True, name=f"fwd-{cloud_port}")
        self.cloud_port = cloud_port
        self.target_port = target_port
        self._stop = threading.Event()

    # ---------------------------------------------------------------- 生命周期
    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        backoff = 2.0
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self._serve()
            except Exception as err:  # noqa: BLE001 - 任何异常都靠重连兜底
                _log(f":{self.cloud_port} 连接异常：{type(err).__name__}: {err}")
            if self._stop.is_set():
                break
            # 刚连上就断（多半是端口被占 / 认证问题）时不要疯狂重试
            if time.monotonic() - started < 3:
                backoff = min(backoff * 1.6, _RECONNECT_MAX)
            else:
                backoff = 2.0  # 稳定运行过一段时间：重置退避
            _log(f":{self.cloud_port} {backoff:.0f}s 后重连…")
            self._stop.wait(backoff)

    def _serve(self) -> None:
        """建立一条 SSH 连接并持续转发，直到连接失效或被要求停止。"""
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        connect_kwargs: dict[str, object] = {
            "hostname": SSH_HOST,
            "port": SSH_PORT,
            "username": SSH_USER,
            "timeout": 20,
            "banner_timeout": 20,
            "auth_timeout": 20,
        }
        if SSH_KEY:
            connect_kwargs["key_filename"] = SSH_KEY
        else:
            connect_kwargs["password"] = SSH_PWD
        client.connect(**connect_kwargs)  # type: ignore[arg-type]

        try:
            transport = client.get_transport()
            if transport is None:
                raise RuntimeError("SSH transport 不可用")
            transport.set_keepalive(KEEPALIVE_SECONDS)
            # 让云端 sshd 在 127.0.0.1:cloud_port 上监听，并把它收到的连接回灌到这边
            transport.request_port_forward("127.0.0.1", self.cloud_port)
            _log(f"就绪：云端 127.0.0.1:{self.cloud_port} -> {TARGET_HOST}:{self.target_port}")

            while not self._stop.is_set() and transport.is_active():
                chan = transport.accept(timeout=1.0)
                if chan is None:
                    continue
                threading.Thread(
                    target=self._handle, args=(chan,), daemon=True
                ).start()
            if not transport.is_active():
                _log(f":{self.cloud_port} SSH 通道已失效")
        finally:
            try:
                client.close()          # 关闭会一并释放云端的监听端口，重连才不会撞端口
            except Exception:            # noqa: BLE001
                pass

    # ---------------------------------------------------------------- 单连接转发
    def _handle(self, chan: "paramiko.Channel") -> None:
        """把一条进站连接接到目标库上，双向对拷。"""
        upstream: socket.socket | None = None
        try:
            upstream = socket.create_connection(
                (TARGET_HOST, self.target_port), timeout=10
            )
        except Exception as err:  # noqa: BLE001 - 目标不可达：直接关掉这条连接
            _log(f":{self.cloud_port} 无法连接目标库 {TARGET_HOST}:{self.target_port}：{err}")
            chan.close()
            return

        def pump(src, dst, close_dst) -> None:
            try:
                while True:
                    data = src.recv(16384)
                    if not data:
                        break
                    dst.sendall(data)
            except Exception:  # noqa: BLE001 - 对端关闭/超时都属正常
                pass
            finally:
                try:
                    close_dst()
                except Exception:  # noqa: BLE001
                    pass

        t = threading.Thread(target=pump, args=(chan, upstream, upstream.close), daemon=True)
        t.start()
        pump(upstream, chan, chan.close)
        t.join(timeout=5)


def main() -> int:
    if not SSH_HOST:
        print("缺少环境变量 DB_TUNNEL_SSH_HOST")
        return 2
    if not SSH_PWD and not SSH_KEY:
        print("缺少环境变量 DB_TUNNEL_SSH_PASSWORD（或 DB_TUNNEL_SSH_KEY）")
        return 2
    try:
        mappings = _parse_map(MAP_SPEC)
    except ValueError as err:
        print(f"映射解析失败：{err}")
        return 2

    _log(f"SSH {SSH_USER}@{SSH_HOST}:{SSH_PORT}  ->  目标库主机 {TARGET_HOST}")
    forwarders = [Forwarder(cloud, target) for cloud, target in mappings]
    for fwd in forwarders:
        fwd.start()
    _log("全部映射已启动，Ctrl+C 停止")
    try:
        while True:
            time.sleep(1)
            if not any(fwd.is_alive() for fwd in forwarders):
                _log("所有转发线程均已退出")
                return 1
    except KeyboardInterrupt:
        _log("正在停止…")
    finally:
        for fwd in forwarders:
            fwd.stop()
        for fwd in forwarders:
            fwd.join(timeout=3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
