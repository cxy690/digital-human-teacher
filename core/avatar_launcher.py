"""数字人服务托管：探活 + 自动拉起（detached 子进程）。

教育服务启动时确保 digital_human 服务在运行（双环境托管，见项目架构决策）：
- 探活：GET ``{avatar_api_base}/api/status`` 返回 200 视为就绪；
- 未运行且允许自启：用数字人专用 conda 环境（CUDA torch）detached 拉起
  ``python run.py --host 127.0.0.1 --port <base 端口> --avatar <id>``；
- 幂等：.launcher.state 记录 PID 与拉起时间；模型加载窗口内不重复拉起，
  避免 uvicorn --reload 触发 lifespan 多次执行时起出多个实例；
- 一切失败只记日志不抛出：前端连不上数字人时自动回退 SVG 机器人。
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

from config.logging import get_logger
from config.settings import settings

log = get_logger("core.avatar_launcher")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIGITAL_HUMAN_DIR = PROJECT_ROOT / "digital_human"
STATE_FILE = DIGITAL_HUMAN_DIR / ".launcher.state"
SERVICE_LOG = DIGITAL_HUMAN_DIR / "avatar_service.log"

LOAD_WINDOW_SECONDS = 240.0  # 模型 + 素材加载宽限（就绪前不重复拉起）
PROBE_TIMEOUT = 1.0

# 数字人专用解释器候选（低于 settings.avatar_python 优先级）
_PYTHON_CANDIDATES = (
    r"D:\anaconda3\envs\livetalking\python.exe",
)


def probe() -> bool:
    """探活数字人服务（就绪判定：/api/status 200）。"""
    try:
        resp = httpx.get(f"{settings.avatar_api_base}/api/status", timeout=PROBE_TIMEOUT)
        return resp.status_code == 200
    except Exception:  # noqa: BLE001 - 未启动/网络不通均视为未就绪
        return False


def ensure_avatar_running() -> bool:
    """幂等确保数字人服务在运行；返回探活结果（False 不等于失败，可能正在加载）。"""
    if not settings.avatar_enable:
        return False
    if probe():
        return True
    if not settings.avatar_autostart:
        return False
    if os.name != "nt":
        # 云端/Linux：数字人服务由外部（compose / systemd / start_cloud.sh）独立托管，
        # 不走这里的 Windows 专用拉起逻辑（_pid_alive 依赖 ctypes.windll，Linux 上不可用）。
        # 探活已在上面完成：连不上只说明服务未就绪，前端自动降级 SVG 机器人。
        log.info("non-Windows platform; avatar autostart skipped (manage digital_human externally)")
        return False

    # 加载窗口内且进程仍活着 → 正在加载，不重复拉起
    state = _read_state()
    if state and _pid_alive(int(state.get("pid") or 0)) and _within_window(state):
        log.info("digital avatar loading (pid={}), spawn skipped", state.get("pid"))
        return False

    python = _resolve_python()
    if python is None:
        log.warning("digital avatar python not found, autostart skipped")
        return False

    port = urlparse(settings.avatar_api_base).port or 8010
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"  # 子进程日志 UTF-8，避免 GBK 乱码
    try:
        log_handle = open(SERVICE_LOG, "ab", buffering=0)
        proc = subprocess.Popen(
            [python, "run.py", "--host", "127.0.0.1", "--port", str(port),
             "--avatar", settings.avatar_id],
            cwd=str(DIGITAL_HUMAN_DIR),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        _write_state({"pid": proc.pid, "ts": time.time()})
        log.info("digital avatar spawned (pid={}); log: {}", proc.pid, SERVICE_LOG)
    except Exception as err:  # noqa: BLE001 - 托管失败不影响教育服务
        log.warning("digital avatar spawn failed: {}", err)
        return False
    return False


# ---------------------------------------------------------------- 内部


def _resolve_python() -> str | None:
    if settings.avatar_python and Path(settings.avatar_python).is_file():
        return settings.avatar_python
    for cand in _PYTHON_CANDIDATES:
        if Path(cand).is_file():
            return cand
    return None


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


def _read_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - 损坏即视为无状态
        return {}


def _write_state(payload: dict) -> None:
    try:
        STATE_FILE.write_text(json.dumps(payload), encoding="utf-8")
    except Exception as err:  # noqa: BLE001
        log.warning("launcher state write failed: {}", err)


def _within_window(state: dict) -> bool:
    ts = float(state.get("ts") or 0)
    return 0 < time.time() - ts < LOAD_WINDOW_SECONDS
