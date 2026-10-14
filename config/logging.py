"""统一日志配置，基于 loguru；提供结构化上下文（会话/学生/模块）并统一落盘。

**两个 sink**：

* **stderr**（带颜色）—— 开发时在终端直接看；
* **文件**（``logs/app.log``，无颜色、自动轮转与清理）—— 排查时翻历史。

**保留策略是三个条件同时生效**（任一条超限就从最旧的开始删，且永远至少留一份）：

1. ``LOG_KEEP``（默认 ``30 days``）—— 多久以前的日志删掉；
2. ``LOG_MAX_FILES``（默认 ``90``）—— 最多留几个文件；
3. ``LOG_MAX_TOTAL_MB``（默认 ``200``）—— 日志总量上限（MB）。

为什么不用 loguru 现成的 ``retention="30 days"``：它一次只认一个条件（0.7.3 实测传列表
会报 ``Cannot infer retention ... 'list'``），而我们既要按时间留、又要防日志被刷爆。
所以这里传一个 callable 自己裁（见 :func:`_make_retention`）。

为什么要有文件 sink：以前只写 stderr，本地跑完关掉终端就没了；云端之所以有文件，
是靠启动脚本重定向 stdout。现在统一由 loguru 直接写 ``logs/app.log``，本地/云端行为一致；
启动脚本那份进程级重定向保留为 ``logs/app.stdout.log``（含 uvicorn 访问日志与崩溃栈）。

目录由 ``settings.log_dir`` 决定（默认 ``logs/``，相对项目根），已被 ``.gitignore`` 排除。
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from config.settings import settings

_CONFIGURED = False

#: 终端格式（带颜色，含「模块:函数:行号」便于直接定位）
_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    # 带出「哪个函数、哪一行」——排查时不必再翻代码定位日志出处
    "<cyan>{extra[module]}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

#: 文件格式（无颜色），与终端信息量一致
_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "{extra[module]}:{function}:{line} | {message}"
)


def log_dir() -> Path:
    """日志目录（不存在则创建）；绝对路径原样使用，相对路径按项目根解析。"""
    root = Path(__file__).resolve().parent.parent
    path = Path(settings.log_dir or "logs")
    path = path if path.is_absolute() else root / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_file() -> Path:
    """主应用日志文件路径（``logs/app.log``）。"""
    return log_dir() / "app.log"


#: 时长单位 -> 秒（把 ``'30 days'`` / ``'12 h'`` 这类配置解析成秒）
_DURATION_UNITS = {
    "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hour": 3600, "hours": 3600,
    "d": 86400, "day": 86400, "days": 86400,
    "w": 604800, "week": 604800, "weeks": 604800,
}
_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([A-Za-z]*)\s*$")


def _keep_seconds(text: str) -> float:
    """把 ``'30 days'`` / ``'12 h'`` 解析成秒；解析不了返回 0（= 不按时间清理）。"""
    match = _DURATION_RE.match(text or "")
    if match is None:
        return 0.0
    unit = (match.group(2) or "s").lower()
    return float(match.group(1)) * _DURATION_UNITS.get(unit, 0.0)


def _safe_unlink(path: str) -> None:
    """删日志文件；删不掉（被占用 / 已经没了）就算了，绝不能因此打断日志写入。"""
    try:
        os.remove(path)
    except OSError:
        pass


def _size_of(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _make_retention(
    keep_sec: float, max_total_mb: int, max_files: int
) -> Callable[[list[str]], None]:
    """构造 loguru 的 ``retention`` 回调：**三个条件一起裁，且永远至少留一份**。

    loguru 0.7.3 的 ``retention`` 一次只认一个条件（传列表直接报错），所以自己裁：
    ① 先删超过保留时长的；② 再删超过份数上限的；③ 最后按总量从最旧的删到不超限。
    一律从最旧的删起 —— 且 loguru 是在**新文件创建之前**调用本函数的，所以「当前
    正在写的那份」天然不在待裁列表里：``LOG_MAX_FILES`` / ``LOG_MAX_TOTAL_MB``
    约束的都是**历史文件**，磁盘上实际会是「上限份数 + 1 个当前文件」。
    """

    def _retention(logs: list[str]) -> None:
        paths = [path for path in logs if os.path.isfile(path)]
        if len(paths) < 2:
            return                                # 只剩一份：留着，别把当前日志删了

        def _mtime(path: str) -> float:
            try:
                return os.stat(path).st_mtime
            except OSError:
                return 0.0

        ordered = sorted(paths, key=_mtime)       # 旧 -> 新
        now = time.time()

        # ① 超过保留时长
        if keep_sec > 0:
            alive: list[str] = []
            for path in ordered:
                if now - _mtime(path) > keep_sec:
                    _safe_unlink(path)
                else:
                    alive.append(path)
            ordered = alive

        # ② 份数上限
        if max_files > 0 and len(ordered) > max_files:
            for path in ordered[: len(ordered) - max_files]:
                _safe_unlink(path)
            ordered = ordered[-max_files:]

        # ③ 总量上限
        if max_total_mb > 0:
            limit = max_total_mb * 1024 * 1024
            total = sum(_size_of(path) for path in ordered)
            while len(ordered) > 1 and total > limit:
                oldest = ordered.pop(0)
                total -= _size_of(oldest)
                _safe_unlink(oldest)

    return _retention


def _retention_policy() -> Any:
    """按配置决定保留策略：没有额外上限时直接用 loguru 原生语法，反之用组合裁剪。"""
    if settings.log_max_total_mb <= 0 and settings.log_max_files <= 0:
        return settings.log_keep
    return _make_retention(
        _keep_seconds(settings.log_keep),
        settings.log_max_total_mb,
        settings.log_max_files,
    )


def setup_logging() -> None:
    """初始化日志，幂等调用。"""
    global _CONFIGURED
    if _CONFIGURED:
        return

    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        colorize=True,
        format=_CONSOLE_FORMAT,
    )
    logger.add(
        log_file(),
        level=settings.log_level,
        rotation=settings.log_rotate,
        retention=_retention_policy(),
        encoding="utf-8",
        # 不打印变量快照/调用栈细节：日志可能被整份拷走，避免顺带泄漏上下文数据
        backtrace=False,
        diagnose=False,
        format=_FILE_FORMAT,
    )
    logger.configure(extra={"module": settings.app_name})
    _CONFIGURED = True


def get_logger(module: str):
    """按模块获取 logger，自动附带 module 字段。"""
    setup_logging()
    return logger.bind(module=module)


__all__ = ["setup_logging", "get_logger", "log_dir", "log_file", "logger"]
