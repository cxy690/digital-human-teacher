"""数据库基础设施：MySQL 优先的 SQLAlchemy 引擎与会话工厂。

设计要点（与 ``_cloud_pull/db`` 的旧实现相比）：
* **引擎惰性创建**：只有真正访问数据库时才建连，避免「仅 import 就要求 pymysql 且必须连得上」；
* **地址可覆盖**：``set_database_url()`` 供单测（SQLite）与多环境切换；
* **自动建库**：MySQL 下目标库不存在时自动 ``CREATE DATABASE``，做到开箱即跑；
* 任何失败都抛给调用方，由 server 启动阶段降级处理（不阻断启动）。
"""
from __future__ import annotations

import re
import threading

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL, make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config.logging import get_logger
from config.settings import settings

log = get_logger("db.base")

#: 库名白名单（用于拼接 ``CREATE DATABASE`` 时的安全校验，防注入）
_DBNAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


class Base(DeclarativeBase):
    """ORM 声明基类。"""


_lock = threading.Lock()
_engine: Engine | None = None
_session_factory: sessionmaker | None = None
_url_override: str | None = None


def set_database_url(url: str) -> None:
    """覆盖数据库地址并丢弃旧引擎（单测 / 多环境用）。"""
    global _url_override, _engine, _session_factory
    with _lock:
        _url_override = url
        if _engine is not None:
            try:
                _engine.dispose()
            except Exception:  # noqa: BLE001 - 释放失败不影响切换
                pass
        _engine = None
        _session_factory = None


def get_database_url() -> str:
    """当前生效的数据库地址（覆盖值优先于配置）。"""
    return _url_override or settings.database_url


def _create_engine(url: str) -> Engine:
    kwargs: dict = {"future": True, "echo": settings.db_echo, "pool_pre_ping": True}
    if url.startswith("sqlite"):
        # SQLite 在多线程（asyncio.to_thread）下需放开同线程限制
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


def get_engine() -> Engine:
    """获取（并按需惰性创建）全局引擎。"""
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                url = get_database_url()
                _engine = _create_engine(url)
                log.info(
                    "数据库引擎就绪：{}",
                    make_url(url).render_as_string(hide_password=True),
                )
    return _engine


def get_session_factory() -> sessionmaker:
    """获取（并按需惰性创建）会话工厂。"""
    global _session_factory
    if _session_factory is None:
        # 先取引擎（它会自行加锁），**不要**在本函数持锁时再调用 get_engine()：
        # ``_lock`` 是非重入锁，嵌套获取会自锁死（首次入口就是 get_session() 时必现）。
        engine = get_engine()
        with _lock:
            if _session_factory is None:
                _session_factory = sessionmaker(
                    bind=engine, expire_on_commit=False, class_=Session
                )
    return _session_factory


def get_session() -> Session:
    """获取一个数据库会话（调用方负责关闭，建议 ``with get_session() as db:``）。"""
    return get_session_factory()()


def ensure_database() -> None:
    """MySQL：目标库不存在则创建（连到服务器、不带库名）。非 MySQL 直接跳过。"""
    url = make_url(get_database_url())
    if not url.drivername.startswith("mysql"):
        return
    dbname = (url.database or "").strip().strip("`")
    if not dbname:
        return
    if not _DBNAME_RE.match(dbname):
        log.warning("数据库名不合法，跳过自动建库：{}", dbname)
        return
    # 注意：不能用 ``url.set(database=None)`` —— SQLAlchemy 的 URL.set 把 None 当作
    # 「保持原值」，结果仍连到不存在的库并报 1049。这里显式构造「不带库名」的服务器 URL。
    server_url = URL.create(
        drivername=url.drivername,
        username=url.username,
        password=url.password,
        host=url.host,
        port=url.port,
        query=dict(url.query),
    )
    server_engine = create_engine(server_url, future=True)
    try:
        with server_engine.connect() as conn:
            conn.execute(
                text(
                    f"CREATE DATABASE IF NOT EXISTS `{dbname}` "
                    "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
            )
            conn.commit()
    finally:
        server_engine.dispose()


def init_db() -> None:
    """建库 + 建表（幂等）。"""
    from db import models  # noqa: F401 - 触发 ORM 模型注册

    ensure_database()
    Base.metadata.create_all(get_engine())


def healthcheck() -> bool:
    """连通性检查（用于启动日志 / 健康检查）。"""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as err:  # noqa: BLE001 - 连不上只记日志
        log.warning("数据库连接失败：{}", err)
        return False


# ---------------------------------------------------------------- 兼容别名（惰性）
# 旧调用面里有 `from db.base import engine, SessionLocal`（db/__init__、db/runtime_sync、
# scripts/init_db_mysql）——旧实现是 import 时立即建引擎。这里用 PEP 562 模块级
# ``__getattr__`` 惰性代理：属性被访问时才创建引擎 / 会话工厂，保留「不在 import 期连库」
# 的收益，同时让旧调用一行不用改。
def __getattr__(name: str):
    if name == "engine":
        return get_engine()
    if name == "SessionLocal":
        return get_session_factory()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
