"""用户仓储与认证业务逻辑（MySQL 存储，见 ``db/models.py``）。

只做「身份认证」：注册、登录、按 id 取用户。授权（谁能访问谁的数据）由
调用方（REST 依赖 / WebSocket 端点）基于返回的 ``role`` / ``id`` 决定。
"""
from __future__ import annotations

import time
import uuid

from sqlalchemy import select

from config.logging import get_logger
from core.auth.security import hash_password, verify_password
from db.base import get_session
from db.models import ROLES, User

log = get_logger("core.auth.service")

MIN_USERNAME = 3
MAX_USERNAME = 64
MIN_PASSWORD = 6
MAX_PASSWORD = 128


class AuthError(Exception):
    """认证 / 注册的业务错误（带 HTTP 语义的 ``code``）。"""

    def __init__(self, message: str, *, code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


def _validate_credentials(username: str, password: str) -> tuple[str, str]:
    username = (username or "").strip()
    if not (MIN_USERNAME <= len(username) <= MAX_USERNAME):
        raise AuthError(f"用户名长度需在 {MIN_USERNAME}-{MAX_USERNAME} 之间")
    if not (MIN_PASSWORD <= len(password or "") <= MAX_PASSWORD):
        raise AuthError(f"密码长度需在 {MIN_PASSWORD}-{MAX_PASSWORD} 之间")
    return username, password


def register_user(
    username: str,
    password: str,
    *,
    role: str = "student",
    display_name: str = "",
    grade: str = "",
) -> dict:
    """注册新用户并返回公开视图。用户名重复 -> ``AuthError(409)``。"""
    username, password = _validate_credentials(username, password)
    role = (role or "student").strip().lower()
    if role not in ROLES:
        raise AuthError(f"角色不合法：{role}")
    with get_session() as db:
        if db.scalar(select(User).where(User.username == username)) is not None:
            raise AuthError("用户名已被占用", code=409)
        user = User(
            id=uuid.uuid4().hex,
            username=username,
            password_hash=hash_password(password),
            role=role,
            display_name=(display_name or username).strip(),
            grade=(grade or "").strip(),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        log.info("新用户注册：{}（{}）", user.username, user.role)
        return user.to_public()


def authenticate(username: str, password: str) -> dict:
    """校验账号密码并返回公开视图。失败 -> ``AuthError(401/403)``。"""
    username = (username or "").strip()
    with get_session() as db:
        user = db.scalar(select(User).where(User.username == username))
        if user is None or not verify_password(password or "", user.password_hash):
            raise AuthError("用户名或密码错误", code=401)
        if not user.is_active:
            raise AuthError("账号已被停用", code=403)
        user.last_login_at = time.time()
        db.commit()
        db.refresh(user)
        return user.to_public()


def get_user(user_id: str) -> dict | None:
    """按 id 取用户公开视图。"""
    if not user_id:
        return None
    with get_session() as db:
        user = db.get(User, user_id)
        return user.to_public() if user is not None else None


def seed_demo_users() -> list[str]:
    """预置演示账号（幂等）：student / parent / teacher / admin，密码均为 123456。"""
    accounts = (
        ("student", "student", "小明同学", "初一"),
        ("parent", "parent", "小明家长", ""),
        ("teacher", "teacher", "王老师", ""),
        ("admin", "admin", "管理员", ""),
    )
    created: list[str] = []
    for username, role, display_name, grade in accounts:
        try:
            register_user(
                username,
                "123456",
                role=role,
                display_name=display_name,
                grade=grade,
            )
            created.append(username)
        except AuthError as err:
            if err.code != 409:  # 已存在则跳过
                raise
    return created
