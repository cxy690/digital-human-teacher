"""FastAPI 依赖：从 ``Authorization: Bearer <token>`` 解析并校验当前用户。"""
from __future__ import annotations

from fastapi import Depends, Header, HTTPException

from config.settings import settings
from core.auth.security import TokenError, decode_token
from core.auth.service import get_user


def extract_token(authorization: str | None) -> str:
    """从 Authorization 头中取出裸 token（兼容不带 ``Bearer`` 前缀）。"""
    if not authorization:
        return ""
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return authorization.strip()


def user_from_token(token: str) -> dict | None:
    """校验 token 并加载用户；任何失败返回 ``None``（不抛异常，供 WS 复用）。"""
    if not token:
        return None
    try:
        payload = decode_token(token, settings.auth_secret)
    except TokenError:
        return None
    return get_user(str(payload.get("sub") or ""))


def current_user(authorization: str | None = Header(default=None)) -> dict:
    """REST 依赖：强制携带合法 Token，返回用户公开信息。"""
    token = extract_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="缺少身份凭证")
    user = user_from_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="身份凭证无效或已过期")
    return user


def require_role(*roles: str):
    """生成一个「要求指定角色」的 REST 依赖。"""

    def _dep(user: dict = Depends(current_user)) -> dict:
        if roles and user.get("role") not in roles:
            raise HTTPException(status_code=403, detail="无权访问该资源")
        return user

    return _dep


def require_developer(authorization: str | None = Header(default=None)) -> dict:
    """REST 依赖：**项目开发人员**门禁（Skills 管理平台等运维面专用）。

    区分方式 = 登录账号的角色：``admin`` 才算开发人员 —— 注册接口只允许
    student/parent/teacher，``admin`` 只能由 :func:`core.auth.service.seed_demo_users`
    预置（或运维改库），天然是「内部账号」。
    ``AUTH_ENABLE=false`` 时直接拒绝：没有身份体系就没有「谁能进管理面」的判定依据，
    宁可不可用也不能裸奔。
    """
    if not settings.auth_enable:
        raise HTTPException(
            status_code=403, detail="身份体系未启用（AUTH_ENABLE=false），无法校验管理员身份")
    user = current_user(authorization)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="仅项目开发人员（admin）可访问")
    return user
