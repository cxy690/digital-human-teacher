"""身份认证包：用户存储（MySQL）+ 密码哈希 + JWT + FastAPI 依赖。

依赖链：``登录/鉴权`` -> ``可信 student_id`` -> ``学情存储``。
本包只解决「你是谁」（认证），「你能做什么」（授权）由调用方按 ``role``/``id`` 判定。
"""
from core.auth.deps import current_user, extract_token, require_role, user_from_token
from core.auth.security import (
    TokenError,
    decode_token,
    encode_token,
    hash_password,
    verify_password,
)
from core.auth.service import (
    AuthError,
    authenticate,
    get_user,
    register_user,
    seed_demo_users,
)

__all__ = [
    "AuthError",
    "TokenError",
    "authenticate",
    "current_user",
    "decode_token",
    "encode_token",
    "extract_token",
    "get_user",
    "hash_password",
    "register_user",
    "require_role",
    "seed_demo_users",
    "user_from_token",
    "verify_password",
]
