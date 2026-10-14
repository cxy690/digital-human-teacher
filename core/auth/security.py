"""认证安全原语：密码哈希（PBKDF2-SHA256）与 JWT（HS256）。

刻意只用标准库：
* 密码哈希用 ``hashlib.pbkdf2_hmac`` —— 无需 bcrypt 的原生编译，Windows 友好；
* Token 用标准 HS256 JWT（自实现，避免引入 pyjwt 依赖）。

对外 API：``hash_password`` / ``verify_password`` / ``encode_token`` / ``decode_token``。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time

#: 密码哈希算法标识（存库格式：``algo$iterations$salt$digest``）
_ALGO = "pbkdf2_sha256"
#: PBKDF2 迭代次数（OWASP 2023 对 SHA256 的建议量级）
_ITERATIONS = 200_000


class TokenError(Exception):
    """Token 非法 / 签名错误 / 已过期。"""


# --------------------------------------------------------------------------- 密码
def hash_password(password: str) -> str:
    """生成带随机盐的密码哈希，可直接存库。"""
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), _ITERATIONS
    )
    return f"{_ALGO}${_ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """常量时间比对；任何格式异常一律视为不通过。"""
    try:
        algo, iterations, salt, digest = stored.split("$")
    except (ValueError, AttributeError):
        return False
    if algo != _ALGO:
        return False
    try:
        candidate = hashlib.pbkdf2_hmac(
            "sha256", (password or "").encode("utf-8"), bytes.fromhex(salt), int(iterations)
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(candidate.hex(), digest)


# --------------------------------------------------------------------------- JWT
def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(segment: str) -> bytes:
    return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))


def _sign(signing_input: bytes, secret: str) -> bytes:
    return hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()


def encode_token(payload: dict, secret: str, *, ttl_seconds: int) -> str:
    """签发 HS256 JWT（自动补 ``iat`` / ``exp``）。"""
    now = int(time.time())
    body = {"iat": now, "exp": now + int(ttl_seconds), **payload}
    header = {"alg": "HS256", "typ": "JWT"}
    segments = "{}.{}".format(
        _b64url_encode(json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")),
        _b64url_encode(json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")),
    )
    signature = _b64url_encode(_sign(segments.encode("ascii"), secret))
    return f"{segments}.{signature}"


def decode_token(token: str, secret: str) -> dict:
    """校验签名与时效并解出载荷；失败抛 :class:`TokenError`。"""
    parts = (token or "").split(".")
    if len(parts) != 3:
        raise TokenError("token 格式不合法")
    header_seg, payload_seg, signature_seg = parts
    signing_input = f"{header_seg}.{payload_seg}".encode("ascii", "ignore")
    expected = _b64url_encode(_sign(signing_input, secret))
    if not hmac.compare_digest(expected, signature_seg):
        raise TokenError("token 签名校验失败")
    try:
        payload = json.loads(_b64url_decode(payload_seg))
    except (ValueError, json.JSONDecodeError) as err:
        raise TokenError("token 载荷不合法") from err
    if not isinstance(payload, dict):
        raise TokenError("token 载荷不合法")
    exp = payload.get("exp")
    if isinstance(exp, (int, float)) and time.time() > exp:
        raise TokenError("token 已过期")
    return payload
