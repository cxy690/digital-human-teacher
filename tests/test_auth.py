"""身份认证单测：密码哈希 / JWT / 用户服务 / REST 接口 / WebSocket 握手鉴权。

用 SQLite 临时库跑，不依赖本机 MySQL；MySQL 的真实建库建表由 ``init_db()`` 在
server 启动时执行（本测试同样覆盖 ``init_db`` 的建表路径）。
"""
from __future__ import annotations

import pytest

from core.auth.security import (
    TokenError,
    decode_token,
    encode_token,
    hash_password,
    verify_password,
)
from core.auth.service import AuthError, authenticate, get_user, register_user, seed_demo_users
from db import base as db_base

SECRET = "unit-test-secret"


@pytest.fixture()
def sqlite_db(tmp_path):
    """把数据层切到临时 SQLite 并建表。"""
    db_base.set_database_url(f"sqlite:///{tmp_path / 'auth.db'}")
    db_base.init_db()
    yield
    db_base.set_database_url(f"sqlite:///{tmp_path / 'auth_after.db'}")


# --------------------------------------------------------------------------- 安全原语
def test_password_hash_roundtrip():
    stored = hash_password("123456")
    assert stored.startswith("pbkdf2_sha256$")
    assert verify_password("123456", stored) is True
    assert verify_password("wrong", stored) is False
    # 相同密码两次哈希不同（随机盐）
    assert hash_password("123456") != stored


def test_verify_password_rejects_garbage():
    assert verify_password("x", "not-a-hash") is False
    assert verify_password("x", "") is False


def test_token_roundtrip():
    token = encode_token({"sub": "u1", "role": "student"}, SECRET, ttl_seconds=60)
    payload = decode_token(token, SECRET)
    assert payload["sub"] == "u1"
    assert payload["role"] == "student"
    assert payload["exp"] > payload["iat"]


def test_token_rejects_tamper_and_expiry():
    token = encode_token({"sub": "u1"}, SECRET, ttl_seconds=60)
    with pytest.raises(TokenError):
        decode_token(token, "another-secret")
    with pytest.raises(TokenError):
        decode_token(token + "x", SECRET)
    expired = encode_token({"sub": "u1"}, SECRET, ttl_seconds=-1)
    with pytest.raises(TokenError):
        decode_token(expired, SECRET)


# --------------------------------------------------------------------------- 用户服务
def test_register_and_authenticate(sqlite_db):
    user = register_user("alice", "secret1", role="student", grade="初一")
    assert user["username"] == "alice"
    assert user["role"] == "student"
    assert "password_hash" not in user          # 绝不外泄哈希

    logged = authenticate("alice", "secret1")
    assert logged["id"] == user["id"]
    assert get_user(user["id"])["username"] == "alice"


def test_register_duplicate_conflicts(sqlite_db):
    register_user("bob", "secret1")
    with pytest.raises(AuthError) as exc:
        register_user("bob", "secret1")
    assert exc.value.code == 409


def test_register_validates_input(sqlite_db):
    with pytest.raises(AuthError):
        register_user("ab", "secret1")          # 用户名过短
    with pytest.raises(AuthError):
        register_user("carol", "123")           # 密码过短
    with pytest.raises(AuthError):
        register_user("dave", "secret1", role="hacker")


def test_authenticate_wrong_password(sqlite_db):
    register_user("eve", "secret1")
    with pytest.raises(AuthError) as exc:
        authenticate("eve", "wrong1")
    assert exc.value.code == 401
    with pytest.raises(AuthError):
        authenticate("nobody", "secret1")


def test_seed_demo_users_is_idempotent(sqlite_db):
    assert len(seed_demo_users()) == 4
    assert seed_demo_users() == []              # 第二次为空（已存在）
    assert authenticate("admin", "123456")["role"] == "admin"


# --------------------------------------------------------------------------- REST 接口
def _rest_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from server.routes import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_rest_register_login_me(sqlite_db):
    client = _rest_client()

    resp = client.post("/api/auth/register", json={"username": "stu1", "password": "secret1"})
    assert resp.status_code == 200
    token = resp.json()["token"]
    assert resp.json()["user"]["username"] == "stu1"

    # 重复注册
    assert client.post(
        "/api/auth/register", json={"username": "stu1", "password": "secret1"}
    ).status_code == 409

    # 登录
    login = client.post("/api/auth/login", json={"username": "stu1", "password": "secret1"})
    assert login.status_code == 200
    token = login.json()["token"]

    # 错误密码
    assert client.post(
        "/api/auth/login", json={"username": "stu1", "password": "bad123"}
    ).status_code == 401

    # 带令牌访问 /me
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["user"]["username"] == "stu1"

    # 不带 / 带错令牌
    assert client.get("/api/auth/me").status_code == 401
    assert client.get(
        "/api/auth/me", headers={"Authorization": "Bearer bad.token.xx"}
    ).status_code == 401


# --------------------------------------------------------------------------- WebSocket 鉴权
def _ws_app():
    from fastapi import FastAPI

    from server.websocket import websocket_endpoint

    app = FastAPI()
    app.add_api_websocket_route("/ws", websocket_endpoint)
    return app


def _first_ws_message(app, path: str, init: dict):
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    client = TestClient(app)
    try:
        with client.websocket_connect(path) as ws:
            ws.send_json(init)
            return ws.receive_json()
    except WebSocketDisconnect as err:  # pragma: no cover - 视关闭时机而定
        return {"type": "closed", "code": err.code}


def test_ws_rejects_invalid_token(sqlite_db, monkeypatch):
    from config.settings import settings

    monkeypatch.setattr(settings, "auth_enable", True)
    monkeypatch.setattr(settings, "auth_required", False)
    msg = _first_ws_message(_ws_app(), "/ws?token=bad.token.value", {"type": "init"})
    assert msg["type"] == "error"
    assert msg["payload"]["code"] == 4401


def test_ws_requires_token_when_enforced(sqlite_db, monkeypatch):
    from config.settings import settings

    monkeypatch.setattr(settings, "auth_enable", True)
    monkeypatch.setattr(settings, "auth_required", True)
    msg = _first_ws_message(_ws_app(), "/ws", {"type": "init"})
    assert msg["type"] == "error"
    assert msg["payload"]["code"] == 4401


def test_ws_anonymous_allowed_in_gray_mode(sqlite_db, monkeypatch):
    from config.settings import settings

    monkeypatch.setattr(settings, "auth_enable", True)
    monkeypatch.setattr(settings, "auth_required", False)
    msg = _first_ws_message(_ws_app(), "/ws", {"type": "init", "student_id": "anon-1"})
    assert msg["type"] == "ready"
