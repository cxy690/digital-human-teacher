"""PlaceholderAvatar（静态图 + 音频占位后端）单测。

覆盖：空音频/缺 ffmpeg/缺形象图 -> 回退 puppet；成功路径 -> 返回 mode:video。
"""
from __future__ import annotations

import base64
import types
from pathlib import Path

import pytest

from models import avatar as avatar_mod
from models.avatar import PlaceholderAvatar


async def test_empty_audio_returns_puppet() -> None:
    result = await PlaceholderAvatar().render(b"")
    assert result["mode"] == "puppet"


async def test_no_ffmpeg_returns_puppet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PlaceholderAvatar, "_resolve_ffmpeg", staticmethod(lambda: None))
    result = await PlaceholderAvatar().render(b"\x00\x01", text="hi")
    assert result["mode"] == "puppet"


async def test_missing_still_returns_puppet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PlaceholderAvatar, "_resolve_ffmpeg", staticmethod(lambda: "ffmpeg"))
    monkeypatch.setattr(PlaceholderAvatar, "_resolve_still", lambda self, ff: None)
    result = await PlaceholderAvatar().render(b"\x00\x01")
    assert result["mode"] == "puppet"


async def test_success_returns_video(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PlaceholderAvatar, "_resolve_ffmpeg", staticmethod(lambda: "ffmpeg"))
    monkeypatch.setattr(PlaceholderAvatar, "_resolve_still", lambda self, ff: "/tmp/x.png")

    def fake_run(cmd, **kwargs):  # 模拟 ffmpeg 产出：把最后一个参数当输出文件
        Path(cmd[-1]).write_bytes(b"FAKEMP4")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(avatar_mod.subprocess, "run", fake_run)
    result = await PlaceholderAvatar().render(b"\x00\x01", text="你好")
    assert result["mode"] == "video"
    assert result["placeholder"] is True
    assert base64.b64decode(result["video"]) == b"FAKEMP4"
