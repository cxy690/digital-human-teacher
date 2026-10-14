"""AI 教师驱动视频源（core/avatar_source.py）单测。"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.avatar_source import (
    compute_file_hash,
    get_avatar_source_info,
    resolve_teacher_video,
    validate_teacher_video,
)

# 归档旧素材中的主底片，用作"可用的真实视频"样本
_REAL_VIDEO = Path("data/ai_teacher/talk.mp4")


# --------------------------------------------------------------- compute_file_hash
def test_compute_file_hash_deterministic(tmp_path: Path) -> None:
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello-education")
    assert compute_file_hash(f) == compute_file_hash(f)
    assert len(compute_file_hash(f)) == 16  # 截取前 16 位


def test_compute_file_hash_changes_with_content(tmp_path: Path) -> None:
    f1 = tmp_path / "a.bin"
    f1.write_bytes(b"aaa")
    f2 = tmp_path / "b.bin"
    f2.write_bytes(b"bbb")
    assert compute_file_hash(f1) != compute_file_hash(f2)


def test_compute_file_hash_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        compute_file_hash(tmp_path / "nope.bin")


# --------------------------------------------------------------- validate_teacher_video
def test_validate_missing_file_fails(tmp_path: Path) -> None:
    ok, reason = validate_teacher_video(tmp_path / "nope.mp4")
    assert not ok
    assert "不存在" in reason


def test_validate_real_video_passes() -> None:
    if not _REAL_VIDEO.is_file():
        pytest.skip("缺少样例视频 data/legacy/talk.mp4")
    ok, reason = validate_teacher_video(_REAL_VIDEO)
    assert ok, reason


# --------------------------------------------------------------- resolve_teacher_video
def test_resolve_uses_primary_when_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _REAL_VIDEO.is_file():
        pytest.skip("缺少样例视频 data/legacy/talk.mp4")
    monkeypatch.setattr("core.avatar_source.settings.ai_teacher_video_path", str(_REAL_VIDEO))
    path, used_fallback = resolve_teacher_video()
    assert str(path) == str(_REAL_VIDEO)
    assert used_fallback is False


def test_resolve_falls_back_when_primary_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    if not _REAL_VIDEO.is_file():
        pytest.skip("缺少样例视频 data/legacy/talk.mp4")
    monkeypatch.setattr(
        "core.avatar_source.settings.ai_teacher_video_path", str(tmp_path / "missing.mp4")
    )
    monkeypatch.setattr("core.avatar_source.settings.ai_teacher_fallback_path", str(_REAL_VIDEO))
    path, used_fallback = resolve_teacher_video()
    assert str(path) == str(_REAL_VIDEO)
    assert used_fallback is True


# --------------------------------------------------------------- get_avatar_source_info
def test_get_avatar_source_info_has_hash() -> None:
    if not _REAL_VIDEO.is_file():
        pytest.skip("缺少样例视频 data/legacy/talk.mp4")
    info = get_avatar_source_info()
    assert info.path
    assert len(info.video_hash) == 16  # 内容哈希，供缓存失效判定
    assert info.clip_dir
    assert info.cache_dir
    assert isinstance(info.used_fallback, bool)
