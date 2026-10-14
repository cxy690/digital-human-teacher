"""clear_cache 清理脚本单测。"""
from __future__ import annotations

from pathlib import Path

from scripts.clear_cache import clear_cache, find_cache_files


def _seed(root: Path) -> None:
    (root / "v1").mkdir(parents=True)
    (root / "v1" / "abc_s0.json").write_text("{}", encoding="utf-8")
    (root / "v1" / "def_s0.json").write_text("{}", encoding="utf-8")
    (root / "v2").mkdir(parents=True)
    (root / "v2" / "xyz_s0.json").write_text("{}", encoding="utf-8")


def test_find_files_all_and_by_version(tmp_path: Path) -> None:
    _seed(tmp_path)
    assert len(find_cache_files(tmp_path)) == 3
    assert len(find_cache_files(tmp_path, version="v1")) == 2


def test_dry_run_keeps_files(tmp_path: Path) -> None:
    _seed(tmp_path)
    files = clear_cache(tmp_path, dry_run=True)
    assert len(files) == 3
    assert all(f.is_file() for f in files)


def test_clear_removes_files(tmp_path: Path) -> None:
    _seed(tmp_path)
    clear_cache(tmp_path)
    assert find_cache_files(tmp_path) == []


def test_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert find_cache_files(tmp_path / "nope") == []
