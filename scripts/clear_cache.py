"""清理数字人 AI 教师相关缓存（人脸预处理等运行时产物）。

缓存按 ``<缓存根>/<版本>/<内容hash>_s<bbox_shift>.json`` 存放；
换教师视频（内容变）后旧缓存会自动失效，本脚本用于手动清理/回收磁盘。

用法（推荐模块方式；直接 python scripts/clear_cache.py 亦可）：
    python -m scripts.clear_cache                 # 清理 data/cache 下全部（默认取配置）
    python -m scripts.clear_cache --version v1    # 只清理某一缓存版本
    python -m scripts.clear_cache --dry-run       # 只预览将删除的文件
    python -m scripts.clear_cache --dir data/cache
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 项目根加入模块搜索路径，兼容 `python scripts/clear_cache.py` 直接运行
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.logging import get_logger  # noqa: E402
from config.settings import settings  # noqa: E402

log = get_logger("scripts.clear_cache")


def find_cache_files(cache_dir: Path, version: str | None = None) -> list[Path]:
    """列出待清理的缓存文件（``version=None`` 表示全部版本）。"""
    root = Path(cache_dir)
    if not root.is_dir():
        return []
    base = root / version if version else root
    if not base.is_dir():
        return []
    return sorted(p for p in base.rglob("*") if p.is_file())


def clear_cache(cache_dir: Path, version: str | None = None, dry_run: bool = False) -> list[Path]:
    """清理缓存文件并移除空目录，返回被（将要）删除的文件列表。

    ``dry_run=True`` 时只返回列表、不实际删除。
    """
    files = find_cache_files(cache_dir, version)
    if dry_run:
        return files
    for f in files:
        try:
            f.unlink()
        except OSError as err:  # noqa: PERF203 - 单个失败不中断整体清理
            log.warning("删除失败：{}（{}）", f, err)
    # 自底向上移除空目录（保留缓存根目录本身）
    root = Path(cache_dir)
    if root.is_dir():
        for d in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            try:
                d.rmdir()
            except OSError:
                pass
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="清理数字人 AI 教师人脸预处理缓存")
    parser.add_argument("--dir", default=settings.avatar_cache_dir, help="缓存根目录（默认取配置）")
    parser.add_argument("--version", default=None, help="只清理指定缓存版本，如 v1")
    parser.add_argument("--dry-run", action="store_true", help="只预览不删除")
    args = parser.parse_args()

    files = clear_cache(Path(args.dir), version=args.version, dry_run=args.dry_run)
    if not files:
        log.info("没有可清理的缓存：{}", args.dir)
        return
    action = "将删除" if args.dry_run else "已删除"
    for f in files:
        log.info("{} {}", action, f)
    log.info("{} 个缓存文件（目录：{}，版本：{}）", len(files), args.dir, args.version or "全部")


if __name__ == "__main__":
    main()
