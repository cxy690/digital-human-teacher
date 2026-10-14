# -*- coding: utf-8 -*-
"""把云端会话数据同步到本机 MySQL（Navicat 可视化查看）。

背景：云端应用把每轮对话落在 ``data/raw/checkpoints/*.json``（会话检查点）。
本脚本在**本机**运行，两步完成同步（云端应用无需任何改动）：

    1) 从云端增量拉取检查点 -> 本地 ``data/raw/cloud/checkpoints/``
    2) 解析导入本机 MySQL（.env 的 DATABASE_URL）：``students / sessions / messages`` 三张表

用法::

    python scripts/pull_chat_to_mysql.py              # 拉取 + 导入（推荐，随时可重复跑）
    python scripts/pull_chat_to_mysql.py --no-pull    # 跳过云端拉取，只导本地已有文件
    python scripts/pull_chat_to_mysql.py --pull-only  # 只拉取不导入（MySQL 还没配好时先攒数据）
    python scripts/pull_chat_to_mysql.py --force      # 云端文件全量重拉

幂等说明：students / sessions 按主键 upsert；messages 按会话「先删后插」重建 ——
同一会话重复同步不会产生重复消息，随时可以再跑。
"""
from __future__ import annotations

import argparse
import json
import os
import posixpath
import stat
import sys
import time
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # 相对路径（data/…）以项目根为准
try:  # Windows 控制台中文输出
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

try:  # 同目录直接运行
    from cloud_ctl import HOST, PORT, PWD, ROOT as CLOUD_ROOT, USER
except ImportError:  # 作为包运行
    from scripts.cloud_ctl import HOST, PORT, PWD, ROOT as CLOUD_ROOT, USER  # type: ignore

from sqlalchemy.exc import OperationalError, SQLAlchemyError

REMOTE_DIR = posixpath.join(CLOUD_ROOT, "data/raw/checkpoints")
# 导入来源：本地跑出来的会话 + 从云端拉回来的会话（两个目录都扫）
LOCAL_DIRS = [
    ROOT / "data" / "raw" / "checkpoints",
    ROOT / "data" / "raw" / "cloud" / "checkpoints",
]


# ---------------------------------------------------------------- 云端拉取
def _connect() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30)
    return client


def pull(force: bool = False) -> tuple[int, int]:
    """增量拉取云端检查点（判据：文件大小 + mtime）；返回 (下载数, 跳过数)。"""
    target = LOCAL_DIRS[1]
    target.mkdir(parents=True, exist_ok=True)
    client = _connect()
    sftp = client.open_sftp()
    downloaded = skipped = 0
    try:
        try:
            entries = sftp.listdir_attr(REMOTE_DIR)
        except IOError:
            print(f"[pull] 云端还没有检查点目录：{REMOTE_DIR}")
            return 0, 0
        for attr in sorted(entries, key=lambda a: a.filename):
            if not stat.S_ISREG(attr.st_mode) or not attr.filename.endswith(".json"):
                continue
            local = target / attr.filename
            if (
                not force
                and local.is_file()
                and local.stat().st_size == attr.st_size
                and abs(local.stat().st_mtime - attr.st_mtime) <= 1
            ):
                skipped += 1
                continue
            sftp.get(posixpath.join(REMOTE_DIR, attr.filename), str(local))
            try:  # 同步 mtime，便于下次增量判断
                os.utime(local, (attr.st_atime, attr.st_mtime))
            except OSError:
                pass
            downloaded += 1
    finally:
        sftp.close()
        client.close()
    return downloaded, skipped


# ---------------------------------------------------------------- 导入 MySQL
def import_all() -> int:
    """把两个来源目录的检查点导入数据库（幂等）。"""
    from sqlalchemy.engine import make_url

    from config.settings import settings
    from db.base import get_session as open_db, init_db
    from db.importer import upsert_session

    # MySQL 场景：目标库不存在时先建库（sqlite 场景是空操作）
    try:
        from init_db_mysql import ensure_mysql_database
    except ImportError:  # 作为包运行
        from scripts.init_db_mysql import ensure_mysql_database  # type: ignore

    ensure_mysql_database(make_url(settings.database_url))

    files: list[Path] = []
    for directory in LOCAL_DIRS:
        if directory.is_dir():
            files.extend(sorted(directory.glob("*.json")))
    if not files:
        print("[import] 没有找到任何检查点文件（先跑一次不带 --no-pull 的同步）")
        return 1

    init_db()
    db = open_db()
    sessions_total = messages_total = 0
    students: set[str] = set()
    try:
        for path in files:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as err:  # noqa: BLE001 - 单个坏文件不阻断整批
                print(f"  [skip] {path.name}: {err}")
                continue
            sid = str(data.get("session_id") or path.stem)
            stu = str(data.get("student_id") or "unknown")
            history = data.get("history") or []
            saved_at = float(data.get("saved_at") or time.time())

            # 与运行时写库共用同一套导入逻辑（db.importer，幂等：先删后插重建消息）
            messages_total += upsert_session(
                db, session_id=sid, student_id=stu, history=history, saved_at=saved_at,
            )
            students.add(stu)
            sessions_total += 1
        db.commit()
    finally:
        db.close()
    print(f"[import] 导入完成：会话 {sessions_total} 个 · 消息 {messages_total} 条 · 学生 {len(students)} 名")
    print("         Navicat 里刷新 Education 连接即可查看（students / sessions / messages）")
    return 0


# ---------------------------------------------------------------- 入口
def main() -> int:
    parser = argparse.ArgumentParser(description="云端会话数据 -> 本机 MySQL 同步")
    parser.add_argument("--no-pull", action="store_true", help="跳过云端拉取，只导本地已有文件")
    parser.add_argument("--pull-only", action="store_true", help="只拉取不导入")
    parser.add_argument("--force", action="store_true", help="云端文件全量重拉")
    args = parser.parse_args()

    if not args.no_pull:
        print(f"[pull] {USER}@{HOST}:{REMOTE_DIR}  ->  {LOCAL_DIRS[1]}")
        try:
            downloaded, skipped = pull(force=args.force)
            print(f"[pull] 下载 {downloaded} 个，跳过 {skipped} 个（未变化）")
        except Exception as err:  # noqa: BLE001 - 网络/凭据问题都要给可执行的提示
            print(f"[pull] 拉取失败：{err}")
            print("       检查 .env 的 CLOUD_PORT / CLOUD_PWD（实例重启后会变）")
            if args.pull_only:
                return 1

    if args.pull_only:
        return 0

    try:
        return import_all()
    except OperationalError as err:
        print(f"[import] 连接失败：{err.orig}")
        print("        检查 .env 的 DATABASE_URL：密码是否已替换（含 @ : # 等字符需 URL 编码）")
        return 1
    except SQLAlchemyError as err:
        print(f"[import] 数据库错误：{err}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
