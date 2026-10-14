# -*- coding: utf-8 -*-
"""把**云端**的 ``logs/`` 拉回本地 ``logs/cloud/``：在本地就能看线上现场。

为什么要有它：日志文件在谁机器上就写谁机器上——云端服务产生的日志在云端磁盘，
本地那份只是本地跑出来的。以前排查线上问题得先 SSH 上去 ``tail``，
现在一条命令把线上日志**增量**同步到本地，用编辑器直接翻。

用法::

    python scripts/pull_logs.py            # 增量：只拉有变化的
    python scripts/pull_logs.py --force    # 全量重拉
    python scripts/pull_logs.py --list     # 只看云端有哪些日志、多大

行为约定（与 ``sync_cloud.py`` 一致）：
* 增量判据是 **文件大小 + mtime**，一致就跳过，不重复下载；
* **只增不删**：云端删掉的日志，本地副本保留（那是历史证据）；
* 拉下来的副本落在 ``logs/cloud/``，与本地日志分开放，不会混淆；
* 已被 ``.gitignore`` 排除，不会误提交。
"""
from __future__ import annotations

import os
import posixpath
import stat
import sys
from pathlib import Path

import paramiko

try:  # 同目录直接运行（python scripts/pull_logs.py）
    from cloud_ctl import HOST, PWD, ROOT, USER, PORT
except ImportError:  # 作为包运行（python -m scripts.pull_logs）
    from scripts.cloud_ctl import HOST, PWD, ROOT, USER, PORT

LOCAL_ROOT = Path(__file__).resolve().parent.parent
REMOTE_LOGS = posixpath.join(ROOT, "logs")


def local_dir() -> Path:
    """本地副本目录：``logs/cloud/``。"""
    path = LOCAL_ROOT / "logs" / "cloud"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _connect() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30)
    return client


def pull_logs(*, force: bool = False, quiet: bool = False) -> tuple[int, int]:
    """增量拉取云端日志；返回 ``(下载数, 跳过数)``。"""
    target_dir = local_dir()
    client = _connect()
    sftp = client.open_sftp()
    downloaded = skipped = 0
    try:
        try:
            entries = sftp.listdir_attr(REMOTE_LOGS)
        except IOError:
            if not quiet:
                print(f"[logs] 云端还没有日志目录：{REMOTE_LOGS}")
            return 0, 0

        for attr in sorted(entries, key=lambda a: a.filename):
            if not stat.S_ISREG(attr.st_mode) or attr.filename.startswith("."):
                continue
            remote = posixpath.join(REMOTE_LOGS, attr.filename)
            local = target_dir / attr.filename
            if (
                not force
                and local.is_file()
                and local.stat().st_size == attr.st_size
                and abs(local.stat().st_mtime - attr.st_mtime) <= 1
            ):
                skipped += 1
                continue
            sftp.get(remote, str(local))
            try:  # 同步 mtime，便于下次增量判断
                os.utime(local, (attr.st_atime, attr.st_mtime))
            except OSError:
                pass
            downloaded += 1
            if not quiet:
                print(f"  [↓] {attr.filename:<20} {attr.st_size / 1024:8.1f} KB")
    finally:
        sftp.close()
        client.close()
    return downloaded, skipped


def list_remote() -> int:
    """只列出云端有哪些日志与大小（不下载）。"""
    client = _connect()
    sftp = client.open_sftp()
    try:
        try:
            entries = sftp.listdir_attr(REMOTE_LOGS)
        except IOError:
            print(f"[logs] 云端还没有日志目录：{REMOTE_LOGS}")
            return 0
        rows = [a for a in entries if stat.S_ISREG(a.st_mode) and not a.filename.startswith(".")]
        print(f"[logs] 云端 {REMOTE_LOGS}（{len(rows)} 个文件）")
        for attr in sorted(rows, key=lambda a: a.filename):
            print(f"  {attr.st_size / 1024:9.1f} KB  {attr.filename}")
        return 0
    finally:
        sftp.close()
        client.close()


def main() -> int:
    argv = sys.argv[1:]
    if "--list" in argv:
        return list_remote()
    force = "--force" in argv or "--all" in argv
    print(f"[logs] {USER}@{HOST}:{REMOTE_LOGS}  ->  {local_dir()}"
          + ("  (全量)" if force else "  (增量)"))
    downloaded, skipped = pull_logs(force=force)
    print(f"[logs] 拉取 {downloaded} 个，跳过 {skipped} 个（未变化）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
