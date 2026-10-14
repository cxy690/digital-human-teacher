"""对比「本地项目」与「算力云」的文件差异（关机前体检用）。

判断维度与 ``sync_cloud.py`` 一致：文件大小 + mtime。
扫描范围也复用它的白名单，避免把日志/缓存算成差异。

输出四类：
    [未同步]  本地有、云端没有           -> 需要 sync_cloud.py
    [本地新]  两边都有，但本地更新       -> 需要 sync_cloud.py
    [云端新]  两边都有，但云端更新       -> 云端改动没拉回来（关机可能丢）
    [仅云端]  云端有、本地完全没有       -> 云端运行时产物或本地漏拉

用法：
    python scripts/check_sync.py
"""
from __future__ import annotations

import posixpath
import sys
from pathlib import Path

import paramiko

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sync_cloud import (  # noqa: E402
    EXCLUDE_DIRS,
    EXCLUDE_REL,
    EXCLUDE_SUFFIX,
    HOST,
    INCLUDE_DIRS,
    INCLUDE_FILES,
    LOCAL_ROOT,
    PORT,
    PWD,
    REMOTE_ROOT,
    USER,
    iter_local_files,
)


# 文本文件在 Windows(CRLF) 与 Linux(LF) 之间会差「行数」个字节，内容其实一样。
# 对这些后缀比大小会误报「不一致」，改为读出来规范化后比对。
_TEXT_SUFFIX = {
    ".py", ".md", ".json", ".txt", ".html", ".js", ".css",
    ".sh", ".toml", ".yml", ".yaml", ".csv", ".ini", ".cfg",
}


def same_text(sftp: paramiko.SFTPClient, rel: str) -> bool:
    """忽略行尾差异比较文本内容；读不到就返回 False（当作真差异）。"""
    try:
        with sftp.open(posixpath.join(REMOTE_ROOT, rel), "rb") as handle:
            remote = handle.read().decode("utf-8", "ignore")
        local = (LOCAL_ROOT / rel).read_text(encoding="utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        return False
    return remote.replace("\r\n", "\n") == local.replace("\r\n", "\n")


def walk_remote(sftp: paramiko.SFTPClient) -> dict[str, tuple[int, float]]:
    """云端白名单内的文件 -> ``{相对路径: (字节数, mtime)}``。"""
    out: dict[str, tuple[int, float]] = {}

    def walk(rpath: str) -> None:
        try:
            entries = sftp.listdir_attr(rpath)
        except IOError:                       # 目录不存在（如云端没这个模块）
            return
        for entry in entries:
            full = posixpath.join(rpath, entry.filename)
            rel = posixpath.relpath(full, REMOTE_ROOT)
            if set(rel.split("/")) & EXCLUDE_DIRS:
                continue
            if rel in EXCLUDE_REL or any(rel.startswith(x + "/") for x in EXCLUDE_REL):
                continue
            if entry.st_mode and (entry.st_mode & 0o040000):   # 目录
                walk(full)
                continue
            if Path(rel).suffix in EXCLUDE_SUFFIX:
                continue
            out[rel] = (int(entry.st_size or 0), float(entry.st_mtime or 0.0))

    for name in INCLUDE_DIRS:
        walk(posixpath.join(REMOTE_ROOT, name))
    for name in INCLUDE_FILES:                # 根目录单文件（README 等）
        full = posixpath.join(REMOTE_ROOT, name)
        try:
            st = sftp.stat(full)
        except IOError:
            continue
        out[name] = (int(st.st_size or 0), float(st.st_mtime or 0.0))
    return out


def main() -> None:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30)
    sftp = client.open_sftp()
    remote = walk_remote(sftp)

    local: dict[str, tuple[int, float]] = {}
    for fp, rel in iter_local_files():
        st = fp.stat()
        local[rel] = (st.st_size, st.st_mtime)

    print(f"本地文件 {len(local)} 个 | 云端文件 {len(remote)} 个")
    print(f"本地根: {LOCAL_ROOT}")
    print(f"云端根: {USER}@{HOST}:{REMOTE_ROOT}\n")

    missing_remote: list[str] = []            # 本地有，云端没有
    newer_local: list[str] = []               # 本地更新
    newer_remote: list[str] = []              # 云端更新（危险：关机可能丢）
    size_diff: list[str] = []                 # 大小不一致

    for rel, (size, mtime) in sorted(local.items()):
        r = remote.get(rel)
        if r is None:
            missing_remote.append(rel)
            continue
        if r[0] != size:
            if Path(rel).suffix in _TEXT_SUFFIX and same_text(sftp, rel):
                continue                      # 仅行尾不同，内容一致
            size_diff.append(f"{rel}  本地 {size}B / 云端 {r[0]}B")
        elif abs(r[1] - mtime) > 1:
            (newer_local if mtime > r[1] else newer_remote).append(rel)

    only_remote = [rel for rel in sorted(remote) if rel not in local]

    def dump(title: str, items: list[str], limit: int = 200) -> None:
        print(f"--- {title}：{len(items)} 个 ---")
        for item in items[:limit]:
            print("   " + item)
        if len(items) > limit:
            print(f"   … 还有 {len(items) - limit} 个")
        if not items:
            print("   （无）")
        print()

    dump("未同步到云端（本地有、云端无）", missing_remote)
    dump("本地较新（需要上传）", newer_local)
    dump("内容大小不一致", size_diff)
    dump("云端较新（本地落后，关机前应拉回）", newer_remote)
    dump("仅云端存在（本地没有）", only_remote)

    same = len(local) - len(missing_remote) - len(newer_local) - len(size_diff) - len(newer_remote)
    print(f"结论：{same} 个文件完全一致；"
          f"{len(missing_remote) + len(newer_local) + len(size_diff)} 个待上传；"
          f"{len(newer_remote) + len(only_remote)} 个需确认")

    sftp.close()
    client.close()


if __name__ == "__main__":
    main()
