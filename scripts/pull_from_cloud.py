# -*- coding: utf-8 -*-
"""云端 -> 本地拉取（**只下载，绝不动云端**）。

为什么需要
----------
``sync_cloud.py`` 是单向「本地 -> 云端」。如果改动是**直接在云端做的**
（例如在云端跑校准脚本改了 ``teacher.json``），下一次同步就会用本地的旧值
把云端的新值覆盖掉——实际发生过：云端校准好的 ``voice_ref_text`` 被本地旧档案冲掉。

本工具把方向反过来：**以云端为准**，把云端与本地不一致的文件拉回本地，
使「本地 == 云端」。云端只读（只跑 ``md5sum`` 与 SFTP 下载），不做任何写入。

用法::

    python scripts/pull_from_cloud.py            # 拉回所有差异文件
    python scripts/pull_from_cloud.py --dry      # 只看差异清单，不下载
    python scripts/pull_from_cloud.py --only data/teachers   # 只拉某个目录
"""
from __future__ import annotations

import hashlib
import os
import posixpath
import sys
from pathlib import Path

import paramiko

try:  # 同目录直接运行（python scripts/pull_from_cloud.py）
    from sync_cloud import (
        EXCLUDE_DIRS, EXCLUDE_REL, EXCLUDE_SUFFIX, HOST, INCLUDE_DIRS,
        INCLUDE_FILES, PWD, PORT, REMOTE_ROOT, USER,
    )
except ImportError:  # 作为包运行（python -m scripts.pull_from_cloud）
    from scripts.sync_cloud import (
        EXCLUDE_DIRS, EXCLUDE_REL, EXCLUDE_SUFFIX, HOST, INCLUDE_DIRS,
        INCLUDE_FILES, PWD, PORT, REMOTE_ROOT, USER,
    )

LOCAL_ROOT = Path(__file__).resolve().parent.parent


def _wanted(rel: str, only: str) -> bool:
    if only and not (rel == only or rel.startswith(only + "/")):
        return False
    if set(rel.split("/")) & EXCLUDE_DIRS:
        return False
    if rel.endswith(EXCLUDE_SUFFIX):
        return False
    if rel in EXCLUDE_REL or any(rel.startswith(x + "/") for x in EXCLUDE_REL):
        return False
    return True


def _local_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    argv = sys.argv[1:]
    dry = "--dry" in argv
    only = ""
    if "--only" in argv:
        idx = argv.index("--only")
        if idx + 1 < len(argv):
            only = argv[idx + 1].strip().strip("/")

    dirs = " ".join(INCLUDE_DIRS)
    # 与 sync_cloud 的白名单/排除规则保持一致，保证两边比对的是同一批文件
    cmd = (
        f"cd {REMOTE_ROOT} && find {dirs} -type f"
        + "".join(f" ! -path '*/{d}/*'" for d in sorted(EXCLUDE_DIRS))
        + "".join(f" ! -name '*{s}'" for s in EXCLUDE_SUFFIX)
        + "".join(f" ! -path './{r}/*'" for r in sorted(EXCLUDE_REL))
        + " | xargs -d '\\n' md5sum"
    )
    for name in INCLUDE_FILES:
        cmd += f"; [ -f {name} ] && md5sum {name}"

    print(f"[pull] {USER}@{HOST}:{REMOTE_ROOT}  ->  {LOCAL_ROOT}   "
          f"mode={'DRY' if dry else 'PULL'}" + (f"   only={only}" if only else ""))

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30)
    _, out, err = c.exec_command(cmd, timeout=300)
    text = out.read().decode("utf-8", "ignore")
    errtext = err.read().decode("utf-8", "ignore")
    if errtext.strip() and "No such file" not in errtext:
        print("[warn] " + errtext.strip()[:500])

    sftp = c.open_sftp()

    same = pulled = failed = 0
    for line in text.splitlines():
        line = line.strip()
        if len(line) < 34 or "  " not in line:
            continue
        md5, _, raw = line.partition("  ")
        # 只剥开头的 "./"，不能用 lstrip("./")：后者按「字符集合」剥，会把
        # .agents/skills/... 的前导点一并吃掉，错认成 agents/skills/...，
        # 于是明明两边一致的技能文件被误报「本地缺文件」，
        # 真拉下来还会在本地多长出一棵 agents/skills/ 假树。
        rel = raw.replace("\\", "/")
        while rel.startswith("./"):
            rel = rel[2:]
        if not _wanted(rel, only):
            continue
        local = LOCAL_ROOT / rel
        if local.is_file() and _local_md5(local) == md5:
            same += 1
            continue
        if dry:
            print(f"  [dry] {rel}" + ("   (本地缺文件)" if not local.is_file() else "   (内容不同)"))
            pulled += 1
            continue
        try:
            local.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(posixpath.join(REMOTE_ROOT, rel), str(local))
            pulled += 1
            print(f"  [down] {rel}" + ("   (本地缺文件)" if not local.is_file() else "   (内容不同)"))
        except Exception as e:  # noqa: BLE001 - 单个文件失败不影响其它
            failed += 1
            print(f"  [FAIL] {rel}: {e}")

    sftp.close()
    c.close()
    print(f"[done] 一致={same} 拉回={pulled} 失败={failed}" + ("   (dry-run)" if dry else ""))


if __name__ == "__main__":
    if os.name == "nt":
        import sys as _s
        _s.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
