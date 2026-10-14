# -*- coding: utf-8 -*-
"""把算力云上的「代码 / 配置 / 文档」拉回本地暂存区 ``_cloud_pull/``。

为什么需要它：``sync_cloud.py`` 只做「本地 → 云端」单向推送，``check_sync.py`` 只比对。
排查「云端是不是有本地没有的改动」时，需要一个**只读、不会覆盖本地工作树**的拉取工具：
拉到 ``_cloud_pull/<相对路径>``，再用普通 diff 工具逐文件比对/挑选合并。

安全的默认行为：
    * **绝不写本地工作树**（只写 ``_cloud_pull/``），本地任何文件都不会被覆盖；
    * 排除模型权重（models_zoo / digital_human/models|musetalk）、``.venvs``、媒体素材
      （png/jpg/mp4/wav/mp3/pptx/db/log……）与 >2MB 的文件 —— 拉回的都是纯文本代码/配置/文档；
    * 排除云端 ``.env``（含密钥）；排除 ``data/cache``、``data/raw`` 等运行时数据；
    * 内容与暂存区完全一致的不会重写（增量，重复执行很快）。

用法：
    python scripts/pull_cloud.py              # 增量拉取到 _cloud_pull/
    python scripts/pull_cloud.py --dry        # 只列出会拉哪些文件
    python scripts/pull_cloud.py --only core  # 只拉某个目录/文件（改一两个文件时快）

连接信息可用环境变量覆盖：CLOUD_HOST / CLOUD_PORT / CLOUD_USER / CLOUD_PWD / CLOUD_ROOT
"""
from __future__ import annotations

import posixpath
import sys
from pathlib import Path

import paramiko

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sync_cloud import (  # noqa: E402
    EXCLUDE_DIRS,
    EXCLUDE_SUFFIX,
    HOST,
    LOCAL_ROOT,
    PWD,
    PORT,
    REMOTE_ROOT,
    USER,
)

STAGE = LOCAL_ROOT / "_cloud_pull"

DIRS = [
    "agents", "clients", "config", "core", "document_parsing", "mcp_tools",
    "models", "rag", "scripts", "server", "tests", "db", "memory", "deploy", "docs",
]
FILES = [
    "pyproject.toml", "requirements.txt", "README.md", "代码地图.md", "算力云部署.md",
    "数字人素材规格.md", "数字人部署指南.md", "本地开发联调.md", "结构对照审计.md",
    "待补齐规划.md", "接口实现状态.md", "_restart_engine.sh",
]
# 只拉这些后缀（纯文本代码/配置/文档）
KEEP_SUFFIX = {
    ".py", ".md", ".json", ".txt", ".html", ".js", ".css", ".sh", ".toml",
    ".yml", ".yaml", ".cfg", ".ini", ".example", ".mjs", ".ps1", ".bat",
}
SKIP_REL_PREFIX = (
    "data/", "deploy/cloud/logs", "clients/student_tablet/clips",
    "digital_human/models", "digital_human/musetalk",
)
MAX_BYTES = 2 * 1024 * 1024          # 单文件 >2MB 的文本一律跳过（防止误拉大文件）


def _norm_eol(text: str) -> str:
    """统一行尾后再比较，避免云端 LF 与本地 CRLF 造成假差异。"""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def main() -> None:
    argv = sys.argv[1:]
    dry = "--dry" in argv
    only = ""
    if "--only" in argv:
        idx = argv.index("--only")
        if idx + 1 < len(argv):
            only = argv[idx + 1].strip().strip("/")

    def wanted(rel: str) -> bool:
        return not only or rel == only or rel.startswith(only + "/")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30)
    sftp = client.open_sftp()

    targets: list[str] = []

    def walk(rpath: str) -> None:
        try:
            entries = sftp.listdir_attr(rpath)
        except IOError:                      # 目录不存在：云端没这个模块
            return
        for entry in entries:
            full = posixpath.join(rpath, entry.filename)
            rel = posixpath.relpath(full, REMOTE_ROOT)
            if set(rel.split("/")) & EXCLUDE_DIRS or rel.startswith("."):
                continue
            if rel.startswith(SKIP_REL_PREFIX):
                continue
            if entry.st_mode and (entry.st_mode & 0o040000):
                walk(full)
                continue
            suffix = Path(rel).suffix.lower()
            if suffix not in KEEP_SUFFIX or suffix in EXCLUDE_SUFFIX:
                continue
            if int(entry.st_size or 0) > MAX_BYTES:
                continue
            if wanted(rel):
                targets.append(rel)

    for name in DIRS:
        walk(posixpath.join(REMOTE_ROOT, name))
    for name in FILES:
        if not wanted(name):
            continue
        try:
            st = sftp.stat(posixpath.join(REMOTE_ROOT, name))
        except IOError:
            continue
        suffix = Path(name).suffix.lower()
        if suffix in KEEP_SUFFIX and int(st.st_size or 0) <= MAX_BYTES:
            targets.append(name)

    print(f"[pull] {USER}@{HOST}:{REMOTE_ROOT}  ->  {STAGE}"
          + (f"   only={only}" if only else "") + ("   (dry-run)" if dry else ""))

    written = unchanged = failed = 0
    for rel in sorted(targets):
        if dry:
            print("  [dry] " + rel)
            continue
        try:
            with sftp.open(posixpath.join(REMOTE_ROOT, rel), "rb") as handle:
                data = handle.read()
            text = data.decode("utf-8")
        except Exception as err:  # noqa: BLE001 - 单文件失败不影响整体
            failed += 1
            print(f"  [FAIL] {rel}: {err}")
            continue
        dst = STAGE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_file():
            # 规范行尾后比较（云端 LF / 本地 CRLF 都视为一致）
            current = dst.read_bytes().decode("utf-8", "ignore")
            if _norm_eol(current) == _norm_eol(text):
                unchanged += 1
                continue
        # 以原始字节落盘：Windows 下 write_text 会把 \n 二次转成 \r\n，
        # 云端 CRLF 文件会变成 \r\r\n（假差异），必须按字节写。
        dst.write_bytes(data)
        written += 1
        print("  [pull] " + rel)

    sftp.close()
    client.close()
    print(f"[done] 拉取/更新={written}  内容一致跳过={unchanged}  失败={failed}"
          + ("   (dry-run)" if dry else ""))


if __name__ == "__main__":
    main()
