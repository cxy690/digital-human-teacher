# -*- coding: utf-8 -*-
"""一键把本地项目同步到算力云（增量上传，只增不删）。

用法：
    python scripts/sync_cloud.py            # 增量同步（只传有变化的文件）
    python scripts/sync_cloud.py --dry      # 只预览将上传哪些文件，不实际传
    python scripts/sync_cloud.py --force    # 忽略差异，全部重传

安全的默认行为：
    * 只上传白名单目录/文件，**不会删除云端任何文件**；
    * 跳过 __pycache__ / *.pyc / .git / .idea / .venvs / 日志 / 备份；
    * 跳过 data/cache、data/raw（运行时缓存/会话数据）与 .env（云端专用配置）；
    * **云端较新的文件默认不动**（打印 [SAFE] 跳过）：云端文件可能被「技能管理
      平台」在线编辑过（保存即写云端、mtime 变新），静默覆盖会丢编辑（2026-10-12
      事故）。确认要以本地覆盖时显式加 --force，或先用 pull_cloud.py 拉回对比。

连接信息可用环境变量覆盖：CLOUD_HOST / CLOUD_PORT / CLOUD_USER / CLOUD_PWD / CLOUD_ROOT
未设置 CLOUD_* 时回退 TUNNEL_SSH_*（与 _tunnel.py / cloud_ctl.py 同一套凭据）——
AutoDL 实例重启/迁移后 SSH 入口会变，只需 setx 更新 TUNNEL_SSH_* 一处。
"""
from __future__ import annotations

import os
import posixpath
import sys
from pathlib import Path

import paramiko

# 云主机地址/端口属私有运维信息，不写入源码：CLOUD_* 优先，其次 TUNNEL_SSH_*
HOST = os.getenv("CLOUD_HOST") or os.getenv("TUNNEL_SSH_HOST", "")
PORT = int(os.getenv("CLOUD_PORT") or os.getenv("TUNNEL_SSH_PORT") or "22")
USER = os.getenv("CLOUD_USER") or os.getenv("TUNNEL_SSH_USER", "root")
PWD = os.getenv("CLOUD_PWD") or os.getenv("TUNNEL_SSH_PASSWORD", "")
REMOTE_ROOT = os.getenv("CLOUD_ROOT", "/root/autodl-tmp/education")

LOCAL_ROOT = Path(__file__).resolve().parent.parent

# 要同步的目录 / 单文件（相对项目根）
INCLUDE_DIRS = [
    # 数字人教师技能事实源（.agents/skills/<学科>/SKILL.md）与维护笔记。
    # 云端主应用直接读这些 md 构建提示词，漏同步会让云端老师「性格回到旧版」。
    ".agents",
    "a2a",
    "agents", "clients", "config", "core", "document_parsing", "mcp_tools",
    "models", "rag", "scripts", "server", "tests", "data",
    # 合并新增（education_fusion）：数据层、测试、数字人子服务、部署脚本
    "db", "memory", "digital_human", "deploy",
    # 技能规划等文档：之前漏了 docs/，本地改了规划文档云端看不到
    "docs",
]
# 根目录单文件
INCLUDE_FILES = [
    "_restart_engine.sh", "pyproject.toml", "requirements.txt", "README.md", "代码地图.md",
    # 根目录的其余文档也一起同步：之前只同步上面 5 个，导致云端的部署/规格文档长期停在旧版
    # （排查问题时照着云端文档看，会看到已经改掉的行为）。
    "算力云部署.md", "数字人素材规格.md", "数字人部署指南.md", "本地开发联调.md",
    "结构对照审计.md", "待补齐规划.md", "接口实现状态.md",
    # 服务器环境/迁移清单：云端也留一份，换机或排查时不用回头找本地
    # （之前漏了它，云端一直看不到这份「唯一事实清单」）。
    "服务器环境与迁移清单.md",
]

#: 文档在**本地**的实际存放目录。
#:
#: ⚠ 这里是历史上一个静默失效点：``INCLUDE_FILES`` 里写的是「根目录同名文件」，
#: 而本地这些文档早已全部归到 ``md_docs/`` 下 —— 于是 ``LOCAL_ROOT/代码地图.md``
#: 不存在、迭代时被静默跳过，云端那份文档从此再没被更新过（照它排查会看到已改掉的行为）。
#: 现在改为：本地根目录找不到时**回落到 md_docs/**，但上传路径仍是云端根目录同名文件，
#: 保持云端原有的文档位置不变（不产生 md_docs/ + 根目录两份副本）。
DOC_DIR = "md_docs"

# 排除：目录名命中任一段即跳过
EXCLUDE_DIRS = {
    "__pycache__", ".git", ".idea", ".venv", ".venvs", "venvs", "node_modules",
    ".playwright-cli", "cache", "raw", "logs", "_backup_20260921",
}
# 排除：扩展名
EXCLUDE_SUFFIX = (".pyc", ".pyo", ".log", ".bak")
# 排除：相对路径（精确 / 前缀）
EXCLUDE_REL = {
    "data/cache", "data/raw", "data/avatar_stream", ".env", "education.db",
    # 教师经验（Skill 自我更新的产物）：**按环境隔离**，同步过去会覆盖掉云端线上
    # 课堂里学到的经验（反之亦然）。要人工搬运时单独拷这一个目录即可。
    "data/teacher_experience",
    # Agent 技能战绩（技能调用收益）：同理，本地测试课的调用记录不该覆盖云端真实战绩。
    "data/skill_usage",
    # 学情记忆（长期记忆，按学生分文件）：两条理由都必须排除 ——
    # ① 与经验同理，按环境隔离：本地测试课的记忆不该混进云端真实学生的画像；
    # ② **它是未成年人的个人数据**，随代码同步扩散到另一台机器属于数据合规风险。
    # 要人工搬运时单独按学生 id 拷那一个 jsonl 文件。
    "data/memory",
    # digital_human 的模型权重（约 4GB）与 vendored musetalk：云端主链路用的是
    # scripts/avatar_engine_musetalk_v15.py + models_zoo/MuseTalk，不需要这两坨。
    "digital_human/models", "digital_human/musetalk",
}


#: 覆盖保护：差异文件小于该体积时，先取云端内容做「行尾归一」比对，
#: 以便区分「仅 mtime 漂移（内容一致）」与「两边内容真的不同」。
_PROBE_MAX_BYTES = 2 * 1024 * 1024
#: 可做内容比对的文本类扩展名（其余按二进制处理，只走 mtime 规则）。
_TEXT_EXTS = {
    ".py", ".md", ".txt", ".json", ".jsonl", ".html", ".css", ".js", ".mjs",
    ".sh", ".ps1", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".csv", ".example",
}


def _norm_eol(text: str) -> str:
    """统一行尾（CRLF / CR -> LF）后再比较，避免跨平台假差异。"""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def iter_local_files():
    """产出 (本地绝对路径, 相对项目根的 posix 路径)。

    ``--only <相对路径前缀>`` 可只同步指定文件/目录，用于改一两个文件时快速上传
    （全量比对要对每个文件做一次远端 stat，几百个文件时会明显变慢）。
    """
    for name in INCLUDE_DIRS:
        base = LOCAL_ROOT / name
        if not base.is_dir():
            continue
        for fp in sorted(base.rglob("*")):
            if not fp.is_file() or fp.is_symlink():
                continue
            rel = fp.relative_to(LOCAL_ROOT).as_posix()
            if set(rel.split("/")) & EXCLUDE_DIRS:
                continue
            if fp.suffix in EXCLUDE_SUFFIX:
                continue
            if rel in EXCLUDE_REL or any(rel.startswith(x + "/") for x in EXCLUDE_REL):
                continue
            yield fp, rel
    for name in INCLUDE_FILES:
        fp = LOCAL_ROOT / name
        if not fp.is_file():
            # 本地根目录没有 -> 到 md_docs/ 下取（上传到云端仍用 name，即根目录路径）
            fp = LOCAL_ROOT / DOC_DIR / name
        if fp.is_file():
            yield fp, name


def main() -> None:
    if not PWD:
        print('缺少 SSH 密码：先 setx TUNNEL_SSH_PASSWORD "密码"（或设置 CLOUD_PWD）并重开终端',
              file=sys.stderr)
        sys.exit(2)
    argv = sys.argv[1:]
    dry = "--dry" in argv
    force = "--force" in argv
    only = ""
    if "--only" in argv:
        idx = argv.index("--only")
        if idx + 1 < len(argv):
            only = argv[idx + 1].strip().strip("/")

    def _wanted(rel: str) -> bool:
        return not only or rel == only or rel.startswith(only + "/")

    print(f"[sync] {LOCAL_ROOT}  ->  {USER}@{HOST}:{REMOTE_ROOT}   "
          f"mode={'DRY' if dry else ('FORCE' if force else 'INCREMENTAL')}"
          + (f"   only={only}" if only else ""))

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30)
    sftp = c.open_sftp()

    made_dirs: set[str] = set()

    def ensure_dir(remote_dir: str) -> None:
        cur = ""
        for part in remote_dir.split("/"):
            if not part:
                continue
            cur += "/" + part
            if cur in made_dirs:
                continue
            try:
                sftp.stat(cur)
            except IOError:
                try:
                    sftp.mkdir(cur)
                except IOError:
                    pass
            made_dirs.add(cur)

    uploaded = skipped = blocked = failed = 0
    for fp, rel in iter_local_files():
        if not _wanted(rel):
            continue
        remote = posixpath.join(REMOTE_ROOT, rel)
        st = fp.stat()
        attr = None
        try:
            attr = sftp.stat(remote)
        except IOError:
            attr = None
        # 远端已存在且 大小+修改时间 一致 -> 跳过
        if (not force and attr is not None
                and attr.st_size == st.st_size
                and abs(attr.st_mtime - st.st_mtime) <= 1):
            skipped += 1
            continue

        # ---- 覆盖保护（2026-10-12 事故后加）--------------------------------
        # 云端文件可能被「技能管理平台」在线编辑过（保存即写云端、mtime 变新）。
        # 只按 size/mtime 判「有差异」就直接上传，会静默丢掉那次编辑。规则：
        #   1) 文本文件取云端内容做行尾归一比对：内容其实相同（仅 mtime 漂移）→
        #      不上传，只对齐 mtime（省流量，也避免无谓写入）；
        #   2) 内容确实不同且**云端 mtime 更新** → 判定「云端有更新」，默认拦下并
        #      提示（人工 pull_cloud 拉回对比；确认要以本地覆盖时加 --force）；
        #   3) 其余（本地更新 / 云端缺失）→ 照常上传。
        if not force and attr is not None:
            remote_text = None
            if attr.st_size <= _PROBE_MAX_BYTES and fp.suffix.lower() in _TEXT_EXTS:
                try:
                    with sftp.open(remote, "rb") as rh:
                        remote_text = rh.read().decode("utf-8", "ignore")
                    local_text = fp.read_bytes().decode("utf-8", "ignore")
                except Exception:  # noqa: BLE001 - 读不动就退回 mtime 规则
                    remote_text = None
            if remote_text is not None and _norm_eol(local_text) == _norm_eol(remote_text):
                try:  # 内容一致（仅 mtime 漂移）：只对齐 mtime，下次即可跳过
                    sftp.utime(remote, (int(st.st_atime), int(st.st_mtime)))
                except IOError:
                    pass
                skipped += 1
                continue
            if attr.st_mtime > st.st_mtime + 2:
                blocked += 1
                print("  [SAFE] " + rel + "  <- 云端较新（可能被在线编辑过），已跳过；"
                      "确认要以本地覆盖请加 --force，或先 pull_cloud 拉回对比")
                continue
        if dry:
            print("  [dry] " + rel)
            uploaded += 1
            continue
        try:
            ensure_dir(posixpath.dirname(remote))
            sftp.put(str(fp), remote)
            try:  # 同步 mtime，便于下次增量判断
                sftp.utime(remote, (int(st.st_atime), int(st.st_mtime)))
            except IOError:
                pass
            uploaded += 1
            print("  [up]  " + rel)
        except Exception as err:  # noqa: BLE001
            failed += 1
            print(f"  [FAIL] {rel}: {err}")

    sftp.close()
    c.close()
    print(f"[done] uploaded={uploaded} skipped={skipped} blocked={blocked} failed={failed}"
          + ("   (dry-run)" if dry else ""))


if __name__ == "__main__":
    main()
