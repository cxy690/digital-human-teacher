# -*- coding: utf-8 -*-
"""算力云 MuseTalk 引擎控制（通过 paramiko SSH 在云机执行）。

用法：
    python scripts/cloud_ctl.py restart   # 重启云端引擎（bash _restart_engine.sh）
    python scripts/cloud_ctl.py stop      # 停止云端引擎
    python scripts/cloud_ctl.py status    # 查看云端引擎进程与健康
    python scripts/cloud_ctl.py logs      # 把云端 logs/ 增量拉回本地 logs/cloud/
                                          # （restart / restart_app / restart_all 之后会自动拉一次）

快照与撤回（云端不是 git 仓库，靠 tar 快照实现「一键撤回」）：
    python scripts/cloud_ctl.py snapshot            # 给当前云端代码打一份快照
    python scripts/cloud_ctl.py snapshots           # 列出已有快照
    python scripts/cloud_ctl.py rollback            # 一键撤回「最近一次快照」
    python scripts/cloud_ctl.py rollback <快照名>    # 撤回指定快照（如 snap_20260928_153000.tgz）

连接信息可用环境变量覆盖：
    CLOUD_HOST / CLOUD_PORT / CLOUD_USER / CLOUD_PWD / CLOUD_ROOT
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import paramiko

try:  # Windows 控制台编码：远端输出可能含 GBK 之外的字符（如题目里的 ²），统一按 UTF-8 输出
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


def _cloud_env(key: str, default: str = "") -> str:
    """读云端连接配置：环境变量优先，其次项目根 ``.env``（.env 不入库）。

    凭据**不写进源码** —— 仓库要推 Git，源码里带密码等于把钥匙贴在门上。
    """
    value = os.getenv(key)
    if value:
        return value
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith(key + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return default


# 云主机地址/端口属私有运维信息，不写入源码：CLOUD_* 优先，其次 TUNNEL_SSH_*
HOST = _cloud_env("CLOUD_HOST") or _cloud_env("TUNNEL_SSH_HOST")
PORT = int(_cloud_env("CLOUD_PORT") or _cloud_env("TUNNEL_SSH_PORT") or "22")
USER = _cloud_env("CLOUD_USER", "root")
PWD = _cloud_env("CLOUD_PWD")
ROOT = _cloud_env("CLOUD_ROOT", "/root/autodl-tmp/education")
SNAP_ROOT = os.getenv("CLOUD_SNAP_ROOT", "/root/autodl-tmp/_snapshots")

# 参与快照 / 回滚的代码目录。与 sync_cloud.py 的上传白名单保持一致，
# 但**不含 data/**（底片与课件是几十 MB 的媒体，打进快照会让每次快照都很慢）——
# 除了 data/teachers/*/teacher.json 这一份「形象标定」文本，它必须能一起撤回，
# 否则「换了形象又撤回」会留下一个指向旧视频的档案。
CODE_DIRS = [
    ".agents",
    "agents", "clients", "config", "core", "document_parsing", "mcp_tools",
    "models", "rag", "scripts", "server", "tests", "db", "memory",
    "digital_human", "deploy",
]
CODE_FILES = ["_restart_engine.sh", "pyproject.toml", "requirements.txt"]
SNAP_TARGETS = " ".join([*CODE_DIRS, *CODE_FILES, "data/teachers/*/teacher.json"])

# ---------------------------------------------------------------- 快照 / 撤回
SNAPSHOT_CMD = (
    f"cd {ROOT} && mkdir -p {SNAP_ROOT} && TS=$(date +%Y%m%d_%H%M%S) && "
    f"tar czf {SNAP_ROOT}/snap_$TS.tgz {SNAP_TARGETS} && "
    f'F={SNAP_ROOT}/snap_$TS.tgz && echo "快照完成：$F" && du -h "$F" && '
    f"ls -t {SNAP_ROOT}/snap_*.tgz | wc -l | xargs echo '现有快照数：'"
)

LIST_SNAP_CMD = (
    f"ls -lht {SNAP_ROOT}/snap_*.tgz 2>/dev/null | head -10; "
    f"echo '--- 含自动备份 ---'; "
    f"ls {SNAP_ROOT}/pre_rollback_*.tgz 2>/dev/null | wc -l | xargs echo 'pre_rollback 份数：'"
)


def rollback_cmd(name: str = "") -> str:
    """生成回滚命令：不传名字就撤回到最近一次快照。

    回滚前会**先给当前状态再打一份 pre_rollback 快照** —— 撤回本身也可能是误操作，
    留一份现场才能再退回去。校验 tar 完整性也在这里做，避免解压半个包把代码目录删了。
    """
    pick = f"{SNAP_ROOT}/{name}" if name else f"$(ls -t {SNAP_ROOT}/snap_*.tgz 2>/dev/null | head -1)"
    return (
        f"cd {ROOT} || exit 1; "
        f'SNAP="{pick}"; '
        f'if [ ! -f "$SNAP" ]; then echo "没有可用快照：$SNAP"; exit 1; fi; '
        f'if ! tar tzf "$SNAP" >/dev/null 2>&1; then echo "快照已损坏，中止：$SNAP"; exit 1; fi; '
        f"mkdir -p {SNAP_ROOT}; "
        f"tar czf {SNAP_ROOT}/pre_rollback_$(date +%Y%m%d_%H%M%S).tgz {SNAP_TARGETS} 2>/dev/null; "
        f"rm -rf {' '.join(CODE_DIRS)}; "
        f'tar xzf "$SNAP" -C {ROOT} && echo "已撤回：$SNAP"; '
        f"echo '提示：回滚后要重启才生效 -> python scripts/cloud_ctl.py restart_all'"
    )

ACTIONS: dict[str, str] = {
    # 只重启 MuseTalk 引擎（改了 scripts/avatar_engine_* 或动了底片时用）
    "restart": f"cd {ROOT} && bash _restart_engine.sh",
    # 只重启主应用（改了 config/ core/ server/ clients/ 时用，不动已加载的模型）
    "restart_app": f"cd {ROOT} && bash deploy/cloud/restart_app.sh",
    # 引擎 + 主应用 + 本地 embedding 服务（RAG 向量检索）都重启；
    # embedding 起不来只提示（RAG 自动退回词法检索），不阻断前两者。
    "restart_all": (f"cd {ROOT} && bash _restart_engine.sh && bash deploy/cloud/restart_app.sh; "
                    f"cd {ROOT} && (bash deploy/cloud/restart_embedding.sh "
                    f"|| echo '[warn] embedding 服务未启动（RAG 退回词法检索，不影响答疑）')"),
    "stop": "pkill -f avatar_engine_musetalk_v15; echo engine-stopped",
    "status": "pgrep -f avatar_engine_musetalk_v15; "
              "curl -s -m 5 http://127.0.0.1:8012/health; echo",
    "health": "echo -n 'cloud app: '; curl -s -m 5 http://127.0.0.1:8001/health; echo; "
              "echo -n 'engine   : '; curl -s -m 5 http://127.0.0.1:8012/health; echo; "
              "echo -n 'embedding: '; curl -s -m 5 http://127.0.0.1:8101/health; echo",
    # RAG 向量检索自检：embedding 服务健康 + 索引元信息 + 应用侧最近一次后端选择
    "rag": (f"echo -n 'embedding: '; curl -s -m 5 http://127.0.0.1:8101/health || echo '(未启动)'; echo; "
            f"echo '--- 索引 meta ---'; cat {ROOT}/data/raw/rag_index/meta.json 2>/dev/null || echo '(未构建：先在云端运行 python scripts/build_rag_index.py)'; echo; "
            f"echo '--- 索引文件 ---'; ls -lh {ROOT}/data/raw/rag_index 2>/dev/null; echo; "
            f"echo '--- 应用侧最近 RAG 日志 ---'; grep -ai rag {ROOT}/logs/app.log 2>/dev/null | tail -6"),
    # 公网地址（cpolar 免费版域名每次重启会变，一律以日志里的新地址为准）
    "public": f"grep -hoE 'https://[A-Za-z0-9.-]+\\.(cpolar\\.[a-z]+|top|io|cn)' "
              f"{ROOT}/logs/cpolar.log 2>/dev/null | tail -1; "
              f"pgrep -f 'cpolar http' >/dev/null && echo '(cpolar running)' "
              f"|| echo '(cpolar NOT running)'",
    # 远端跑任意命令：python scripts/cloud_ctl.py exec "ls -la"
    "exec": "",
    # 快照 / 撤回（撤回基准）：改代码前先 snapshot，不满意就 rollback
    "snapshot": SNAPSHOT_CMD,
    "snapshots": LIST_SNAP_CMD,
}


def run_remote(cmd: str, timeout: int = 180) -> str:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30)
    _, out, err = c.exec_command(cmd, timeout=timeout)
    text = out.read().decode("utf-8", "ignore") + err.read().decode("utf-8", "ignore")
    c.close()
    return text


def _auto_pull_logs() -> None:
    """把云端日志增量拉回本地（``logs/cloud/``）；失败只提示，不影响主流程。"""
    try:
        try:
            from pull_logs import pull_logs
        except ImportError:  # 作为包运行（python -m scripts.cloud_ctl）
            from scripts.pull_logs import pull_logs

        downloaded, skipped = pull_logs(quiet=True)
        print(f"[logs] 已同步云端日志到 logs/cloud/：新增 {downloaded} 个，跳过 {skipped} 个（未变化）")
    except Exception as err:  # noqa: BLE001 - 拉日志是顺手的事，不该影响重启
        print(f"[logs] 自动拉取云端日志失败（不影响服务）：{err}")


def main() -> None:
    argv = sys.argv[1:]
    action = (argv[0] if argv else "status").lower()
    if action == "logs":
        try:
            from pull_logs import pull_logs
        except ImportError:
            from scripts.pull_logs import pull_logs

        force = "--force" in argv or "--all" in argv
        downloaded, skipped = pull_logs(force=force)
        print(f"[logs] 已同步到 logs/cloud/：新增 {downloaded} 个，跳过 {skipped} 个")
        return
    if action == "exec":
        cmd = " ".join(argv[1:]).strip()
        if not cmd:
            print('usage: python scripts/cloud_ctl.py exec "<shell command>"', file=sys.stderr)
            sys.exit(2)
    elif action == "rollback":
        # 可带快照名（snap_xxx.tgz）；不带则撤回到最近一次快照
        cmd = rollback_cmd(argv[1].strip() if len(argv) > 1 else "")
    else:
        cmd = ACTIONS.get(action)
        if cmd is None:
            print(f"unknown action: {action} "
                  f"(use restart|restart_app|restart_all|stop|status|health|public|rag|"
                  f"snapshot|snapshots|rollback|exec)",
                  file=sys.stderr)
            sys.exit(2)
    # 打包/解包比查健康慢，给快照与撤回更长的超时
    timeout = 600 if action in ("snapshot", "rollback") else 180
    print(run_remote(cmd, timeout=timeout))
    # 重启类动作之后，把云端日志自动拉回本地 —— 这样「改代码 → 同步 → 重启 → 看日志」
    # 闭环里最后一步不用手动做，本地 logs/cloud/ 里就是刚才那台机器上的现场。
    if action in ("restart", "restart_app", "restart_all"):
        _auto_pull_logs()


if __name__ == "__main__":
    main()
