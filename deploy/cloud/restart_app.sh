#!/usr/bin/env bash
# 只重启「主应用」（不动 MuseTalk / CosyVoice 引擎）。
#
# 踩过的坑（务必保留这套写法）：
#   * `fuser` 在算力云镜像里没有 -> `fuser -k 8001/tcp` 静默失败；
#   * `ss -lntp | grep :8001` 在部分环境下拿不到该进程 -> 按端口反查也失败；
#   * `pkill -f server.app` 会匹配到「执行这条命令的 shell」自身 -> 把自己杀掉。
# 结果就是：旧进程一直占着端口，新进程绑定失败立刻退出，界面表现为
# 「改了代码/配置却不生效」。这里改为直接扫描 /proc/*/cmdline 精确匹配，
# 并且只杀「python -m server.app」这一类进程，不误伤调用方 shell。
set -u
ROOT=/root/autodl-tmp/education
cd "$ROOT" || exit 1
LOG="$ROOT/deploy/cloud/logs/app.log"
mkdir -p "$(dirname "$LOG")"
PATTERN=".venvs/app/bin/python -m server.app"

find_pids() {
  local p cmd
  for p in /proc/[0-9]*; do
    p="${p#/proc/}"
    [ "$p" = "$$" ] && continue
    [ "$p" = "${PPID:-0}" ] && continue
    cmd="$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || true)"
    case "$cmd" in
      *"$PATTERN"*) echo "$p" ;;
    esac
  done
}

OLD="$(find_pids | sort -u | tr '\n' ' ')"
if [ -n "${OLD// /}" ]; then
  echo "[restart] 停止旧应用进程：$OLD"
  # shellcheck disable=SC2086
  kill -9 $OLD 2>/dev/null || true
fi
for _ in $(seq 1 15); do
  [ -z "$(find_pids)" ] && break
  sleep 1
done

setsid nohup "$ROOT/.venvs/app/bin/python" -m server.app >"$LOG" 2>&1 </dev/null &
sleep 2
NEW="$(find_pids | sort -u | tr '\n' ' ')"
echo "[restart] 新进程：$NEW"

for _ in $(seq 1 30); do
  curl -sf -m 4 http://127.0.0.1:8001/health >/dev/null 2>&1 && break
  sleep 2
done
echo "[restart] health: $(curl -s -m 6 http://127.0.0.1:8001/health)"
