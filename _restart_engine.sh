#!/usr/bin/env bash
# 临时：云端 MuseTalk 引擎重启脚本（多老师模式：按 teacher_id 懒加载学科底片）。
set -u
ROOT=/root/autodl-tmp/education
cd "$ROOT" || exit 1

pkill -f "avatar_engine_musetalk_v1[5]" 2>/dev/null
for _ in $(seq 1 25); do pgrep -f avatar_engine_musetalk_v1[5] >/dev/null 2>&1 || break; sleep 1; done
pkill -9 -f "avatar_engine_musetalk_v1[5]" 2>/dev/null
sleep 2

MUSETALK_DIR="$ROOT/models_zoo/MuseTalk" \
ENGINE_PORT=8012 ENGINE_HOST=127.0.0.1 \
AVATAR_TEACHERS_DIR="$ROOT/data/teachers" \
AVATAR_LEGACY_PRELOAD=0 \
AVATAR_CLIP_DIR="$ROOT/data/teachers" \
MUSETALK_FP16=1 \
setsid nohup "$ROOT/.venvs/musetalk/bin/python" scripts/avatar_engine_musetalk_v15.py \
  >> deploy/cloud/logs/musetalk.log 2>&1 < /dev/null &

echo $! > logs/musetalk_engine.pid
echo "restarted pid $(cat logs/musetalk_engine.pid)"
