#!/bin/bash
# 重启本地 embedding 服务（RAG 向量检索用；OpenAI 兼容 /v1/embeddings @ 8101）。
# 依赖环境：.venvs/embed（torch CPU + sentence-transformers），装法见 scripts/embedding_server.py 头部注释。
# 模型下载走 hf-mirror 镜像，缓存到持久盘（系统盘重置也不用重下）。
set -u
cd "$(dirname "$0")/../.." || exit 1

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
PORT="${EMBED_PORT:-8101}"
MODEL="${EMBED_MODEL:-BAAI/bge-small-zh-v1.5}"
mkdir -p logs

if [ ! -x .venvs/embed/bin/python ]; then
  echo "[embed] 缺少 .venvs/embed 环境，请先安装："
  echo "  python3 -m venv .venvs/embed"
  echo "  .venvs/embed/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu"
  echo "  .venvs/embed/bin/pip install sentence-transformers"
  exit 1
fi

pkill -f 'scripts/embedding_server.py' 2>/dev/null
sleep 1
nohup .venvs/embed/bin/python scripts/embedding_server.py --port "$PORT" --model "$MODEL" >> logs/embedding_server.log 2>&1 &
echo "[embed] 新进程：$!"

for i in $(seq 1 60); do
  if curl -s -m 2 "http://127.0.0.1:${PORT}/health" >/dev/null; then
    echo -n "[embed] health: "
    curl -s "http://127.0.0.1:${PORT}/health"
    echo
    exit 0
  fi
  sleep 1
done
echo "[embed] 启动超时（首次启动会下载模型，可看 logs/embedding_server.log 或稍后重试）"
exit 1
