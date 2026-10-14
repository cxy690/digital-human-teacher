# deploy — 部署

容器化部署（Docker）与算力云全栈部署脚本。

## 文件说明

| 文件 / 子目录 | 用途 |
| --- | --- |
| `Dockerfile` | 基于 `python:3.11-slim`：安装 `libglib2.0-0 libgl1 ffmpeg`，`pip install .` 装包，暴露 8000，`uvicorn server.app:app` 启动。 |
| `docker-compose.yml` | 单服务 `education`：读根目录 `.env`，挂载 `../data`，映射 8000，`restart: unless-stopped`；环境变量强制 `ENV=prod DEBUG=false`。 |
| `cloud/` | 算力云（Linux + GPU）全栈部署脚本与配置（见该目录 README）。 |

## 使用

```bash
# 容器部署（项目根目录执行）
docker compose -f deploy/docker-compose.yml up --build

# 算力云全栈（详见 cloud/README.md）
bash deploy/cloud/check_gpu.sh && bash deploy/cloud/setup.sh && bash deploy/cloud/start_all.sh
```

## 设计要点

- **镜像只装主链路**：MuseTalk / CosyVoice / FunASR 等 GPU 引擎**不装进主镜像**（依赖 torch 与独立 venv），云端由 `cloud/` 脚本以独立进程托管——避免「requirements 里重复装 torch 被覆盖成 CPU 版」。
- **配置外置**：所有配置经 `.env` 注入，镜像内不含任何密钥；数据目录（`data/`）挂载持久化。
- **纯 Mock 可跑**：不配密钥的容器开箱即起（教学链路 Mock 降级），适合冒烟与演练。
