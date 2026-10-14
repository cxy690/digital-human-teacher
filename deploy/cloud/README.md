# deploy/cloud — 算力云全栈部署

在 GPU 算力云（Linux）上一键部署「主应用 + Qwen3/LLM + FunASR + CosyVoice + MuseTalk」全栈的脚本集。端口约定：主应用 8001、vLLM 8100、FunASR 10095、CosyVoice 8090、MuseTalk 8012。

## 脚本说明

| 脚本 | 用途 |
| --- | --- |
| `check_gpu.sh` | GPU 自检（显卡 / 驱动 / 显存）。 |
| `setup.sh` | 基础环境搭建：系统依赖 + 各引擎独立 venv。 |
| `download_models.sh` | 克隆模型仓库 + 下载权重。 |
| `start_all.sh` | 一键启动全栈（按依赖序拉起各引擎与主应用）。 |
| `stop_all.sh` | 停止全部服务。 |
| `pause.sh` | 一键停止（停服务 + 停公网隧道），之后可放心「关机/停止」（切勿点「释放」）。 |
| `restart_app.sh` | 只重启主应用（不动引擎）；扫 `/proc/*/cmdline` 精确匹配，避免「旧进程占端口、新进程退出、改动不生效」。 |
| `restart_tts.sh` | 只重启 CosyVoice TTS 引擎（2/3 通用，fp16 关闭）。 |
| `resume.sh` | 关机后一键恢复（含 cloudflared 隧道，输出公网链接）。 |
| `switch_mode.sh` | 切换答疑大脑：`online`（API）/ `offline`（本地模型），对应 `mode_online.env` / `mode_offline.env`。 |
| `env.cloud.example` | 云端环境变量模板（另含 `BOARD_FIRST_MODE=single_call` 等云端推荐值）。 |
| `mode_online.env` / `mode_offline.env` | 两种大脑模式的配置片段。`mode_online.env` 含明文 `LLM_API_KEY`，已被 `.gitignore` 忽略；首次使用请 `cp mode_online.env.example mode_online.env` 并填入自己的密钥。 |
| `constraints.txt` | 依赖版本约束。 |
| `_bashrc_hook.sh` | shell 环境钩子（内部使用）。 |

## 设计要点

- **每引擎独立 venv / 独立进程**：torch 版本互不干扰（`.venvs/musetalk`、`.venvs/cosy` 等），按显存与需求单独启停。
- **重启的正确姿势**：镜像里没有 `fuser`、`ss` 反查不可靠、`pkill -f server.app` 会误杀调用方 shell——统一改为扫 `/proc/*/cmdline` 精确匹配，脚本注释里保留了这套踩坑记录。
- **隧道与关机流程固化**：`pause.sh` 先停隧道再停服务，`resume.sh` 恢复后打印公网链接；避免直接关机丢隧道或误点释放。
