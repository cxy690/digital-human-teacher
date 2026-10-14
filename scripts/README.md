# scripts — 引擎与辅助脚本

独立进程的引擎服务、一次性工具与运维脚本。全部支持在项目根目录执行（`python -m scripts.<名>` 或 `python scripts/<名>.py`）。

## 引擎服务（独立进程 / 独立环境）

| 脚本 | 用途 |
| --- | --- |
| `llm_engine_local.py` | 本地 LLM 推理服务（Qwen3-8B-AWQ，transformers+AWQ），OpenAI 兼容 `POST /v1/chat/completions`（含 SSE 流式），默认 :8100。 |
| `asr_engine_funasr.py` | FunASR 语音识别引擎（SenseVoiceSmall + fsmn-vad，支持热词），`POST /recognize`，默认 127.0.0.1:10095；无 GPU 自动回落 CPU。 |
| `tts_engine_cosyvoice.py` | CosyVoice TTS 引擎（兼容 1/2/3，零样本克隆），`POST /api/v1/tts`，默认 127.0.0.1:8090。 |
| `avatar_engine_musetalk_v15.py` | MuseTalk v1.5 口型引擎（MediaPipe 求人脸框，绕开 mmcv 编译；多底片懒加载），:8012。 |
| `avatar_adapter.py` | 数字人渲染适配器参考实现：把统一 `POST /render` 契约转发到任意上游引擎；无 GPU 可配 `AVATAR_STUB_VIDEO` 先跑通链路。 |

## 一键启动 / 云端运维

| 脚本 | 用途 |
| --- | --- |
| `start_local.ps1` / `start_local.bat` | 本机一键启动（`-Engine musetalk\|wav2lip` + 学生端），自动同步 `.env` 的 `AVATAR_BACKEND`。 |
| `start_all.ps1` | GPU 机器一键启动（Wav2Lip :8011 + 学生端），打印局域网 IP 供同学访问。 |
| `start_cloud.ps1` | 云端架构一键启停（云端引擎 → 隧道 → 本机主应用），支持 `-Stop/-ForceRestart/-NoEngine/-NoTts/-NoApp`。 |
| `sync_cloud.py` / `sync_cloud.ps1` | 本地 → 算力云**增量上传**（白名单、只增不删、跳过缓存与运行时数据）；`.ps1` 为包装。 |
| `pull_cloud.py` | 云端 → 本地增量**只读**拉取到 `_cloud_pull/` 暂存区（绝不覆盖工作树），便于 diff 挑选合并。 |
| `check_sync.py` | 本地与云端文件差异体检（关机前防丢改动）。 |
| `cloud_ctl.py` | SSH 远程控制云端引擎/TTS（restart / stop / status）。 |
| `db_reverse_tunnel.py` | 数据库反向隧道：云端应用访问内网 MySQL / Milvus（每个映射独立 SSH 连接 + 断线自动重连）。 |

## 自检 / 演示 / 开发辅助

| 脚本 | 用途 |
| --- | --- |
| `run_demo.py` | **离线演示**：无密钥无服务，跑通「路由 → 答疑 → 麦控」全链路并把事件打印到控制台。 |
| `selfcheck.py` | 8 步项目自检：学科 / 教师档案 / 技能模块 / 关键导入 / 节点地图 / Agent 自主性。 |
| `smoke_courseware.py` | 课件端到端冒烟（清单 → 页图 → WS 下发 `slide`+`slide_state`+标注）。 |
| `verify_stream_board.py` | 白板「边流边分批上板」一致性验证（假 LLM 流模拟真实输出）。 |
| `check_frontend_js.mjs` | 前端体检：`clients/*/index.html` 内联 JS 语法解析 + HTML 注释配对。 |
| `render_agent_flow.py` | 渲染 Agent 总览图 `docs/agent_flow.png`（节点清单取自 `agents/nodes.py`）。 |
| `seed_data.py` | 数据目录种子（确保 `data/` 各子目录齐全）。 |
| `migrate_teacher_skills_to_md.py` | **一次性**迁移：旧 Python 技能合并为 `SKILL.md`（已执行完毕，日常直接改 md）。 |

## 教师素材（换人 / 加变体）

| 脚本 | 用途 |
| --- | --- |
| `add_teacher_variant.py` | 新增同科风格变体老师：底片归一化 + 抽干声 + 待机帧标定 + 写档案，一步完成。 |
| `extract_teacher_voices.py` | 从 `talk.mp4` 抽干净干声 `voice_ref.wav`；`--asr` 生成逐字对应的 `ref_text`（克隆必需）。 |
| `apply_edge_voices.py` | 批量校正 edge 音色（性别匹配、同科错开、剔除实测不存在的方言音色）。 |
| `apply_variant_voices.py` | 给变体老师接克隆音色（复用同学科主老师干声），支持 `--dry` / `--revert`。 |
| `proc_idle_replace.py` | 待机动画换源加工：水印修复 / 取景 / 眨眼锚点选段 / 人脸区域接缝判定的无缝循环。 |
| `export_pptx_images.ps1` | 用本机 PowerPoint COM 把 `.pptx` 导出为 `p{i}.png` 页图（`data/courseware/<deck_id>/`）。 |

## 排障

| 脚本 | 用途 |
| --- | --- |
| `view_activity.py` | 按 `run_id` 还原图执行链路（含「进行中 / 中断」判定），排障「哪一步卡住」。 |
| `clear_cache.py` | 清理数字人人脸预处理缓存（`--version` / `--dry-run` / `--dir`）。 |

## 设计要点

- **引擎独立环境**：torch 系引擎（MuseTalk / CosyVoice / FunASR / 本地 LLM）各自 venv、独立进程，主项目 requirements 不装 torch（防被覆盖成 CPU 版）。
- **同步只增不删**：`sync_cloud.py` 按白名单只上传、不做远端删除；运行时产物（经验 / 记忆 / 缓存）与环境文件（云端 `.env`）均排除。
- **一次性脚本标识**：迁移类脚本（`migrate_teacher_skills_to_md.py`）执行完成后保留作为历史记录，重跑需谨慎。
