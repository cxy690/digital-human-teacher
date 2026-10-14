# digital_human — 数字人子服务（WebRTC 推流，默认关闭）

独立的数字人实时推流服务：MuseTalk 口型渲染 + edge-tts + WebRTC 音视频轨道，把「待机动画 + 逐句口型播报」以 WebRTC 媒体流推给学生端浏览器。与主应用的 `AVATAR_BACKEND`（逐句渲染视频）**正交**——本服务负责「独立托管 + 实时推流」这条路。

启动：`python digital_human/run.py`（默认 `:8010`）；主应用可经 `AVATAR_ENABLE=true` 探活并自动拉起（`core/avatar_launcher.py`）。

## 文件说明

| 文件 / 子目录 | 用途 |
| --- | --- |
| `run.py` | 服务入口。 |
| `config.yaml` / `config.cloud.yaml` | 本机 / 云端配置：素材目录、端口、批大小、fps、预滚时长、发送降采样、模型路径、TTS 音色等（云端模板另配 TURN）。 |
| `fetch_models.sh` | 拉取 MuseTalk / VAE / Whisper 权重。 |
| `genavatar.py` | 离线生成口型素材包（用于预渲染或调试）。 |
| `requirements.txt` | 本服务独立依赖（torch / aiortc / edge-tts 等），**不与主项目 requirements 混装**。 |
| `server/` | HTTP 信令 + 渲染引擎 + WebRTC 轨道（见该目录 README）。 |
| `musetalk/` | MuseTalk 推理核心（第三方代码，随仓库携带，含 `utils/face_detection` 等）。 |

## 设计要点

- **进程隔离**：依赖 CUDA torch，独立解释器/venv（`AVATAR_PYTHON` 指定或自动探测），失败只记日志不影响主应用。
- **幂等拉起**：`.launcher.state` 记录 PID —— 240s 模型加载窗口内不重复拉起，防 `uvicorn --reload` 起出多个实例抢显存。
- **云端必须配 TURN**：AutoDL 等无公网 IP 环境，aiortc 只收集 host 候选浏览器连不上；`RTC_STUN_URL` / `RTC_TURN_URL` 环境变量优先于 yaml。
- **素材即配置**：每个科目一个素材目录（待机动画 + 口型素材同源），教育端按学生所选科目经 `POST /api/avatar` 切换。
