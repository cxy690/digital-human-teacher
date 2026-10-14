# digital_human/server — 数字人服务内部实现

数字人子服务的四件套：HTTP/信令（aiohttp）+ 渲染引擎（MuseTalk 服务化）+ TTS + WebRTC 输出轨道。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `__init__.py` | 包声明。 |
| `app.py` | HTTP 与信令：`POST /offer`（WebRTC 信令）、`POST /api/echo`（播报文本，可带 voice/expression/avatar）、`POST /api/avatar`（切换激活素材）、`GET /api/status`（`speaking`=有播报任务，`playing`=口型就绪真正开播）、`POST /api/interrupt`。 |
| `engine.py` | 渲染引擎 `DigitalHumanEngine`：`_worker` 线程消费播报任务、空闲循环待机帧；推理流程 = 文本 → TTS(16k) → whisper 音频特征 → UNet 批推理 → VAE 解码 → 贴回原帧（blending）。 |
| `tts.py` | edge-tts 合成（渲染线程内同步调用）：产物两路——16k float32（whisper 特征）与 48k int16 PCM（WebRTC 音轨）；音色不支持时自动退回默认音色重试。 |
| `webrtc.py` | aiortc 输出轨道：`VideoTrack`（按音频时钟取对应口型帧，时间戳取墙钟防积压漂移）+ `AudioTrack`（48k/20ms 块，段序号变化重置读指针）。 |
| `config.py` | 配置加载：`config.yaml` + 命令行覆盖；相对路径统一相对 `digital_human/` 解析。 |

## 设计要点

- **线程模型**：渲染在后台线程，WebRTC 轨道在各自事件循环按自身时钟读快照——引擎只做线程安全的快照读写，两端解耦。
- **自适应 fps + 预滚缓冲**：启动自测渲染吞吐定 fps 上限（`fps_margin` 安全系数），开播前预滚 `preroll_seconds` 吸收批次间推理空档，防欠帧冻嘴。
- **发送降采样**：720x1180 原始尺寸 VP8 软编码撑不满帧率（积压 → 时间戳漂移 → 嘴画落后声音），默认半分辨率发送；本机配置置 1.0 与待机画质对齐。
- **状态语义**：`speaking` 与 `playing` 分开——显示端据 `playing` 切换实时画面，避免「还在准备阶段就切走待机」。
