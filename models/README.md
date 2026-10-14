# models — 模型客户端（脑 / 耳 / 口 / 脸）

四类模型能力的统一接口与可插拔实现：**LLM（脑）/ ASR（耳）/ TTS（口）/ Avatar（脸）**，外加视觉理解走 VLM。所有 provider 通过环境变量切换，无密钥时自动降级 Mock——不配置任何密钥即可跑通全链路。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `__init__.py` | 包门户。 |
| `base.py` | 抽象基类与公共数据结构：`BaseLLMClient` / `BaseASRClient` / `BaseTTSClient` / `BaseAvatarClient`；`ToolCall` / `ToolChatResult`（工具调用结果，`raw` 保留服务端原样字段供 ReAct 回填）。 |
| `llm.py` | LLM 客户端：默认 OpenAI 兼容协议（可指向 DeepSeek / 私有化 Qwen3）；`parse_tool_calls()` 归一化工具调用（arguments 是 JSON 字符串，解析失败保留原串交由工具层报错，不打断课堂）；无密钥降级 `MockLLM`。 |
| `asr.py` | ASR 客户端：默认 OpenAI 兼容 `/audio/transcriptions`（Whisper 类）；云上另有 FunASR 引擎适配（`scripts/asr_engine_funasr.py`）；降级 `MockASR`。 |
| `tts.py` | TTS 客户端：OpenAI / Edge（内置）/ CosyVoice 3（兼容 2）/ GPT-SoVITS；`split_sentences()` 按句切分供流式合成；`_teacher_voice_ref()` 按老师取克隆音色配置，缺档案回退全局配置（旧单老师行为）。 |
| `avatar.py` | 数字人客户端：`get_avatar` 按 `AVATAR_BACKEND` 返回 `musetalk` / `latentsync` / `http` / `placeholder` / `mock`（默认，前端照片木偶）。 |

## 设计要点

- **统一接口 + 工厂函数**：上层只依赖 `base.py` 的抽象与 `get_llm()` / `get_asr()` / `get_tts()` / `get_avatar()` 工厂，替换供应商或切私有化部署不改上层代码。
- **降级链**：无密钥 → Mock；渲染不可用 → `PlaceholderAvatar`（静态图+音频占位视频）→ `MockAvatar`（照片木偶）。
- **逐句渲染契约**：数字人配合 `synthesize_stream` 首句音频一出即渲染推流；**返回的视频必须自带音轨**（否则置 `AVATAR_VIDEO_HAS_AUDIO=false`，由 Supervisor 额外下发 `ai_audio` 补声音）。
- **失败不阻断播报**：渲染异常由调用方捕获后回退照片木偶；一次工具调用参数不合法只报错到工具层。
