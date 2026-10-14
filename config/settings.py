"""运行配置：从环境变量 / .env 加载，集中管理模型、音视频、白板开关。"""
from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """项目全局配置。所有敏感信息只从环境变量读取，不落库。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- 基础 --
    app_name: str = "education"
    env: str = Field(default="local", description="local / dev / prod")
    debug: bool = True
    log_level: str = "INFO"

    # -- 日志落盘（运行时日志统一进这个文件夹，见 config/logging.py）--
    # 保留策略**三个条件同时生效**（任一超限就从最旧的开始删，且永远至少留一份）：
    #   log_keep（多久以前的删） + log_max_files（最多几份） + log_max_total_mb（总量上限）
    # 均可被 .env 覆盖，例如：LOG_KEEP=30 days / LOG_MAX_FILES=90 / LOG_MAX_TOTAL_MB=200
    log_dir: str = Field(default="logs", description="日志目录（相对项目根或绝对路径）")
    log_rotate: str = Field(default="00:00", description="轮转策略：'00:00' 按天 / '10 MB' 按大小")
    log_max_total_mb: int = Field(default=200, description="日志总量上限(MB)，超出删最旧；0=不限")
    log_max_files: int = Field(default=90, description="日志文件份数上限，超出删最旧；0=不限")
    log_keep: str = Field(default="30 days", description="保留时长（'30 days' / '12 h' 这类写法）")

    # -- 大模型（OpenAI 兼容协议） --
    llm_provider: str = "openai"       # openai / qwen3（私有化 vLLM/SGLang，OpenAI 兼容）
    llm_api_base: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = 30.0
    llm_enable_thinking: bool = False  # Qwen3 思考模式开关（直播场景建议关闭以降延迟）
    # 额外请求体：JSON 字符串，原样合并进 /chat/completions 的 payload，
    # 用于传厂商专有开关（不填 = 不传，对其它 OpenAI 兼容厂商零影响）。
    # 典型用法（DeepSeek V4 关思考）：LLM_EXTRA_BODY={"thinking": {"type": "disabled"}}
    # 为什么要关：思考模型默认开思考，推理 token 会**吃掉 max_tokens 预算**并显著
    # 抬高首响延迟——直播课堂场景下会出现「讲解变短甚至为空」。
    llm_extra_body: str = ""
    # 单次回复最大 token（未显式指定时的默认值）。答疑要输出 JSON（话术 + 多步板书），
    # 1024 在复杂题上会被截断，导致 JSON 解析失败、AI 只回一句「这道题我这边没组织好」。
    llm_max_tokens: int = 3072

    # -- 意图识别小模型（agents/router） --
    # 只在「规则判不准 / 复杂·多意图」时调用，不占常见输入（控制/闲聊/单一提问）的首响预算。
    # 不填 model 则复用上面的主 LLM —— 零新增基建；也可指向更小更快的模型
    # （如 deepseek-flash / qwen-turbo / gpt-4o-mini）以进一步压延迟与成本。
    intent_llm_enable: bool = True
    intent_llm_model: str = ""            # 空 = 沿用 llm_model
    intent_llm_api_base: str = ""         # 空 = 沿用 llm_api_base
    intent_llm_api_key: str = ""          # 空 = 沿用 llm_api_key
    intent_llm_timeout_seconds: float = 2.5   # 超时即回退规则结果，不阻塞答疑
    intent_llm_max_tokens: int = 160          # 分类输出只有几十个 token，给足即可

    # -- 教师 Skill 自我更新（经验沉淀，见 core/skills/experience） --
    # 边界：只往 data/teacher_experience/<老师key>.jsonl 追加「经验条目」并注入提示词，
    # **不改写人格 Skill 本体**（core/skills/teacher_personality 的提示词永远由人维护）。
    skill_self_update_enable: bool = True     # 总开关：回合结束后异步复盘
    skill_experience_dir: str = "data/teacher_experience"   # 经验落盘目录（按环境隔离）
    skill_experience_inject_limit: int = 6    # 每次讲解最多注入几条经验（0=不注入）
    # 每位老师最多保留多少条生效经验：**0 = 不限（默认）**，永远不自动淘汰。
    # 自动更新学到的东西不该被系统自己丢掉；提示词侧的条数上限由
    # SKILL_EXPERIENCE_INJECT_LIMIT 单独控制，文件再涨也不会撑大 prompt。
    # 若显式设为正数，超出时才停用分数最低的条目（仍不删文件，可 restore）。
    skill_experience_max_items: int = 0
    # ---- L2 进化决策层：按**实测收益**决定注入哪几条（见经验模块的 select / feedback）----
    # 关掉时退回旧行为：按 ``hits × confidence`` 静态排序取前 N 条。
    # 静态排序衡量的是「被反复发现过几次」，不是「有没有帮上忙」——
    # 一条被反复提到、注入后毫无作用的经验，照样排第一。
    # 开着则按 Beta 后验**采样**定序（Thompson）：战绩好的多数时候被选中（利用），
    # 战绩未知的后验宽、偶尔会被试到（探索）—— 名额因此会流向真正有用的那几条。
    skill_experience_bandit_enable: bool = True
    # 是否在回合结束后把「这批经验帮上忙了没有」回写进经验文件（bandit 的奖励信号）。
    # 关掉则经验只进不出、bandit 永远没有反馈（等于没装），一般不用关。
    skill_experience_feedback_enable: bool = True
    skill_reflect_max_items: int = 2          # 单次复盘最多沉淀几条
    skill_reflect_min_confidence: float = 0.6 # 低于该置信度的经验不落盘
    # 复盘节流（秒）：0=每次答疑都复盘；调大可在高频提问时省 LLM 开销
    skill_reflect_min_interval_seconds: float = 0.0

    # -- Agent 技能（模型可自主调用的技能单元 + 自我更新，见 docs/agent_skills.md）--
    # 与原子工具的分工：工具是「查教材/解方程」这类无判断能力，技能是「何时表扬、
    # 要不要翻旧账」这类带判断的业务能力。两者对模型是同一个接口（function calling）。
    agent_skills_enable: bool = True           # 是否把内置技能暴露给模型（关掉=回到只有工具的旧行为）
    agent_skill_usage_dir: str = "data/skill_usage"   # 技能战绩落盘目录（按环境隔离，不随代码同步）
    # 每个技能最多附加几条「学到的经验」作为实战补充（0=不接经验增强）。
    # 人写的 instruction 是基底，学到的经验只在其上追加，与人格层「人格+经验」同构。
    agent_skill_learned_max: int = 2
    # 战绩过差的技能是否**自动停用**（默认关：与经验抽检台同一立场 ——
    # 自动学习到的东西不该被系统自己静悄悄丢掉，先让人看一眼）。
    # 打开后：pulls 达标且有效率低于阈值就不再暴露给模型（可人工恢复）。
    agent_skill_auto_retire: bool = False
    agent_skill_min_pulls: int = 5             # 少于这么次实战不下结论（样本太少没有意义）
    agent_skill_min_gain_rate: float = 0.3     # 有效率低于它认为「这个技能没什么用」

    # -- Agent 自主性（工具调用 / 自主规划 / ReAct 闭环，见 docs/agent_autonomy.md）--
    # 三者可独立关闭：任一关掉都退回「确定性流水线」，行为与加装前一致。
    agent_tools_enable: bool = True              # 是否把工具暴露给模型（原生 function calling）
    agent_tool_max_calls: int = 6                # 单回合工具调用总次数上限（防失控）
    agent_react_enable: bool = True              # 答疑走「思考 -> 行动 -> 观察」闭环
    agent_react_max_steps: int = 4               # 单回合最多几轮「思考 + 工具」
    agent_react_timeout_seconds: float = 45.0    # 闭环总超时；超时用已有内容收尾，不阻塞上课
    agent_planner_enable: bool = True            # 回合开始前先产出执行计划（由模型自主决定）
    agent_planner_timeout_seconds: float = 6.0   # 规划超时；超时回退规则计划
    agent_planner_max_tokens: int = 400

    # -- 答疑产物的输出 token 上限 --
    # 注意这是**上限，不是目标**：抬高它几乎不花钱（模型该写多少还是写多少），
    # 只是不再把长讲解截断。
    #
    # 为什么必须给足：答疑契约要求一次产出 speech（最长 400 字，重点页实测 300~400）
    # + steps（4~6 步）+ knowledge_points（2~3）+ pitfalls（1~2）+ equation/answer
    # + followup（含 options），实测正好卡在 900~1024 的边界上。而**截断的后果不是
    # 「话短一点」**，是 JSON 不闭合 -> 整块降级 -> 白板全空、步骤全丢。
    # L4 评测实测到过一次：一道「讲讲二次函数的图像和性质」话术断在半句、
    # 无板书无步骤，规则分从 100 掉到 35（见 evals/ 的门禁报告）。
    #
    # 对照：课件讲解用 1400（agents/lecturer/agent.py 的 _SLIDE_MAX_TOKENS），
    # 而答疑的字段比讲课还多，所以这里比它更宽。
    answer_max_tokens: int = 1800

    # -- 长期记忆（学情记忆：错题 / 薄弱点 / 偏好，见 core/memory）--
    # 与「教师经验」正交：经验是老师会了什么（按老师存），记忆是这位学生哪儿不会（按学生存）。
    memory_enable: bool = True
    memory_dir: str = "data/memory"              # 落盘目录（运行时产物，按环境隔离）
    memory_inject_limit: int = 5                 # 每次讲解最多注入几条学情记忆（0=不注入）
    # 单个学生最多保留多少条生效记忆：**0 = 不限（默认）**，永远不自动淘汰
    # （理由同 SKILL_EXPERIENCE_MAX_ITEMS：自动学到的内容不主动丢；
    #  注入条数由 MEMORY_INJECT_LIMIT 单独控制）。
    memory_max_items: int = 0
    memory_distill_enable: bool = False          # 是否用 LLM 从回合里蒸馏薄弱点（默认关，省额度）
    memory_distill_min_interval_seconds: float = 120.0   # 蒸馏节流（秒）

    # -- 多模态视觉大模型（题目图片理解 / 圈画定位） --
    vlm_model: str = "gpt-4o"

    # -- 实时语音 Agent --
    asr_provider: str = "openai"       # openai / funasr（中文课堂优，流式+VAD）/ mock
    asr_api_base: str = ""
    asr_api_key: str = ""
    asr_model: str = "whisper-1"
    funasr_model: str = "paraformer-zh"  # FunASR 中文模型
    funasr_hotwords: str = ""            # FunASR 热词（学科术语提升识别率）
    tts_api_base: str = ""
    tts_api_key: str = ""
    tts_model: str = "tts-1"
    tts_voice: str = "alloy"
    tts_provider: str = "openai"  # openai / edge / cosyvoice（CosyVoice 3，兼容 2）/ gptsovits（GPT-SoVITS）/ mock
    # 克隆音色参考（CosyVoice 3/2 / GPT-SoVITS zero-shot 克隆用）
    tts_ref_audio: str = ""              # 参考干声路径/URL
    tts_ref_text: str = ""               # 参考干声对应文本
    cosyvoice_api_base: str = ""         # CosyVoice 3/2 推理服务地址
    gptsovits_api_base: str = ""         # GPT-SoVITS 推理服务地址
    # 豆包（火山引擎）大模型语音合成 —— 走 v3 HTTP 单向流式接口。
    # 鉴权**只认 API Key**（语音控制台「API Key 管理」创建的那串）；
    # AppID + Access Token 在这条链路上恒定 401（load grant not found），别混用。
    doubao_api_key: str = ""
    doubao_api_base: str = "https://openspeech.bytedance.com"
    doubao_resource_id: str = "volc.service_type.10029"   # 大模型语音合成资源位
    doubao_clone_resource_id: str = "seed-icl-2.0"        # 复刻音色专用资源位（与 S_ 音色成对）
    doubao_speaker: str = "zh_female_shuangkuaisisi_moon_bigtts"  # 全局预置音色
    # 按老师分派复刻音色：``math=S_abc123,chinese=S_def456``（key = 学科 key = teacher_id）。
    # 没配到的老师回落到 doubao_speaker；配了 S_ 音色的会自动切到复刻资源位。
    doubao_teacher_speakers: str = ""
    # 全局语速系数（<1 变慢、>1 变快，1.0 = 原速）：零样本克隆会模仿参考干声的语速，
    # 各老师快慢不一，这里在 TTS 汇聚点统一「变速不变调」（ffmpeg atempo，保音高）。
    # 环境变量 TTS_SPEED（如 0.9 = 慢 10%）。允许范围 0.5~2.0。
    tts_speed: float = 1.0

    # -- OCR --
    ocr_provider: str = "paddle"  # paddle / cloud
    ocr_cloud_key: str = ""

    # -- 实时通信 --
    host: str = "0.0.0.0"
    port: int = 8000
    websocket_path: str = "/ws"

    # -- 音视频 --
    rtc_provider: str = "webrtc"   # webrtc / agora / livekit
    rtc_app_id: str = ""
    rtc_app_secret: str = ""

    # -- 白板 --
    board_snapshot_dir: str = "data/raw/board_snapshots"

    # -- RAG --
    rag_top_k: int = 4
    # 检索后端：auto = 配了 Embedding 服务就用向量、否则词法 bigram；vector / keyword 可强制
    rag_backend: str = "auto"
    # Embedding 服务（OpenAI 兼容 /v1/embeddings；留空 = 不用向量检索，走词法）
    # 例：云端私有化 http://127.0.0.1:8101/v1，或第三方 API（bge-m3 / text-embedding-3-small 等）
    embedding_api_base: str = ""
    embedding_api_key: str = ""
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_batch_size: int = 16
    embedding_timeout_seconds: float = 30.0
    # 向量索引落盘目录（scripts/build_rag_index.py 生成；启动时加载，不在线重建）
    rag_index_dir: str = "data/raw/rag_index"

    # -- 编排引擎（builtin：自研 TaskGraph；langgraph：LangGraph StateGraph） --
    orchestrator: str = Field(default="builtin", description="builtin / langgraph")

    # -- 回合级调用链日志（见 core/trace.py）--
    # 开启后，每次学生输入产出回复时会**多打一条 INFO 日志**，列出这次经过了哪些函数、
    # 各花多少毫秒（缩进表示调用层级）。排查「卡在哪一步 / 到底走了哪条分支」很有用；
    # 嫌日志吵可置 false —— 只少一条日志，功能行为完全不变。
    trace_enable: bool = True
    # 调用链日志是否带耗时：false（默认）= 只列「哪个 py 的哪个函数」；true = 每步追加耗时毫秒
    trace_show_ms: bool = False

    # -- 回合裁决落盘（L0 数据观测层，见 core/verdict.py）--
    # 每次学生输入处理完，把**已经产生但只活在内存里**的成败信号
    # （JSON 降级 / 质检扣分 / 调用链异常 / 总耗时）追加进当天一个 JSONL。
    # 它不新增埋点、不改任何行为，只是把信号留下来 —— 没有它就无法回答
    # 「系统在变好还是变坏」。上层（健康画像 / 失败聚类 / 回归门禁）都读它。
    verdict_enable: bool = True
    verdict_dir: str = "data/raw/verdicts"   # 运行时产物，工作区已 gitignore
    verdict_window_days: int = 7             # 健康摘要默认统计最近几天
    verdict_keep_days: int = 30              # 落盘保留天数（写入时顺手清理）
    # 是否随记录保存整个调用链：关掉可省约 80% 体积，但就看不到「卡在哪一步」。
    # 与 TRACE_SHOW_MS 无关 —— 无论 trace 日志开不开，这里都拿得到每步耗时。
    verdict_keep_steps: bool = True

    # -- L3 失败聚类 + 根因分析（见 core/failure_cluster.py）--
    # 聚类本身是纯计算、不花钱，永远可用；这里控制的是**根因分析**那一小段 ——
    # 它要调模型读样本、产出「根因 + 该改哪里」。只在人显式调用时跑
    # （命令行 / 接口），绝不挂主链路。
    failure_analysis_enable: bool = True
    failure_analysis_max_tokens: int = 700

    # -- 数字人音频下发（真实 TTS 音频 + 口型视素） --
    # 注意：这两个键名里的 digital_human 指的是「数字人口型所需的音频/视素下发」，
    # 与合并时已清理掉的 digital_human/ 子服务（WebRTC 路线）无关。
    digital_human_enable: bool = True          # 是否下发真实 TTS 音频（数字人口型用）
    digital_human_viseme_enable: bool = True   # 是否随音频下发视素时间线

    # -- 数字人渲染引擎（真人形象视频） --
    # L3 实时口型：mock / placeholder（静态图+音频占位视频）/ http / musetalk（MuseTalk1.5）/ latentsync
    avatar_backend: str = "mock"
    avatar_api_base: str = ""                  # 渲染服务地址，如 http://127.0.0.1:8010
    avatar_api_key: str = ""
    avatar_timeout_seconds: float = 60.0
    avatar_source_image: str = ""              # 老师肖像/底片图（http 渲染服务驱动用）
    # 渲染帧率：必须与底片视频实际 fps 一致（当前基元库 talk/nod/shake/shrug/point.mp4 均为 24fps，见
    # scripts/avatar_engine_musetalk_v15.py:_read_frames）。客户端 fps 若与底片 fps 不一致，
    # 会导致 whisper 特征切块数量与容器帧率错配，表现为画面加速/口型对不上。
    avatar_fps: int = 24
    # 人脸框上下平移（MuseTalk bbox_shift 语义）：正值向下扩（多包下巴/脖子），
    # 负值向上扩（多包额头/头发）；官方推荐区间 [-10, 10]，常用 0 或 -5。
    avatar_bbox_shift: int = -4
    # 渲染引擎返回的视频是否自带音轨（真实引擎：是，口型与语音天然同步）；
    # 置 False 时（如用无声示例视频空跑链路）会额外下发 TTS 音频，保证学生听得到声音。
    avatar_video_has_audio: bool = True
    # 无 TTS 音频时是否仍调用渲染服务：仅用于「示例视频 / 桩服务」验证链路，生产保持 False。
    avatar_allow_silent_audio: bool = False
    # 视频数字人渲染失败时，是否降级为「静态照片 + 近似口型」的照片木偶。
    # 默认 False = 保持视频优先：渲染失败只上报状态并提示，不切换到木偶形象
    # （木偶口型是 viseme 近似值，观感像「形象坏了/重影」，容易被误判为画质问题）。
    avatar_fallback_puppet: bool = False
    # 单句视频渲染的失败重试次数：引擎重启/加载中会短暂不可用，重试可避免误降级。
    avatar_render_attempts: int = 3
    # 每句 TTS 音频尾部追加的纯静音时长（秒）：口型引擎处理静音时嘴部趋于闭合，
    # 故补静音可让生成的视频以「闭嘴稳定帧」收尾，讲完切回待机时才能平滑交叉淡化；0 = 关闭。
    # ⚠️ 只给「整段话的最后一句」补：句间也补会把口型切成一段一段的（末句闭嘴、下句重新张嘴），
    # 观感就是「每句话说完嘴都断开」，衔接不上下一句。
    avatar_tail_silence_seconds: float = 0.6
    # 同一段话内部「句子与句子之间」补的静音时长（秒）：默认 0 = 不补，
    # 上一句的末帧嘴型直接接下一句的首帧（口型连续衔接）；需要一点自然停顿感可调 0.1~0.2。
    # 注：MuseTalk 引擎自带首尾静音裁剪（MUSETALK_TRIM_SILENCE=1）时，这段静音会被裁掉。
    avatar_sentence_gap_seconds: float = 0.0
    # 渲染好的单句视频改为「落盘 + /media/avatar_stream/<file>.mp4」由前端流式加载，
    # 避免 1~3MB base64 经 WebSocket 传输造成 JSON.parse/解码卡顿；False = 仍走 base64。
    avatar_stream_url_enable: bool = True

    # -- 数字人子服务托管（合并自 education_fusion：digital_human/，MuseTalk 口型 + WebRTC 推流）--
    # 与上面的 avatar_backend 正交：avatar_backend 决定「后端逐句渲染视频」走哪条链路，
    # 这里决定「是否启用/托管独立的数字人子服务」（:8010，由 core/avatar_launcher.py 探活拉起）。
    # 默认关闭，主链路行为不变。
    avatar_enable: bool = False
    # 数字人专用解释器（CUDA torch）；留空自动探测常见 conda 环境
    avatar_python: str = ""
    # 是否在服务启动时自动拉起数字人子服务；云上/Linux 置 False（由外部脚本托管）
    avatar_autostart: bool = True
    # 云上部署：数字人服务公网地址（学生端浏览器直连它拉 WebRTC）；留空按请求 Host 动态改写
    avatar_public_base: str = ""
    # 未选学科时的兜底素材包
    avatar_id: str = "chinese"
    avatar_tts_voice: str = "zh-CN-XiaoxiaoNeural"

    # -- WebRTC ICE（数字人媒体流跨公网连接用）--
    # 本机演示用默认 STUN 即可；云上 NAT 后必须配 STUN，仍打不通再加 TURN 中继
    rtc_stun_url: str = "stun:stun.l.google.com:19302"
    rtc_turn_url: str = ""
    rtc_turn_username: str = ""
    rtc_turn_credential: str = ""

    # -- 数字人形象/音色：一律由 data/teachers/<teacher_key>/teacher.json 决定 --
    # （学科 -> 老师 -> 底片/音色 的解析见 config/teachers.py；引擎按 teacher_id 取底片）
    # 表情技能：学生答对时由 LLM 判定情境并下发表情贴纸（见 core/skills/expression_skill.py）
    avatar_expression_enable: bool = True

    # -- AI 教师驱动视频源（换人只改这里；见 data/ai_teacher/ 与 core/avatar_source.py）--
    # 主底片：默认 data/ai_teacher/talk.mp4；文件缺失或校验不通过时自动回退到 fallback。
    ai_teacher_video_path: str = "data/ai_teacher/talk.mp4"
    # 回退底片：默认与主底片同源（data/ai_teacher/talk.mp4）；仍可用 AI_TEACHER_FALLBACK_PATH 指向别处
    ai_teacher_fallback_path: str = "data/ai_teacher/talk.mp4"
    # 动作基元库目录：主底片 talk.mp4 + 手势 nod/shake/shrug/point.mp4（+ 可选 idle.mp4）。
    ai_teacher_clip_dir: str = "data/ai_teacher"
    # 人脸预处理本地缓存根目录（按 video_hash 分子目录，换源自动失效；见 scripts/clear_cache.py）。
    avatar_cache_dir: str = "data/cache"
    # 校验阈值：分辨率最短边（像素）与最低帧率（MuseTalk 标准 24fps，不得低于）。
    ai_teacher_min_side: int = 512
    ai_teacher_min_fps: float = 24.0

    # -- 教师档案（多老师：每个学科一套底片 + 标定 + 音色；见 config/teachers.py）--
    # 兼容两种写法：项目 .env 里历史上用过 AVATAR_TEACHERS_DIR，代码里是 TEACHERS_DIR。
    # 之前只认 TEACHERS_DIR，导致改 AVATAR_TEACHERS_DIR 时「看起来改了其实没生效」。
    teachers_dir: str = Field(
        default="data/teachers",
        validation_alias=AliasChoices("teachers_dir", "avatar_teachers_dir"),
    )

    # -- 教师人格技能（事实源 .agents/skills/<学科>/SKILL.md）--
    # 每个学科一门独立人格（身份 / 语言风格 / 教学行为 / 互动边界）。
    # 同一份 md 的 frontmatter 还供「风格层」（深度 / 语速 / 讲课风格）使用：
    # 人格层按「学科」统一，其余按「选中的老师变体」差异化。置 false 可回到旧行为。
    teacher_personality_enable: bool = True

    # -- 白板先行（先写「题意理解」上板，再结合它给出最终解答）--
    # off=关闭；single_call=方案B 单次 LLM 产出两段（省一次往返）；
    # two_call=方案A 两次 LLM 调用（两段时序更强，延迟更高）。
    board_first_mode: str = "off"

    # -- 讲课风格：plain=大白话/通俗易懂（推荐中小学）；normal=按学段默认；academic=规范术语 --
    teaching_style: str = "plain"

    # -- 互动式追问（讲解中追问：讲完题主动提问并等学生作答） --
    interaction_enable: bool = True            # 讲完题是否主动追问
    interaction_max_followups: int = 1         # 每道题最多追问次数（防无限循环）
    interaction_max_retries: int = 2           # 单个追问允许的作答重试次数

    # -- 1v1 学科辅导课（学科选择页 -> 打招呼 -> 讲解+出题 -> 总结答疑）--
    course_enable: bool = True
    course_duration_minutes: int = 40      # 一节课时长（分钟）：开课倒计时与「续费」时长
    course_quiz_enable: bool = True        # 讲解完是否自动生成模拟题并等学生作答
    # 出题节奏：讲解满多少分钟后才出第一道模拟题（默认 30 分钟）。
    # 一节课 40 分钟 = 讲解 30 分钟 + 出题/作答 10 分钟（阶段比例见
    # config/constants.py 的 TUTOR_PHASE_RATIOS）：前 30 分钟专心讲解不被打断，
    # 到点后由讲解回合结束（_turn_teach）或会话空闲巡检（core/loop.py）触发出题；
    # 出一道题后重新起算（下一题再等满一个间隔）。
    # 一节课的阶段计时/结算**暂不实现**（后期再做），只用「上次出题时间」做节流，
    # 不需要后端长驻定时器。0 = 恢复旧的「讲完即出题」行为。
    course_quiz_interval_minutes: int = 30
    # 开课后是否**自动**开始讲课件（PPT）：
    #   False（默认）= 先自由问答，学生自己点「课件」选了哪份才进讲解；
    #   True          = 打完招呼就直接开该学科的课件并一路连讲。
    # 即「先答疑、后讲课」还是「一上来就讲课」，由这个开关决定（.env 可覆盖）。
    courseware_auto_start: bool = False
    # 连讲（讲完一页自动翻下一页）**等前端把当前页音频播完**再翻页的上限（秒）。
    # 之前是「按渲染速度翻页」，渲染快于播放时会出现「画面已翻到下一页、
    # 耳朵还在听上一页」。设为 0 = 不等待（恢复旧行为）。
    # 默认给得比较宽松（5 分钟）：逐页讲解是「讲透」而不是念标题，一页朗读 2~4 分钟很正常，
    # 上限只是兜底（前端正常播完会自己回报，不需要等满）。
    courseware_page_wait_seconds: float = 300.0
    # -- 课程计费（见 core/course_billing.py）--
    course_price: float = 0.0              # 整节课价格（0 = 只提示收费、不显示金额）
    course_currency: str = "元"            # 金额单位（仅前端展示用）
    course_free_seconds: int = 120         # 免费时长：≤该秒数结束不收费，超过则收整节课费用
    course_renew_prompt_seconds: int = 300  # 剩余多少秒开始弹「续费 / 知道了」提示（默认 5 分钟）

    # -- 空闲主动问候（学生静默 N 秒后 AI 主动打招呼，并“对应上 AI 刚才问的问题”）--
    idle_nudge_enable: bool = True
    idle_nudge_seconds: int = 30               # 静默多少秒后触发一次主动问候（可 .env 覆盖）
    idle_nudge_max: int = 3                    # 每次静默期最多触发几次

    # 教学护栏与内容安全（面向未成年人为硬性要求）
    guardrail_enable: bool = True
    guardrail_subject: str = ""                    # 限定本学科（空=不限定）
    guardrail_grade: str = ""                      # 限定本学段（空=不限定）
    content_safety_enable: bool = True             # 全量输出内容安全过滤
    ai_content_watermark_enable: bool = True       # AI 生成内容标识/水印
    # 安全事件审计：危机/敏感命中落盘（JSONL，只追加）。落盘是「上报」的实质——
    # push_event 只到当前连接，没人在线就等于没上报，事后也查不到。
    safety_audit_enable: bool = True
    safety_audit_path: str = "data/raw/safety_events.jsonl"

    # -- 数据库（用户 / 学情存储）--
    # MySQL 优先（默认指向本机库 teacher）；测试用 SQLite。生产请用 DATABASE_URL 覆盖，
    # 不要把真实口令写进仓库（此处默认值仅为本地开箱即跑）。
    database_url: str = "mysql+pymysql://root:123456@localhost:3306/teacher?charset=utf8mb4"
    db_echo: bool = False
    # 运行时实时写库：会话落盘（检查点）时同步镜像进数据库，Navicat 里可直接看。
    #   auto（默认）= 仅当 DATABASE_URL 为 MySQL 时开启（本机联调/演示场景）；
    #   on  = 强制开启（含 SQLite，测试用）；off = 关闭（恢复纯 JSON 检查点行为）。
    # 写库在后台线程执行，失败仅告警一次、不影响答疑；实现见 db/runtime_sync.py。
    runtime_db_write: str = "auto"

    # -- 真题库（数字人出题的真题来源，RAG 侧 MySQL 库 ai_teacher.exercise_item）--
    # 与主库（teacher：用户/学情）分离：这里 29k+ 道初中数学真题（7/8/9 年级，来源为
    # 教辅 docx 解析，见 rag/online/config.yaml 同库）。出题优先选真题（有标准答案、
    # 可溯源）；库不可用 / 无匹配时回退 LLM 即兴生成，课堂不中断。测试环境默认关（见 conftest）。
    exercise_bank_enable: bool = True
    exercise_bank_url: str = "mysql+pymysql://root:123456@localhost:3306/ai_teacher?charset=utf8mb4"
    exercise_bank_types: str = "choice,fill"   # 允许选题的题型（solve/other 答案形态不适合自动判分）
    exercise_bank_max_stem_chars: int = 160    # 题干超长不进课堂（播报/白板都不便）
    exercise_bank_allow_review: bool = False   # 是否允许 needs_human_review/suspicious 的题（默认只用质检干净题）
    exercise_bank_skip_figure: bool = True     # 跳过依赖插图的题（题干含「如图」；课堂无图可展示）

    # -- 身份认证（见 core/auth、db/models）--
    auth_enable: bool = True              # 是否启用身份体系（关闭则完全不校验）
    auth_required: bool = False           # true = 未登录一律拒绝（WS 硬门禁）；false = 灰度（带 token 才校验）
    auth_allow_register: bool = True      # 是否开放自助注册
    auth_secret: str = "dev-secret-change-me"   # JWT 签名密钥（生产必须替换）
    auth_token_ttl_minutes: int = 720     # 登录令牌有效期（分钟）
    auth_seed_demo: bool = True           # 启动时预置演示账号（student/parent/teacher/admin）
    cors_allow_origins: str = "*"         # 逗号分隔的允许来源；默认 * 保持与旧行为一致

    # -- 防逆向 / 防盗用 --
    anti_rev_secret: str = "dev-secret-change-me"
    rate_limit_per_min: int = 240

    # -- 活动日志（节点到节点的连接 + 图执行轨迹，见 core/activity_log.py）--
    # 落盘到 ``<logs>/activity/activity-<日期>.jsonl``（append-only，按天滚动）。
    # 默认关闭：逐节点记录量很大（连讲课件一页就有 4~5 条），排障时再打开。
    activity_log_enable: bool = False
    activity_log_dir: str = "logs"            # 运行日志根目录（运行时产物，工作区已 gitignore）
    activity_log_max_mb: float = 20.0         # 单文件上限，超过后按 ``.1/.2`` 序号继续写（0=不限制）
    activity_log_keep_days: int = 7           # 保留天数：写入时顺手删除更早的文件（0=不清理）


@lru_cache
def get_settings() -> Settings:
    """单例配置，避免重复解析 .env。"""
    return Settings()


settings = get_settings()
