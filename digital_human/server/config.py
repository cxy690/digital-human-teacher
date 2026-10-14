"""配置加载：config.yaml + 命令行覆盖。

所有相对路径统一相对 digital_human 根目录解析；avatar_root 缺省指向项目级 data/avatars。
"""
import os

import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # digital_human/
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DEFAULTS = {
    "avatar_id": "ai3",
    "avatar_root": os.path.join(PROJECT_ROOT, "data", "avatars"),
    "listen_host": "0.0.0.0",
    "listen_port": 8010,
    "batch_size": 8,
    "fps": 25,
    "min_fps": 12,               # 自适应 fps 下限（低于此会有明显掉帧感）
    "fps_margin": 0.85,          # 自适应 fps 安全系数：fps ≤ 实测渲染吞吐 × 该值
    "preroll_seconds": 1.2,      # 开播前预滚缓冲时长（吸收批次间推理空档，防欠帧冻嘴）
    "warmup_frames": 16,         # 启动自测渲染帧数（0 = 关闭自适应，直接用 fps 为目标值）
    "vae_batch_decode": False,   # 本机 8GB 默认逐帧解码；云端 config.cloud.yaml 置 true 走整批
    "send_scale": 0.5,           # WebRTC 发送降采样；本机 config.yaml 置 1.0 与待机画质对齐
    "musetalk_dir": os.path.join(BASE_DIR, "models", "musetalkV15"),
    "vae_dir": os.path.join(BASE_DIR, "models", "sd-vae"),
    "whisper_dir": os.path.join(BASE_DIR, "models", "whisper"),
    "tts": {
        "voice": "zh-CN-XiaoxiaoNeural",
        "rate": "+0%",
        "volume": "+0%",
    },
}

_PATH_KEYS = ("avatar_root", "musetalk_dir", "vae_dir", "whisper_dir")


def _resolve_config_path(path=None):
    """配置档优先级：显式 path > DH_PROFILE 对应的 config.<profile>.yaml > config.yaml。

    本机与云端各用一份 yaml、互不覆盖：云端设 DH_PROFILE=cloud（或 run.py --profile cloud）
    即加载 config.cloud.yaml，未设时回落本机 config.yaml。
    """
    if path:
        return path
    profile = os.environ.get("DH_PROFILE", "").strip()
    if profile:
        candidate = os.path.join(BASE_DIR, "config.%s.yaml" % profile)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(BASE_DIR, "config.yaml")


def load_config(path=None, overrides=None):
    cfg = dict(DEFAULTS)
    cfg["tts"] = dict(DEFAULTS["tts"])

    path = _resolve_config_path(path)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        for k, v in user.items():
            if k == "tts" and isinstance(v, dict):
                # 仅接受已知 tts 配置键，避免多余键传入 EdgeTTS 构造器
                cfg["tts"].update({kk: vv for kk, vv in v.items() if kk in cfg["tts"] and vv is not None})
            elif v is not None:
                cfg[k] = v

    for k, v in (overrides or {}).items():
        if v is not None:
            cfg[k] = v

    # 相对路径统一相对 digital_human 根解析
    for k in _PATH_KEYS:
        v = cfg.get(k)
        if v and not os.path.isabs(v):
            cfg[k] = os.path.abspath(os.path.join(BASE_DIR, v))
    return cfg
