"""模型 backend 适配器单测：验证指定模型的选择逻辑与请求载荷构造。

覆盖 FunASR / CosyVoice2 / GPT-SoVITS / Qwen3 / MuseTalk / LatentSync 的
工厂选择、载荷构造与 Mock 降级，全程不发起真实网络请求。
"""
from __future__ import annotations

import base64

import pytest

from config.settings import settings
from models import asr as asr_mod
from models import avatar as avatar_mod
from models import llm as llm_mod
from models import tts as tts_mod
from models.asr import FunASRClient, MockASR, get_asr
from models.avatar import (
    LatentSyncAvatar,
    MockAvatar,
    MuseTalkAvatar,
    get_avatar,
)
from models.llm import MockLLM, OpenAICompatLLM, Qwen3LLM, get_llm
from models.tts import CosyVoiceTTS, GptSoVitsTTS, MockTTS, TeacherTTS, get_tts


@pytest.fixture(autouse=True)
def _clear_factory_caches():
    """每个用例前后清空 lru_cache，避免工厂单例串味。"""
    for factory in (get_asr, get_tts, get_llm, get_avatar):
        factory.cache_clear()
    yield
    for factory in (get_asr, get_tts, get_llm, get_avatar):
        factory.cache_clear()


# --------------------------------------------------------------------- ASR
def test_get_asr_selects_funasr(monkeypatch):
    monkeypatch.setattr(settings, "asr_provider", "funasr")
    monkeypatch.setattr(settings, "asr_api_base", "http://127.0.0.1:10095")
    monkeypatch.setattr(settings, "funasr_model", "paraformer-zh")
    client = get_asr()
    assert isinstance(client, FunASRClient)
    assert client.model == "paraformer-zh"


def test_get_asr_fallback_mock(monkeypatch):
    monkeypatch.setattr(settings, "asr_provider", "mock")
    assert isinstance(get_asr(), MockASR)


# --------------------------------------------------------------------- TTS
def test_get_tts_selects_cosyvoice(monkeypatch):
    """克隆型引擎一律包在 ``TeacherTTS`` 门面里。

    门面的职责是**按老师分派**：能用素材干声的老师走克隆，素材里读不出人声的老师
    回退 edge 自带音色（且必须保持性别正确）。所以这里不能断言「拿到的是 CosyVoiceTTS」，
    而要断言「克隆引擎被正确装配进它、并且带了 edge 兜底」。
    """
    monkeypatch.setattr(settings, "tts_provider", "cosyvoice")
    monkeypatch.setattr(settings, "cosyvoice_api_base", "http://127.0.0.1:9000")
    monkeypatch.setattr(settings, "tts_ref_audio", "/ref.wav")
    client = get_tts()
    assert isinstance(client, TeacherTTS)
    assert isinstance(client.clone, CosyVoiceTTS)
    assert client.clone.ref_audio == "/ref.wav"
    assert client.edge is not None, "缺少 edge 兜底：素材不足的老师会没有声音"


def test_get_tts_selects_gptsovits(monkeypatch):
    monkeypatch.setattr(settings, "tts_provider", "gptsovits")
    monkeypatch.setattr(settings, "gptsovits_api_base", "http://127.0.0.1:9880")
    client = get_tts()
    assert isinstance(client, TeacherTTS)
    assert isinstance(client.clone, GptSoVitsTTS)
    assert client.edge is not None


def test_get_tts_fallback_mock(monkeypatch):
    monkeypatch.setattr(settings, "tts_provider", "mock")
    assert isinstance(get_tts(), MockTTS)


def test_cosyvoice_payload_zero_shot_clone():
    client = CosyVoiceTTS("http://x", speaker="teacher", ref_audio="/a.wav", ref_text="你好")
    payload = client._payload("同学们好")
    assert payload["text"] == "同学们好"
    assert payload["stream"] is True
    assert payload["ref_audio"] == "/a.wav" and payload["ref_text"] == "你好"


def test_gptsovits_payload_ref_audio_path():
    client = GptSoVitsTTS("http://x", ref_audio="/a.wav", prompt_text="你好")
    payload = client._payload("上课")
    assert payload["ref_audio_path"] == "/a.wav"
    assert payload["prompt_text"] == "你好"
    assert payload["text_lang"] == "zh"


# --------------------------------------------------------------------- LLM
def test_get_llm_selects_qwen3(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "qwen3")
    monkeypatch.setattr(settings, "llm_api_key", "sk-local")
    monkeypatch.setattr(settings, "llm_model", "Qwen3-14B")
    client = get_llm()
    assert isinstance(client, Qwen3LLM)
    assert client.model == "Qwen3-14B"


def test_get_llm_openai_when_no_provider(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "llm_api_key", "sk-x")
    client = get_llm()
    assert isinstance(client, OpenAICompatLLM) and not isinstance(client, Qwen3LLM)


def test_get_llm_fallback_mock_without_key(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "qwen3")
    monkeypatch.setattr(settings, "llm_api_key", "")
    assert isinstance(get_llm(), MockLLM)


# --------------------------------------------------------------------- Avatar
def test_get_avatar_selects_musetalk(monkeypatch):
    monkeypatch.setattr(settings, "avatar_backend", "musetalk")
    monkeypatch.setattr(settings, "avatar_api_base", "http://127.0.0.1:8010")
    client = get_avatar()
    assert isinstance(client, MuseTalkAvatar) and client.produces_video is True


def test_get_avatar_selects_latentsync(monkeypatch):
    monkeypatch.setattr(settings, "avatar_backend", "latentsync")
    monkeypatch.setattr(settings, "avatar_api_base", "http://127.0.0.1:8010")
    assert isinstance(get_avatar(), LatentSyncAvatar)


def test_get_avatar_fallback_mock(monkeypatch):
    monkeypatch.setattr(settings, "avatar_backend", "mock")
    client = get_avatar()
    assert isinstance(client, MockAvatar) and client.produces_video is False


def test_musetalk_payload_has_model_and_fps():
    client = MuseTalkAvatar("http://x", fps=30)
    payload = client._payload(b"audio", "文本")
    assert payload["model"] == "musetalk-1.5"
    assert payload["fps"] == 30
    assert payload["audio"] == base64.b64encode(b"audio").decode("ascii")


def test_musetalk_default_fps_matches_base_video():
    """默认 fps 必须与引擎底片兜底值（24）对齐，否则音画错位。"""
    client = MuseTalkAvatar("http://x")
    payload = client._payload(b"audio", "文本")
    assert payload["fps"] == 24


def test_musetalk_bbox_shift_only_in_payload_when_set():
    """bbox_shift=None 时不下发（沿用引擎侧 MUSETALK_BBOX_SHIFT 默认）；传入时覆盖。"""
    client_default = MuseTalkAvatar("http://x")
    assert "bbox_shift" not in client_default._payload(b"a", "t")

    client_shifted = MuseTalkAvatar("http://x", bbox_shift=-5)
    assert client_shifted._payload(b"a", "t")["bbox_shift"] == -5


def test_get_avatar_passes_settings_to_musetalk(monkeypatch):
    """工厂必须把 settings.avatar_fps / avatar_bbox_shift 透传给客户端。"""
    monkeypatch.setattr(settings, "avatar_backend", "musetalk")
    monkeypatch.setattr(settings, "avatar_api_base", "http://127.0.0.1:8012")
    monkeypatch.setattr(settings, "avatar_fps", 25)
    monkeypatch.setattr(settings, "avatar_bbox_shift", -3)
    client = get_avatar()
    assert isinstance(client, MuseTalkAvatar)
    assert client.fps == 25
    assert client.bbox_shift == -3


# --------------------------------------------------------------------- 模块导入完整性
def test_new_backends_importable_from_models_pkg():
    import models

    for name in ("FunASRClient", "CosyVoiceTTS", "GptSoVitsTTS", "Qwen3LLM",
                 "MuseTalkAvatar", "LatentSyncAvatar"):
        assert hasattr(models, name)
    # 触发一次模块属性访问，确保无导入副作用错误
    assert asr_mod.FunASRClient and tts_mod.CosyVoiceTTS and llm_mod.Qwen3LLM and avatar_mod.MuseTalkAvatar
