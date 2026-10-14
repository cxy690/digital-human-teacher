"""pytest 兜底配置。

两件事：
1. **测试隔离**：清空外部服务密钥，保证单测离线可跑、不产生费用、不因网络抖动而挂住。
   必须在导入 ``config.settings`` / ``models.*`` 之前设置——环境变量优先级高于 .env 文件，
   密钥为空时 get_llm()/get_asr()/get_tts() 一律降级为 Mock。
   历史坑：未做隔离时，测试会真的去调 .env 里配的 LLM（deepseek-chat），套件表现为「卡住」。
2. **异步兜底**：未安装 pytest-asyncio 时提供极简 async 执行器，保证 async 测试也能运行。
"""
from __future__ import annotations

import asyncio
import inspect
import os

# --- 1. 测试隔离（必须最先执行）---
# INTENT_LLM_API_KEY 也要清：意图分类小模型是独立 key，不清的话单测会真的调外部 API
# （慢、依赖网络、且产生费用）。
for _key in (
    "LLM_API_KEY", "INTENT_LLM_API_KEY", "ASR_API_KEY", "TTS_API_KEY", "OCR_CLOUD_KEY", "VLM_API_KEY"
):
    os.environ[_key] = ""
# 显式走 Mock，避免依赖「密钥为空」这一隐式降级
os.environ["LLM_PROVIDER"] = "mock"
os.environ["ASR_PROVIDER"] = "mock"
os.environ["TTS_PROVIDER"] = "mock"
# 数字人：测试不需要真渲染引擎（否则会去连 :8012 的 MuseTalk 服务）
os.environ["AVATAR_BACKEND"] = "mock"
os.environ["AVATAR_ENABLE"] = "false"
# 真题库（ai_teacher.exercise_item）：单测不连真库（否则每次出题都会试连本机 MySQL）。
# 需要题库路径的用例自行 monkeypatch settings.exercise_bank_enable=True
# 并 exercise_bank.set_picker(fake) 注入假题源。
os.environ["EXERCISE_BANK_ENABLE"] = "false"
# 编排与答疑形态固定为「自研图 + 结构化答疑」：否则开发者 .env 里的
# ORCHESTRATOR=langgraph / BOARD_FIRST_MODE=single_call 会把测试带到另一条链路上，
# 结果「在 A 机器通过、在 B 机器失败」。要测另一条链路的用例自行 monkeypatch settings。
os.environ["ORCHESTRATOR"] = "builtin"
os.environ["BOARD_FIRST_MODE"] = "off"

import pytest  # noqa: E402

try:  # 已安装 pytest-asyncio 时交给它处理
    import pytest_asyncio  # noqa: F401

    _HAS_ASYNCIO = True
except ImportError:  # pragma: no cover - 取决于环境
    _HAS_ASYNCIO = False


if not _HAS_ASYNCIO:

    @pytest.hookimpl(tryfirst=True)
    def pytest_pyfunc_call(pyfuncitem: pytest.Function):  # type: ignore[no-redef]
        testfunc = pyfuncitem.obj
        if inspect.iscoroutinefunction(testfunc):
            kwargs = {
                name: pyfuncitem.funcargs[name]
                for name in pyfuncitem._fixtureinfo.argnames
            }
            asyncio.run(testfunc(**kwargs))
            return True
        return None
