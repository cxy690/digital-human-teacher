"""config 包：全局配置、常量与日志。"""
import os

# 让 paddlepaddle（protobuf 3.x 生成代码）与 mediapipe（protobuf 4.x）共存
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from config.constants import (
    BoardOp,
    ClassPhase,
    IntentType,
    MicState,
)
from config.logging import get_logger, setup_logging
from config.settings import Settings, get_settings, settings

__all__ = [
    "BoardOp",
    "ClassPhase",
    "IntentType",
    "MicState",
    "Settings",
    "get_settings",
    "settings",
    "get_logger",
    "setup_logging",
]
