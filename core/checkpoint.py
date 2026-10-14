"""检查点：把会话关键状态（白板指令、对话历史、题目缓存）落盘，支持断线续聊。

实时场景下网络抖动很常见，学生刷新/重连后应能恢复白板内容与答疑上下文。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from config.logging import get_logger
from core.state.session import SessionState

log = get_logger("core.checkpoint")

_DEFAULT_DIR = Path("data/raw/checkpoints")


class Checkpointer:
    """基于 JSON 文件的会话检查点（可替换为 Redis / DB 实现）。"""

    def __init__(self, directory: str | os.PathLike[str] = _DEFAULT_DIR) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.dir / f"{session_id}.json"

    def save(self, state: SessionState) -> Path:
        """保存会话快照。"""
        payload = {
            "session_id": state.session_id,
            "student_id": state.student_id,
            "saved_at": time.time(),
            "board_seq": state.board_seq,
            "board_images": state.board_images,
            "board_ops": [op.model_dump() for op in state.board_ops[-500:]],
            "history": [m.model_dump() for m in state.history[-200:]],
        }
        path = self._path(state.session_id)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        log.debug("checkpoint saved: {}", path)
        return path

    def load(self, session_id: str) -> dict | None:
        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as err:
            log.warning("checkpoint load failed for {}: {}", session_id, err)
            return None

    def delete(self, session_id: str) -> None:
        path = self._path(session_id)
        if path.exists():
            path.unlink()
