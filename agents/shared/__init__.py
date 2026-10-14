"""agents.shared 包：学生端与 AI 教师共享的能力（麦控、白板）。"""
from agents.shared.mic_control import MicController
from agents.shared.whiteboard import Whiteboard

__all__ = ["MicController", "Whiteboard"]
