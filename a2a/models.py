"""A2A 协议数据模型（Agent Card / Task / Message / Part / Artifact）。

字段名沿用 A2A 规范的 **camelCase**，便于与外部实现（其它厂商的 A2A client）直接互通，
不用做二次转换。

只实现 ``text`` / ``data`` 两类 Part：``file``（文件、音视频）暂不暴露 ——
数字人视频这类大产物仍走前端 WebSocket 链路（``ai_avatar.stream_url``），
不放进 agent 互操作面，避免把 1~3MB 的二进制塞进 JSON-RPC。
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def now_iso() -> str:
    """A2A 状态时间戳（ISO 8601，带时区）。"""
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


class Part(BaseModel):
    """消息 / 产物的一个片段。"""

    kind: Literal["text", "data"] = "text"
    text: str | None = None
    data: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def of_text(cls, text: str, **meta: Any) -> "Part":
        return cls(kind="text", text=text, metadata=meta)

    @classmethod
    def of_data(cls, data: dict[str, Any], **meta: Any) -> "Part":
        return cls(kind="data", data=data, metadata=meta)


class Message(BaseModel):
    """一次交互消息（学生提问 / 老师回复）。"""

    kind: Literal["message"] = "message"
    messageId: str = Field(default_factory=lambda: new_id("msg"))
    role: Literal["user", "agent"] = "user"
    parts: list[Part] = Field(default_factory=list)
    contextId: str | None = None
    taskId: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def text(self) -> str:
        """把所有文本 Part 拼成一句话（当前只支持文本输入）。"""
        return "\n".join(
            p.text or "" for p in self.parts if p.kind == "text"
        ).strip()


class Artifact(BaseModel):
    """任务产物：讲解话术、白板指令、追问题目等。"""

    artifactId: str = Field(default_factory=lambda: new_id("art"))
    name: str = ""
    description: str = ""
    parts: list[Part] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


TaskState = Literal[
    "submitted", "working", "input-required", "completed", "canceled", "failed"
]


class TaskStatus(BaseModel):
    state: TaskState = "submitted"
    timestamp: str = Field(default_factory=now_iso)
    message: Message | None = None


class Task(BaseModel):
    """A2A 任务：一次问答的完整生命周期载体。"""

    kind: Literal["task"] = "task"
    id: str = Field(default_factory=lambda: new_id("task"))
    contextId: str = Field(default_factory=lambda: new_id("ctx"))
    status: TaskStatus = Field(default_factory=TaskStatus)
    artifacts: list[Artifact] = Field(default_factory=list)
    history: list[Message] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def touch(self, state: TaskState, message: Message | None = None) -> None:
        """推进任务状态（同时刷新时间戳）。"""
        self.status = TaskStatus(state=state, message=message)


# --------------------------------------------------------------------------- 卡片
class AgentSkill(BaseModel):
    """Agent 对外声明的一项能力。"""

    id: str
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)


class AgentCapabilities(BaseModel):
    streaming: bool = False
    pushNotifications: bool = False


class AgentCard(BaseModel):
    """Agent Card：外部 client 用它发现能力与调用入口。

    暴露在两个路径上：``/.well-known/agent-card.json``（0.3.x）与
    ``/.well-known/agent.json``（0.2.x，同一份内容，兼容老客户端）。
    """

    protocolVersion: str = "0.3.0"
    name: str = "education-tutor"
    description: str = ""
    url: str = "/a2a"
    version: str = "0.1.0"
    preferredTransport: str = "JSONRPC"
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    defaultInputModes: list[str] = Field(default_factory=lambda: ["text/plain"])
    defaultOutputModes: list[str] = Field(
        default_factory=lambda: ["text/plain", "application/json"]
    )
    skills: list[AgentSkill] = Field(default_factory=list)
