"""a2a 包：A2A（Agent2Agent）协议的最小实现。

当前范围（与 ``AgentCard.capabilities`` 声明一致）：

* ✅ Agent Card 发现（``/.well-known/agent-card.json``、``/.well-known/agent.json``）
* ✅ ``message/send``（阻塞式）/ ``tasks/get`` / ``tasks/cancel``
* ✅ Task 生命周期：submitted -> working -> input-required|completed|failed|canceled
* ❌ ``message/stream``（SSE 流式）、push notification、认证

⚠️ 包名 ``a2a`` 与官方 SDK（``a2a-sdk``，其导入名也是 ``a2a``）冲突：本项目根目录
在 ``sys.path`` 靠前，会遮蔽已安装的 SDK。本实现是自研的，不依赖 SDK；若将来要
换成官方 SDK，请先改包名（如 ``a2a_gateway``）。
"""
from a2a.card import build_agent_card
from a2a.executor import SupervisorAgentExecutor
from a2a.models import AgentCard, Artifact, Message, Part, Task
from a2a.rpc import handle_rpc
from a2a.store import A2ATaskStore

__all__ = [
    "AgentCard",
    "Artifact",
    "A2ATaskStore",
    "Message",
    "Part",
    "SupervisorAgentExecutor",
    "Task",
    "build_agent_card",
    "handle_rpc",
]
