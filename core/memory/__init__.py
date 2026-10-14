"""长期记忆（学情记忆）：跨会话记住「这位学生哪儿不会、习惯怎样」。

模块分工：

| 文件 | 职责 |
|---|---|
| ``models.py`` | 记忆条目（kind / 权重 / 时间衰减 / 命中次数） |
| ``store.py`` | 按学生分文件的 append-only JSONL，含去重、淘汰与**物理删除**（合规） |
| ``prompt.py`` | 把检索到的记忆渲染成可注入的 system 片段 |
| ``writer.py`` | 写入方：判题判错零成本记错题；可选模型蒸馏薄弱点 |

入口约定：读取走 :func:`build_memory_prompt_for`（``agents/shared/teaching_layers``
在装配提示词时调用），写入走 :class:`MemoryWriter`（``Supervisor`` 在判题 / 答疑结束
时调用）。除这两处外，链路上不再有别的记忆读写点 —— 记忆是**旁路增强**，
任何异常都只告警，绝不允许阻塞课堂。
"""
from core.memory.models import KIND_LABELS, MemoryItem
from core.memory.prompt import build_memory_prompt, build_memory_prompt_for, render_memory
from core.memory.store import MemoryStore, get_memory_store
from core.memory.writer import MemoryWriter

__all__ = [
    "KIND_LABELS",
    "MemoryItem",
    "MemoryStore",
    "MemoryWriter",
    "build_memory_prompt",
    "build_memory_prompt_for",
    "get_memory_store",
    "render_memory",
]
