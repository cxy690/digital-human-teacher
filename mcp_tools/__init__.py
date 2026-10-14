"""mcp_tools 包：工具注册中心 + MCP Server。"""
from mcp_tools.protocol import (
    LATEST_PROTOCOL_VERSION,
    MCPServer,
    MCPError,
    error_payload,
)
from mcp_tools.registry import ToolRegistry, ToolSpec, build_default_registry

__all__ = [
    "LATEST_PROTOCOL_VERSION",
    "MCPServer",
    "MCPError",
    "ToolRegistry",
    "ToolSpec",
    "build_default_registry",
    "error_payload",
]
