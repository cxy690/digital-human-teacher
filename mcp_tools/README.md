# mcp_tools — 工具注册中心 + MCP Server

把 OCR、知识检索、白板、数学求解等能力用**标准 MCP（Model Context Protocol）报文**暴露给任意 MCP 客户端，支持 **stdio** 与 **Streamable HTTP** 两种传输。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `__init__.py` | 包门户。 |
| `registry.py` | 工具注册中心：6 个工具——`ocr_parse` / `knowledge_search` / `whiteboard_write` / `whiteboard_clear` / `math_solve` / `math_verify`；`ToolSpec` 描述工具签名与参数。 |
| `protocol.py` | MCP 协议核心（传输无关）：JSON-RPC 2.0 方法分发——`initialize`（握手+能力协商）、`notifications/initialized`、`tools/list`、`tools/call`、`ping`；注册表 `parameters` 映射为 MCP 的 `inputSchema`。 |
| `server.py` | 入口：stdio 模式（本地子进程，供 Claude Desktop / IDE 等 MCP Host）、Streamable HTTP 模式（单端点 `POST /mcp`，由主应用自动挂载）、`--selftest` 自检。 |

## 三种用法

```bash
python -m mcp_tools.server              # stdio（Host 以子进程方式拉起）
python -m mcp_tools.server --http --port 8020   # 独立 HTTP（主应用已挂 POST /mcp，可不必再起）
python -m mcp_tools.server --selftest   # 自检：握手 + 列工具 + 调工具，不发网络请求
```

MCP Host 配置示例：

```json
{"mcpServers": {"education": {
  "command": "python", "args": ["-m", "mcp_tools.server"],
  "cwd": "<项目根目录>"}}}
```

## 设计要点

- **工具实现零改动**：协议层只做报文转换，工具本体仍在 `registry.py`，HTTP 与 stdio 共享同一个 `MCPServer`。
- **与 `a2a/rpc.py` 同构**：JSON-RPC 2.0 标准错误码两个协议层保持一致（PARSE_ERROR / INVALID_REQUEST / METHOD_NOT_FOUND / INVALID_PARAMS / INTERNAL_ERROR）。
- **协议版本协商**：支持 `2025-06-18 / 2025-03-26 / 2024-11-05`，客户端版本不在清单时回落最新版。
- **与主应用同源**：`server.py` 与 `server/app.py` 用同款引导（根目录入 `sys.path` + `os.chdir`），保证三种启动方式都能读到 `.env` 与 `data/`。
