from __future__ import annotations

import asyncio
import importlib.util
import json
from collections.abc import Mapping
from typing import Any

from campus_agent.mcp.schemas import ToolEnvelope


TOOL_NAMES = (
    "campus_knowledge_search",
    "secondhand_book_search",
    "student_schedule_query",
    "course_review_search",
    "course_offering_search",
)


class MCPDependencyError(RuntimeError):
    """Raised when the official MCP SDK is not installed."""


class MCPToolError(RuntimeError):
    """Raised when an MCP tool call fails or violates the shared result contract."""


def dependency_available() -> bool:
    return importlib.util.find_spec("mcp") is not None


class MCPToolClient:
    """Synchronous application adapter over the SDK's in-memory MCP Client."""

    def __init__(self, server: object) -> None:
        self.server = server

    def list_tools(self) -> list[dict[str, Any]]:
        return _run_sync(self._list_tools())

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> ToolEnvelope:
        if name not in TOOL_NAMES:
            raise MCPToolError(f"不允许调用未知工具：{name}。")
        safe_arguments = dict(arguments)
        return _run_sync(self._call_tool(name, safe_arguments))

    async def _list_tools(self) -> list[dict[str, Any]]:
        Client = _client_class()
        async with Client(self.server) as client:
            result = await client.list_tools()
        return [
            {
                "name": tool.name,
                "title": getattr(tool, "title", None),
                "description": tool.description,
                "input_schema": tool.input_schema,
                "output_schema": getattr(tool, "output_schema", None),
            }
            for tool in result.tools
        ]

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> ToolEnvelope:
        Client = _client_class()
        async with Client(self.server) as client:
            result = await client.call_tool(name, arguments)
        if result.is_error:
            detail = _content_text(result.content) or "MCP 工具返回了错误。"
            raise MCPToolError(detail[:1000])
        payload = result.structured_content
        if payload is None:
            raw = _content_text(result.content)
            try:
                payload = json.loads(raw)
            except (TypeError, json.JSONDecodeError) as exc:
                raise MCPToolError("MCP 工具没有返回可解析的结构化结果。") from exc
        return _validate_envelope(payload, expected_tool=name)


def _client_class() -> type[Any]:
    try:
        from mcp import Client
    except ImportError as exc:
        raise MCPDependencyError(
            "未安装官方 MCP Python SDK，请运行 pip install -r requirements.txt。"
        ) from exc
    return Client


def _run_sync(coroutine: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    coroutine.close()
    raise MCPToolError("同步 MCP 客户端不能在正在运行的异步事件循环中调用。")


def _content_text(blocks: object) -> str:
    if not isinstance(blocks, list):
        return ""
    parts: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts).strip()


def _validate_envelope(payload: object, *, expected_tool: str) -> ToolEnvelope:
    if not isinstance(payload, dict):
        raise MCPToolError("MCP 工具结构化结果必须是对象。")
    if payload.get("tool") != expected_tool:
        raise MCPToolError("MCP 工具结果与调用的工具名称不一致。")
    ok = payload.get("ok")
    data_mode = payload.get("data_mode")
    items = payload.get("items")
    message = payload.get("message")
    warnings = payload.get("warnings")
    if not isinstance(ok, bool):
        raise MCPToolError("MCP 工具结果缺少布尔字段 ok。")
    if not isinstance(data_mode, str) or not data_mode:
        raise MCPToolError("MCP 工具结果缺少 data_mode。")
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise MCPToolError("MCP 工具结果 items 必须是对象数组。")
    if not isinstance(message, str):
        raise MCPToolError("MCP 工具结果缺少 message。")
    if not isinstance(warnings, list) or not all(isinstance(item, str) for item in warnings):
        raise MCPToolError("MCP 工具结果 warnings 必须是字符串数组。")
    return {
        "ok": ok,
        "tool": expected_tool,
        "data_mode": data_mode,
        "items": items,
        "message": message,
        "warnings": warnings,
    }
