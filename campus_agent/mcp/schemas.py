from __future__ import annotations

from typing import Any, TypedDict


DATA_MODE_DEMO = "demo"
DATA_MODE_KNOWLEDGE = "local_knowledge_base"


class ToolEnvelope(TypedDict):
    """Uniform structured result returned by every campus MCP tool."""

    ok: bool
    tool: str
    data_mode: str
    items: list[dict[str, Any]]
    message: str
    warnings: list[str]


def envelope(
    *,
    ok: bool,
    tool: str,
    data_mode: str,
    items: list[dict[str, Any]] | None = None,
    message: str,
    warnings: list[str] | None = None,
) -> ToolEnvelope:
    return {
        "ok": ok,
        "tool": tool,
        "data_mode": data_mode,
        "items": items or [],
        "message": message,
        "warnings": warnings or [],
    }


DEMO_WARNING = "当前结果来自模拟数据，仅用于第三阶段功能演示，不代表真实校园服务数据。"
