from __future__ import annotations

from collections.abc import Callable
from typing import Any

from campus_agent.config import AppConfig
from campus_agent.mcp.client import MCPDependencyError
from campus_agent.mcp.repositories import CampusToolRepository
from campus_agent.mcp.schemas import DATA_MODE_DEMO, DATA_MODE_KNOWLEDGE, ToolEnvelope, envelope
from campus_agent.rag.service import KnowledgeBase


def create_mcp_server(
    config: AppConfig,
    knowledge_base: KnowledgeBase | None = None,
) -> object:
    """Build the real MCPServer that exposes all phase-three campus tools."""

    try:
        from mcp.server import MCPServer
    except ImportError as exc:
        raise MCPDependencyError(
            "未安装官方 MCP Python SDK，请运行 pip install -r requirements.txt。"
        ) from exc

    knowledge = knowledge_base or KnowledgeBase(config)
    knowledge.ensure_ready()
    repository = CampusToolRepository(config, knowledge)
    server = MCPServer("SCUT Campus Agent Tools")

    @server.tool()
    def campus_knowledge_search(query: str, limit: int = 4) -> ToolEnvelope:
        """Search the local campus knowledge base. Use for campus facts and procedures."""

        return _safe_tool_call(
            "campus_knowledge_search",
            DATA_MODE_KNOWLEDGE,
            lambda: repository.campus_knowledge_search(query, limit),
        )

    @server.tool()
    def secondhand_book_search(
        book_name: str, price_limit: float | None = None
    ) -> ToolEnvelope:
        """Search demo secondhand-book listings by title and optional maximum price."""

        return _safe_tool_call(
            "secondhand_book_search",
            DATA_MODE_DEMO,
            lambda: repository.secondhand_book_search(book_name, price_limit),
        )

    @server.tool()
    def student_schedule_query(
        student_id: str = "DEMO001", weekday: str | None = None
    ) -> ToolEnvelope:
        """Query a demo student's schedule; only anonymous IDs such as DEMO001 are accepted."""

        return _safe_tool_call(
            "student_schedule_query",
            DATA_MODE_DEMO,
            lambda: repository.student_schedule_query(student_id, weekday),
        )

    @server.tool()
    def course_review_search(course_name: str) -> ToolEnvelope:
        """Search aggregate demo course ratings and review characteristics by course name."""

        return _safe_tool_call(
            "course_review_search",
            DATA_MODE_DEMO,
            lambda: repository.course_review_search(course_name),
        )

    @server.tool()
    def course_offering_search(
        course_name: str = "通识", campus: str | None = None
    ) -> ToolEnvelope:
        """Search demo course offerings, meeting times and week ranges; never real enrollment."""

        return _safe_tool_call(
            "course_offering_search",
            DATA_MODE_DEMO,
            lambda: repository.course_offering_search(course_name, campus),
        )

    return server


def run_stdio_server(config: AppConfig) -> None:
    """Run the campus MCP server over stdio for external MCP hosts."""

    server = create_mcp_server(config)
    run = getattr(server, "run")
    run(transport="stdio")


def _safe_tool_call(
    name: str,
    data_mode: str,
    operation: Callable[[], ToolEnvelope],
) -> ToolEnvelope:
    try:
        return operation()
    except ValueError as exc:
        return envelope(
            ok=False,
            tool=name,
            data_mode=data_mode,
            message=str(exc),
        )
    except Exception:
        return envelope(
            ok=False,
            tool=name,
            data_mode=data_mode,
            message="工具数据暂时不可用，请检查本地数据文件和服务日志。",
        )
