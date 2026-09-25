from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace

from campus_agent.agent import ChatService
from campus_agent.config import AppConfig
from campus_agent.rag.service import KnowledgeBase
from campus_agent.server import serve


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="campus-agent",
        description="华工校园助手阶段 1–3 Demo",
    )
    parser.add_argument("--verbose", action="store_true", help="显示详细服务日志")
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser("serve", help="启动网页对话入口（默认命令）")
    serve_parser.add_argument("--host", help="监听地址，默认读取 APP_HOST")
    serve_parser.add_argument("--port", type=int, help="监听端口，默认读取 APP_PORT")

    ingest_parser = subparsers.add_parser("ingest", help="重建本地知识索引")
    ingest_parser.add_argument("--force", action="store_true", help="即使源文件未变也重建")

    search_parser = subparsers.add_parser("search", help="只运行知识检索，便于调试")
    search_parser.add_argument("query", help="检索问题")
    search_parser.add_argument("--limit", type=int, default=4, help="最多返回几条")

    ask_parser = subparsers.add_parser("ask", help="执行一次完整问答")
    ask_parser.add_argument("message", help="用户问题")
    ask_parser.add_argument("--json", action="store_true", help="输出结构化 JSON")

    subparsers.add_parser("chat", help="启动终端连续对话")
    subparsers.add_parser("check", help="检查模型配置与知识库状态")
    subparsers.add_parser("mcp-serve", help="通过 stdio 启动校园 MCP Server")
    subparsers.add_parser("tools-list", help="通过 MCP Client 列出校园工具")
    tool_call_parser = subparsers.add_parser("tool-call", help="通过 MCP Client 调试单个工具")
    tool_call_parser.add_argument("name", help="工具名称")
    tool_call_parser.add_argument(
        "--arguments",
        default="{}",
        help="JSON 对象参数，例如 {\"book_name\":\"高等数学\"}",
    )
    return parser


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = AppConfig.from_env()
    command = args.command or "serve"

    if command == "serve":
        host = getattr(args, "host", None)
        port = getattr(args, "port", None)
        if port is not None and not 1 <= port <= 65535:
            parser.error("--port 必须在 1 到 65535 之间")
        if host or port is not None:
            config = replace(config, host=host or config.host, port=port or config.port)
        serve(config)
        return 0

    knowledge_base = KnowledgeBase(config)
    if command == "mcp-serve":
        from campus_agent.mcp.server import run_stdio_server

        run_stdio_server(config)
        return 0
    if command in {"tools-list", "tool-call"}:
        from campus_agent.mcp.client import MCPDependencyError, MCPToolClient, MCPToolError
        from campus_agent.mcp.server import create_mcp_server

        knowledge_base.ensure_ready()
        try:
            client = MCPToolClient(create_mcp_server(config, knowledge_base))
            if command == "tools-list":
                _print_json({"tools": client.list_tools()})
                return 0
            try:
                arguments = json.loads(args.arguments)
            except json.JSONDecodeError as exc:
                parser.error(f"--arguments 不是有效 JSON：{exc}")
            if not isinstance(arguments, dict):
                parser.error("--arguments 必须是 JSON 对象")
            _print_json(client.call_tool(args.name, arguments))
            return 0
        except (MCPDependencyError, MCPToolError) as exc:
            print(f"MCP 工具调用失败：{exc}", file=sys.stderr)
            return 1
    if command == "ingest":
        report = knowledge_base.ensure_ready(force=args.force)
        _print_json(report.as_dict())
        return 0 if not any(issue.severity == "error" for issue in report.issues) else 1
    if command == "search":
        knowledge_base.ensure_ready()
        hits = knowledge_base.search(args.query, top_k=max(1, min(args.limit, 20)))
        _print_json(
            {
                "query": args.query,
                "count": len(hits),
                "results": [hit.as_source(index) for index, hit in enumerate(hits, start=1)],
            }
        )
        return 0
    if command == "check":
        knowledge_base.ensure_ready()
        _print_json(
            {
                "provider": config.provider_status,
                "knowledge": knowledge_base.status(),
                "tooling": ChatService(
                    config,
                    knowledge_base=knowledge_base,
                    llm_client=None,
                ).tooling_status(),
                "project_root": str(config.project_root),
            }
        )
        return 0

    service = ChatService(config, knowledge_base=knowledge_base)
    if command == "ask":
        result = service.chat(args.message)
        if args.json:
            _print_json(result.as_dict())
        else:
            print(result.answer)
            if result.sources:
                print("\n来源：")
                for source in result.sources:
                    print(f"- [{source['id']}] {source['title']} {source.get('source_url') or source['path']}")
        return 0
    if command == "chat":
        print("华工校园助手终端对话（输入 :quit 退出，:clear 清空会话）")
        session_id: str | None = None
        while True:
            try:
                message = input("\n你：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见。")
                break
            if message == ":quit":
                break
            if message == ":clear":
                if session_id:
                    service.sessions.clear(session_id)
                session_id = None
                print("会话已清空。")
                continue
            if not message:
                continue
            try:
                result = service.chat(message, session_id)
            except ValueError as exc:
                print(f"输入错误：{exc}")
                continue
            session_id = result.session_id
            print(f"助手：{result.answer}")
        return 0
    parser.error(f"未知命令：{command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
