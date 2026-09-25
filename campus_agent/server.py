from __future__ import annotations

import ipaddress
import json
import logging
import mimetypes
import re
import socket
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from campus_agent.agent import ChatService
from campus_agent.config import AppConfig
from campus_agent.models import ChatStreamPlan


LOGGER = logging.getLogger("campus_agent.server")
MAX_REQUEST_BYTES = 1_000_000
WEB_CITATION = re.compile(r"[ \t]*\[来源\d+\]")
WEB_CITATION_FULL = re.compile(r"\[来源\d+\]")


class _WebCitationStripper:
    """Remove citations even when a model splits them across stream chunks."""

    def __init__(self) -> None:
        self.pending = ""

    def feed(self, text: str) -> str:
        output: list[str] = []
        for character in text:
            if not self.pending:
                if character == "[":
                    self.pending = character
                else:
                    output.append(character)
                continue
            self.pending += character
            if character == "]":
                if not WEB_CITATION_FULL.fullmatch(self.pending):
                    output.append(self.pending)
                self.pending = ""
            elif len(self.pending) > 24 or character in "\r\n":
                output.append(self.pending)
                self.pending = ""
        return "".join(output)

    def flush(self) -> str:
        remainder = self.pending
        self.pending = ""
        return remainder

    def reset(self) -> None:
        self.pending = ""


class CampusHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler], app: ChatService):
        self.address_family = socket.AF_INET6 if ":" in address[0] else socket.AF_INET
        super().__init__(address, handler)
        self.app = app


class CampusRequestHandler(BaseHTTPRequestHandler):
    server_version = "CampusAgent/0.5"
    protocol_version = "HTTP/1.1"

    @property
    def app(self) -> ChatService:
        server = self.server
        assert isinstance(server, CampusHTTPServer)
        return server.app

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlsplit(self.path)
        if parsed.path == "/api/health":
            self._json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "stage": [1, 2, 3, 4, 5],
                    "request_timeout_seconds": max(60, self.app.config.llm_timeout_seconds + 5),
                    "planning": {"configured": True, "mode": "bounded-multi-agent",
                                 "tasks": ["course_recommendation", "book_comparison"], "max_tool_calls": 6},
                    "multi_agent": {
                        "configured": True,
                        "roles": ["planner", "executor", "reviewer"],
                        "reviewer_fail_closed": True,
                    },
                    "provider": self.app.config.provider_status,
                    "knowledge": self.app.knowledge_base.status(),
                    "tooling": self.app.tooling_status(),
                },
            )
            return
        if parsed.path == "/api/knowledge/search":
            query = (parse_qs(parsed.query).get("q") or [""])[0].strip()
            if not query:
                self._error(HTTPStatus.BAD_REQUEST, "查询参数 q 不能为空。")
                return
            hits = self.app.knowledge_base.search(query)
            self._json(
                HTTPStatus.OK,
                {
                    "query": query,
                    "count": len(hits),
                    "results": [hit.as_source(index) for index, hit in enumerate(hits, start=1)],
                },
            )
            return
        self._static(parsed.path)

    def do_HEAD(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        if parsed.path == "/api/health":
            self._json(HTTPStatus.OK, {}, head_only=True)
            return
        self._static(parsed.path, head_only=True)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        try:
            payload = self._read_json()
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        try:
            if parsed.path == "/api/chat":
                message = payload.get("message")
                if not isinstance(message, str):
                    raise ValueError("字段 message 必须是字符串。")
                requested_session = payload.get("session_id")
                if requested_session is not None and not isinstance(requested_session, str):
                    raise ValueError("字段 session_id 必须是字符串。")
                plan = self.app.stream_chat(message, requested_session)
                self._chat_stream(plan)
                return
            if parsed.path == "/api/session/clear":
                session_id = payload.get("session_id")
                if not isinstance(session_id, str) or not session_id:
                    raise ValueError("字段 session_id 不能为空。")
                cleared = self.app.sessions.clear(session_id)
                self._json(HTTPStatus.OK, {"status": "cleared", "existed": cleared})
                return
            if parsed.path == "/api/knowledge/rebuild":
                if not self._is_trusted_local_request():
                    self._error(HTTPStatus.FORBIDDEN, "知识库重建仅允许从本机同源页面发起。")
                    return
                report = self.app.knowledge_base.rebuild()
                self._json(
                    HTTPStatus.OK,
                    {"status": "completed", "knowledge": report.as_dict()},
                )
                return
            self._error(HTTPStatus.NOT_FOUND, "接口不存在。")
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception:
            LOGGER.exception("Request failed: %s", parsed.path)
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "服务处理请求时发生错误，请查看服务端日志。")

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "")
        if raw_length in {"", "0"}:
            raise ValueError("JSON 请求体不能为空。")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("Content-Length 无效。") from exc
        if length < 0 or length > MAX_REQUEST_BYTES:
            raise ValueError("请求体过大。")
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise ValueError("接口只接受 application/json。")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("请求体不是有效的 UTF-8 JSON。") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON 请求体必须是对象。")
        return payload

    def _static(self, request_path: str, *, head_only: bool = False) -> None:
        web_root = (self.app.config.project_root / "web").resolve()
        relative = unquote(request_path).lstrip("/") or "index.html"
        candidate = (web_root / relative).resolve()
        try:
            candidate.relative_to(web_root)
        except ValueError:
            self._error(HTTPStatus.NOT_FOUND, "页面不存在。")
            return
        if not candidate.is_file():
            self._error(HTTPStatus.NOT_FOUND, "页面不存在。")
            return
        mime_type, _ = mimetypes.guess_type(candidate.name)
        body = candidate.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{mime_type or 'application/octet-stream'}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self._security_headers()
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _json(
        self,
        status: HTTPStatus,
        payload: dict[str, Any],
        *,
        head_only: bool = False,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._security_headers()
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _chat_stream(self, plan: ChatStreamPlan) -> None:
        self.close_connection = True
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store, no-transform")
        self.send_header("Connection", "close")
        self._security_headers()
        self.end_headers()

        stripper = _WebCitationStripper()
        try:
            self._write_stream_event(
                {
                    "type": "start",
                    "session_id": plan.session_id,
                    "mode": plan.mode,
                    "provider": plan.provider,
                    "knowledge_used": plan.knowledge_used,
                }
            )
            replaced = False
            for update in plan.updates:
                if update.kind == "delta":
                    visible = stripper.feed(update.text)
                    if visible:
                        self._write_stream_event({"type": "delta", "text": visible})
                elif update.kind == "replace":
                    replaced = True
                    stripper.reset()
                    visible = WEB_CITATION.sub("", update.text).strip()
                    self._write_stream_event({"type": "replace", "text": visible})
                elif update.kind in {"agent_start", "agent_result", "tool_start", "tool_result", "plan", "step_start", "step_result", "task_result"}:
                    self._write_stream_event(
                        {"type": update.kind, **update.payload}
                    )
            if not replaced:
                remainder = stripper.flush()
                if remainder:
                    self._write_stream_event({"type": "delta", "text": remainder})
            self._write_stream_event({"type": "done"})
        except (BrokenPipeError, ConnectionResetError):
            LOGGER.info("Client disconnected while reading chat stream")

    def _write_stream_event(self, event: dict[str, object]) -> None:
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        self.wfile.write(line.encode("utf-8"))
        self.wfile.flush()

    def _error(self, status: HTTPStatus, detail: str) -> None:
        self._json(status, {"detail": detail})

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'",
        )

    def _is_loopback_client(self) -> bool:
        try:
            return ipaddress.ip_address(self.client_address[0]).is_loopback
        except ValueError:
            return False

    @staticmethod
    def _is_loopback_host(hostname: str | None) -> bool:
        if not hostname:
            return False
        if hostname.lower() == "localhost":
            return True
        try:
            return ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            return False

    def _is_trusted_local_request(self) -> bool:
        if not self._is_loopback_client():
            return False
        host_header = self.headers.get("Host", "")
        try:
            target = urlsplit(f"//{host_header}")
        except ValueError:
            return False
        if (
            not self._is_loopback_host(target.hostname)
            or target.username is not None
            or target.password is not None
        ):
            return False
        origin_header = self.headers.get("Origin")
        if not origin_header:
            return True
        try:
            origin = urlsplit(origin_header)
        except ValueError:
            return False
        if (
            origin.scheme != "http"
            or not self._is_loopback_host(origin.hostname)
            or origin.username is not None
            or origin.password is not None
            or origin.path not in {"", "/"}
            or origin.query
            or origin.fragment
            or (origin.hostname or "").lower() != (target.hostname or "").lower()
        ):
            return False
        try:
            target_port = target.port or 80
            origin_port = origin.port or 80
        except ValueError:
            return False
        return origin_port == target_port

    def log_message(self, format: str, *args: object) -> None:
        LOGGER.info("%s - %s", self.address_string(), format % args)


def create_server(
    config: AppConfig,
    app: ChatService | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
) -> CampusHTTPServer:
    service = app or ChatService(config)
    return CampusHTTPServer((host or config.host, config.port if port is None else port), CampusRequestHandler, service)


def serve(config: AppConfig) -> None:
    server = create_server(config)
    actual_host, actual_port = server.server_address[:2]
    display_host = "127.0.0.1" if actual_host in {"0.0.0.0", "::"} else actual_host
    if ":" in display_host:
        display_host = f"[{display_host}]"
    print(f"华工校园助手已启动：http://{display_host}:{actual_port}")
    print("按 Ctrl+C 停止服务。")
    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        print("\n正在停止服务…")
    finally:
        server.server_close()
