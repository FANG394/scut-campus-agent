from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path

from campus_agent.agent import ChatService
from campus_agent.rag.service import KnowledgeBase
from campus_agent.server import create_server
from tests.helpers import FakeEmbeddings, PROJECT_ROOT, make_config


class BlockingStreamingLLM:
    provider = "fake-stream"
    model_name = "test-model"

    def __init__(self) -> None:
        self.release = threading.Event()
        self.completed = False
        self.generate_called = False

    def generate(self, *_args: object) -> object:
        self.generate_called = True
        raise AssertionError("native stream must not call generate()")

    def stream(self, *_args: object):
        yield "学生不得"
        if not self.release.wait(timeout=5):
            raise RuntimeError("test stream was not released")
        yield "以任何方式占座。[来"
        yield "源1]"
        self.completed = True


class FakeToolClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def call_tool(self, name: str, arguments: dict[str, object]):
        self.calls.append((name, arguments))
        return {
            "ok": True,
            "tool": "secondhand_book_search",
            "data_mode": "demo",
            "items": [
                {
                    "book": "高等数学（上册）",
                    "edition": "第七版",
                    "price": 18,
                    "condition": "八成新",
                    "campus": "五山校区",
                    "pickup": "公共区域",
                    "seller_alias": "演示卖家A",
                }
            ],
            "message": "找到 1 本模拟二手书。",
            "warnings": ["模拟数据。"],
        }


class DynamicBookCatalogToolClient:
    """Small catalog double whose result is derived from the supplied arguments."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.books = [
            {
                "listing_id": "BOOK-DEMO-005",
                "book": "概率论与数理统计",
                "edition": "浙大第五版",
                "price": 20,
                "condition": "八成新",
                "campus": "大学城校区",
                "pickup": "教学区公共区域",
                "seller_alias": "演示卖家E",
            },
            {
                "listing_id": "BOOK-DEMO-006",
                "book": "线性代数",
                "edition": "同济第七版",
                "price": 16,
                "condition": "九成新",
                "campus": "五山校区",
                "pickup": "图书馆门前",
                "seller_alias": "演示卖家F",
            },
        ]

    def call_tool(self, name: str, arguments: dict[str, object]):
        copied_arguments = dict(arguments)
        self.calls.append((name, copied_arguments))
        if name != "secondhand_book_search":
            raise AssertionError(f"unexpected tool: {name}")

        query = str(copied_arguments.get("book_name", "")).casefold()
        price_limit = copied_arguments.get("price_limit")
        items = [
            dict(book)
            for book in self.books
            if query in str(book["book"]).casefold()
            and (
                price_limit is None
                or float(book["price"]) <= float(price_limit)
            )
        ]
        return {
            "ok": True,
            "tool": name,
            "data_mode": "demo",
            "items": items,
            "message": (
                f"找到 {len(items)} 本符合“{copied_arguments.get('book_name')}”的模拟二手书。"
                if items
                else f"没有找到符合“{copied_arguments.get('book_name')}”的模拟二手书。"
            ),
            "warnings": ["当前返回的是模拟数据，仅用于演示。"],
        }


class ServerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        temporary_root = Path(cls.temporary.name)
        config = make_config(
            PROJECT_ROOT,
            source_dir=PROJECT_ROOT / "data" / "knowledge" / "source",
            index_file=temporary_root / "index.json",
        )
        knowledge = KnowledgeBase(config, embeddings=FakeEmbeddings())
        cls.config = config
        cls.knowledge = knowledge
        app = ChatService(config, knowledge_base=knowledge)
        cls.server = create_server(config, app=app, host="127.0.0.1", port=0)
        cls.port = int(cls.server.server_address[1])
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=3)
        cls.temporary.cleanup()

    def request(
        self,
        method: str,
        path: str,
        payload: object | None = None,
        *,
        headers: dict[str, str] | None = None,
        raw_body: str | None = None,
    ) -> tuple[int, dict[str, str], object]:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        request_headers = dict(headers or {})
        body = raw_body.encode("utf-8") if raw_body is not None else None
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")
        connection.request(method, path, body=body, headers=request_headers)
        response = connection.getresponse()
        data = response.read()
        response_headers = {key.lower(): value for key, value in response.getheaders()}
        content_type = response_headers.get("content-type", "")
        decoded: object = (
            json.loads(data.decode("utf-8"))
            if "application/json" in content_type and data
            else data.decode("utf-8", errors="replace")
        )
        status = response.status
        connection.close()
        return status, response_headers, decoded

    def test_health_and_static_page(self) -> None:
        status, headers, payload = self.request("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["stage"], [1, 2, 3, 4, 5])
        self.assertEqual(payload["multi_agent"]["roles"], ["planner", "executor", "reviewer"])
        self.assertEqual(payload["knowledge"]["documents"], 4)
        self.assertEqual(payload["tooling"]["tool_count"], 5)
        self.assertEqual(headers["x-frame-options"], "DENY")

        status, headers, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["content-type"])
        self.assertIn("华工校园助手", body)
        self.assertIn("default-src 'self'", headers["content-security-policy"])

        status, _headers, app_js = self.request("GET", "/app.js")
        self.assertEqual(status, 200)
        self.assertIn("application/x-ndjson", app_js)
        self.assertIn("tool_start", app_js)
        self.assertNotIn("回答依据", app_js)

    def test_chat_endpoint_exposes_sanitised_mcp_tool_trace(self) -> None:
        tool_client = FakeToolClient()
        app = ChatService(
            self.config,
            knowledge_base=self.knowledge,
            tool_client=tool_client,
        )
        server = create_server(self.config, app=app, host="127.0.0.1", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = http.client.HTTPConnection(
            "127.0.0.1", int(server.server_address[1]), timeout=5
        )
        try:
            body = json.dumps(
                {"message": "帮我找一本30元以内的高等数学二手书"},
                ensure_ascii=False,
            ).encode("utf-8")
            connection.request(
                "POST",
                "/api/chat",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            events = [
                json.loads(line)
                for line in response.read().decode("utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(response.status, 200)
            self.assertEqual(
                [event["type"] for event in events],
                ["start", "tool_start", "tool_result", "delta", "done"],
            )
            self.assertEqual(events[1]["name"], "secondhand_book_search")
            self.assertEqual(events[1]["arguments"]["price_limit"], 30.0)
            self.assertNotIn("items", events[2])
            self.assertEqual(tool_client.calls[0][0], "secondhand_book_search")
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    def test_chat_endpoint_keeps_book_tool_context_for_screenshot_questions(self) -> None:
        tool_client = DynamicBookCatalogToolClient()
        app = ChatService(
            self.config,
            knowledge_base=self.knowledge,
            tool_client=tool_client,
        )
        server = create_server(self.config, app=app, host="127.0.0.1", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        session_id: str | None = None
        all_events: list[list[dict[str, object]]] = []
        questions = (
            "有卖概率论二手书吗",
            "有卖概率论吗",
            "有卖概率论与数理统计吗",
        )
        try:
            for question in questions:
                payload: dict[str, object] = {"message": question}
                if session_id is not None:
                    payload["session_id"] = session_id
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                connection = http.client.HTTPConnection(
                    "127.0.0.1", int(server.server_address[1]), timeout=5
                )
                try:
                    connection.request(
                        "POST",
                        "/api/chat",
                        body=body,
                        headers={"Content-Type": "application/json"},
                    )
                    response = connection.getresponse()
                    events = [
                        json.loads(line)
                        for line in response.read().decode("utf-8").splitlines()
                        if line.strip()
                    ]
                    self.assertEqual(response.status, 200)
                finally:
                    connection.close()

                self.assertEqual(
                    [event["type"] for event in events],
                    ["start", "tool_start", "tool_result", "delta", "done"],
                )
                if session_id is None:
                    session_id = str(events[0]["session_id"])
                self.assertEqual(events[0]["session_id"], session_id)
                self.assertEqual(events[1]["name"], "secondhand_book_search")
                self.assertEqual(events[2]["status"], "completed")
                self.assertEqual(events[2]["item_count"], 1)
                answer = "".join(
                    str(event.get("text", ""))
                    for event in events
                    if event["type"] == "delta"
                )
                self.assertIn("概率论与数理统计", answer)
                self.assertIn("20 元", answer)
                self.assertNotIn("没有找到", answer)
                all_events.append(events)

            expected_names = ["概率论", "概率论", "概率论与数理统计"]
            self.assertEqual(
                [arguments["book_name"] for _name, arguments in tool_client.calls],
                expected_names,
            )
            self.assertEqual(
                [events[1]["arguments"]["book_name"] for events in all_events],
                expected_names,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    def test_chat_endpoint_streams_answer_without_sources(self) -> None:
        status, headers, body = self.request(
            "POST", "/api/chat", {"message": "公共自习室允许占座吗？"}
        )

        self.assertEqual(status, 200)
        self.assertIn("application/x-ndjson", headers["content-type"])
        events = [json.loads(line) for line in body.splitlines() if line.strip()]
        self.assertEqual(events[0]["type"], "start")
        self.assertEqual(events[0]["mode"], "rag")
        self.assertTrue(events[0]["knowledge_used"])
        self.assertEqual(events[-1], {"type": "done"})
        deltas = [event["text"] for event in events if event["type"] == "delta"]
        self.assertGreaterEqual(len(deltas), 1)
        answer = "".join(deltas)
        self.assertIn("不得以任何方式占座", answer)
        self.assertNotIn("[来源", answer)
        self.assertTrue(all("sources" not in event for event in events))

    def test_native_model_stream_reaches_http_before_generation_finishes(self) -> None:
        llm = BlockingStreamingLLM()
        app = ChatService(
            self.config,
            knowledge_base=self.knowledge,
            llm_client=llm,
        )
        server = create_server(self.config, app=app, host="127.0.0.1", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = http.client.HTTPConnection(
            "127.0.0.1", int(server.server_address[1]), timeout=5
        )
        try:
            body = json.dumps(
                {"message": "公共自习室允许占座吗？"}, ensure_ascii=False
            ).encode("utf-8")
            connection.request(
                "POST",
                "/api/chat",
                body=body,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/x-ndjson",
                },
            )
            response = connection.getresponse()
            start = json.loads(response.readline().decode("utf-8"))
            first_delta = json.loads(response.readline().decode("utf-8"))

            self.assertEqual(start["type"], "start")
            self.assertEqual(first_delta, {"type": "delta", "text": "学生不得"})
            self.assertFalse(llm.completed)
            self.assertFalse(llm.generate_called)

            llm.release.set()
            remaining = [
                json.loads(line)
                for line in response.read().decode("utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(llm.completed)
            self.assertEqual(remaining[-1], {"type": "done"})
            self.assertNotIn("[来源", "".join(
                event.get("text", "") for event in [first_delta, *remaining]
            ))
        finally:
            llm.release.set()
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    def test_search_endpoint_requires_query(self) -> None:
        status, _headers, payload = self.request("GET", "/api/knowledge/search")
        self.assertEqual(status, 400)
        self.assertIn("q 不能为空", payload["detail"])

        status, _headers, payload = self.request(
            "GET", "/api/knowledge/search?q=" + "%E4%BD%93%E6%B5%8B"
        )
        self.assertEqual(status, 200)
        self.assertGreater(payload["count"], 0)

    def test_rebuild_rejects_cross_origin_and_accepts_exact_local_origin(self) -> None:
        cross_origin = f"http://localhost:{self.port}"
        status, _headers, payload = self.request(
            "POST",
            "/api/knowledge/rebuild",
            {},
            headers={"Origin": cross_origin},
        )
        self.assertEqual(status, 403)
        self.assertIn("同源", payload["detail"])

        same_origin = f"http://127.0.0.1:{self.port}"
        status, _headers, payload = self.request(
            "POST",
            "/api/knowledge/rebuild",
            {},
            headers={"Origin": same_origin},
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "completed")

    def test_post_requires_nonempty_json_object(self) -> None:
        status, _headers, payload = self.request(
            "POST",
            "/api/chat",
            headers={"Content-Type": "application/json"},
            raw_body="",
        )
        self.assertEqual(status, 400)
        self.assertIn("不能为空", payload["detail"])

        status, _headers, payload = self.request("POST", "/api/chat", ["not", "object"])
        self.assertEqual(status, 400)
        self.assertIn("必须是对象", payload["detail"])


if __name__ == "__main__":
    unittest.main()
