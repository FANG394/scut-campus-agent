from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from campus_agent.agent import ChatService
from campus_agent.llm import LLMResult
from campus_agent.rag.service import KnowledgeBase
from tests.helpers import FakeEmbeddings, PROJECT_ROOT, make_config


class FakeLLM:
    def __init__(self, text: str) -> None:
        self.text = text

    def generate(self, _instructions: str, _messages: list[object]) -> LLMResult:
        return LLMResult(text=self.text, provider="fake", model="test-model")


class FakeStreamingLLM:
    provider = "fake-stream"
    model_name = "test-model"

    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.generate_called = False

    def generate(self, *_args: object) -> LLMResult:
        self.generate_called = True
        raise AssertionError("streaming path must not call generate()")

    def stream(self, *_args: object):
        yield from self.chunks


class FakeToolClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, object]]] = []

    def call_tool(self, name: str, arguments: dict[str, object]):
        self.calls.append((name, arguments))
        return self.response


class CatalogToolClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.books = [
            {
                "listing_id": "BOOK-DEMO-001",
                "book": "高等数学（上册）",
                "edition": "同济大学第七版",
                "author": "示例作者",
                "price": 18,
                "condition": "八成新",
                "campus": "五山校区",
                "pickup": "图书馆门前",
                "seller_alias": "演示卖家A",
                "posted_at": "2026-08-28",
            },
            {
                "listing_id": "BOOK-DEMO-005",
                "book": "概率论与数理统计",
                "edition": "浙大第五版",
                "author": "盛骤等",
                "price": 20,
                "condition": "八成新",
                "campus": "大学城校区",
                "pickup": "教学区公共区域",
                "seller_alias": "演示卖家E",
                "posted_at": "2026-08-25",
            },
        ]

    def call_tool(self, name: str, arguments: dict[str, object]):
        self.calls.append((name, dict(arguments)))
        if name != "secondhand_book_search":
            raise AssertionError(f"unexpected tool: {name}")
        query = str(arguments.get("book_name", "")).casefold()
        price_limit = arguments.get("price_limit")
        matches = [
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
            "items": matches,
            "message": (
                f"找到 {len(matches)} 本符合“{arguments.get('book_name')}”的模拟二手书。"
                if matches
                else f"没有找到符合“{arguments.get('book_name')}”的模拟二手书。"
            ),
            "warnings": ["当前返回的是模拟数据，仅用于演示。"],
        }


class NoCallLLM:
    def __init__(self) -> None:
        self.generate_calls = 0
        self.stream_calls = 0

    def generate(self, *_args: object) -> LLMResult:
        self.generate_calls += 1
        raise AssertionError("clarification guard must not call generate()")

    def stream(self, *_args: object):
        self.stream_calls += 1
        raise AssertionError("clarification guard must not call stream()")
        yield ""  # pragma: no cover


class CountingStreamingLLM:
    provider = "counting"
    model_name = "test-model"

    def __init__(self) -> None:
        self.generate_calls = 0
        self.stream_calls = 0

    def generate(self, *_args: object) -> LLMResult:
        self.generate_calls += 1
        return LLMResult(text="需要回退到结构化工具结果", provider="counting", model="test-model")

    def stream(self, *_args: object):
        self.stream_calls += 1
        yield "不应进入普通模型流"


class ChatServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        temporary_root = Path(cls.temporary.name)
        cls.config = make_config(
            PROJECT_ROOT,
            source_dir=PROJECT_ROOT / "data" / "knowledge" / "source",
            index_file=temporary_root / "index.json",
        )
        cls.knowledge = KnowledgeBase(cls.config, embeddings=FakeEmbeddings())
        cls.knowledge.ensure_ready()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def make_service(
        self,
        llm: object | None = None,
        tool_client: object | None = None,
    ) -> ChatService:
        return ChatService(
            self.config,
            knowledge_base=self.knowledge,
            llm_client=llm,
            tool_client=tool_client,
        )

    def test_known_campus_question_uses_knowledge_and_sources(self) -> None:
        result = self.make_service().chat("公共自习室允许占座吗？")

        self.assertEqual(result.mode, "rag")
        self.assertTrue(result.knowledge_used)
        self.assertTrue(result.sources)
        self.assertIn("[来源1]", result.answer)
        self.assertTrue(all(source["path"].endswith(".docx") for source in result.sources))
        self.assertTrue(any("本地原始文档" in warning for warning in result.warnings))

    def test_unknown_campus_fact_abstains(self) -> None:
        result = self.make_service().chat("图书馆校外访问怎么使用？")

        self.assertEqual(result.mode, "knowledge_gap")
        self.assertFalse(result.knowledge_used)
        self.assertIn("不能直接猜测", result.answer)

    def test_short_follow_up_uses_previous_user_topic(self) -> None:
        service = self.make_service()
        first = service.chat("公共自习室允许占座吗？")
        second = service.chat("校外人员呢？", first.session_id)

        self.assertEqual(second.mode, "rag")
        self.assertIn("禁止校外人员使用", second.answer)

    def test_pending_review_source_adds_reliability_warning(self) -> None:
        result = self.make_service().chat("体测当天需要携带校园卡吗？")

        self.assertEqual(result.mode, "rag")
        self.assertTrue(any("待核验" in warning for warning in result.warnings))
        self.assertEqual(result.sources[0]["verification_status"], "pending-review")

    def test_sensitive_message_is_not_saved_or_sent(self) -> None:
        service = self.make_service(FakeLLM("不应被调用"))
        result = service.chat("我的统一认证密码是 DEMO_SECRET_123，账号怎么激活？")

        self.assertEqual(result.mode, "sensitive_input_blocked")
        self.assertEqual(service.sessions.history(result.session_id), [])
        self.assertIn("不会发送给外部模型", result.answer)

    def test_unlabelled_real_student_number_is_blocked_before_schedule_tool(self) -> None:
        tool_client = FakeToolClient({})
        service = self.make_service(tool_client=tool_client)

        result = service.chat("查询 202612345678 周二的课表")

        self.assertEqual(result.mode, "sensitive_input_blocked")
        self.assertEqual(tool_client.calls, [])
        self.assertEqual(service.sessions.history(result.session_id), [])

    def test_secondhand_request_actively_calls_mcp_tool(self) -> None:
        tool_client = FakeToolClient(
            {
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
        )

        result = self.make_service(tool_client=tool_client).chat(
            "帮我找一本30元以内的高等数学二手书"
        )

        self.assertEqual(result.mode, "tool")
        self.assertEqual(tool_client.calls[0][0], "secondhand_book_search")
        self.assertEqual(tool_client.calls[0][1]["price_limit"], 30.0)
        self.assertIn("以下为模拟数据", result.answer)
        self.assertEqual(result.tool_calls[0]["status"], "completed")

    def test_screenshot_three_questions_all_use_book_tool(self) -> None:
        tool_client = CatalogToolClient()
        service = self.make_service(tool_client=tool_client)

        first = service.chat("有卖概率论二手书吗")
        second = service.chat("有卖概率论吗", first.session_id)
        third = service.chat("有卖概率论与数理统计吗", first.session_id)

        self.assertEqual(
            [call[1]["book_name"] for call in tool_client.calls],
            ["概率论", "概率论", "概率论与数理统计"],
        )
        for result in (first, second, third):
            self.assertEqual(result.mode, "tool")
            self.assertIn("概率论与数理统计", result.answer)
            self.assertIn("20 元", result.answer)
            self.assertNotIn("没有找到", result.answer)

    def test_recent_book_tool_routes_contextual_follow_up(self) -> None:
        tool_client = CatalogToolClient()
        service = self.make_service(tool_client=tool_client)
        first = service.chat("找一本概率论二手书")

        second = service.chat("那概率论呢", first.session_id)

        self.assertEqual(second.mode, "tool")
        self.assertEqual(len(tool_client.calls), 2)
        self.assertEqual(tool_client.calls[-1][1]["book_name"], "概率论")

    def test_book_context_is_available_on_third_follow_up(self) -> None:
        tool_client = CatalogToolClient()
        service = self.make_service(tool_client=tool_client)
        first = service.chat("找一本概率论二手书")
        service.chat("你好", first.session_id)
        service.chat("谢谢", first.session_id)

        follow_up = service.chat("那概率论呢", first.session_id)

        self.assertEqual(follow_up.mode, "tool")
        self.assertEqual(len(tool_client.calls), 2)

    def test_book_context_expires_before_fourth_follow_up(self) -> None:
        tool_client = CatalogToolClient()
        service = self.make_service(tool_client=tool_client)
        first = service.chat("找一本概率论二手书")
        service.chat("你好", first.session_id)
        service.chat("谢谢", first.session_id)
        service.chat("你是谁", first.session_id)

        service.chat("那概率论呢", first.session_id)

        self.assertEqual(len(tool_client.calls), 1)

    def test_failed_tool_call_does_not_seed_book_context(self) -> None:
        tool_client = FakeToolClient(
            {
                "ok": False,
                "tool": "secondhand_book_search",
                "data_mode": "demo",
                "items": [],
                "message": "模拟工具暂时不可用。",
                "warnings": ["调用失败。"],
            }
        )
        service = self.make_service(tool_client=tool_client)
        first = service.chat("找一本概率论二手书")

        service.chat("那概率论呢", first.session_id)

        self.assertEqual(first.mode, "tool_error")
        self.assertEqual(len(tool_client.calls), 1)

    def test_incomplete_transaction_clarifies_without_tool_or_model(self) -> None:
        llm = NoCallLLM()
        tool_client = CatalogToolClient()
        service = self.make_service(llm=llm, tool_client=tool_client)

        result = service.chat("有卖二手书吗")

        self.assertEqual(result.mode, "clarification")
        self.assertEqual(result.provider, "tool-routing-guard")
        self.assertIn("补充书名", result.answer)
        self.assertIn("不会猜测库存或价格", result.answer)
        self.assertEqual(tool_client.calls, [])
        self.assertEqual(llm.generate_calls, 0)
        self.assertEqual(llm.stream_calls, 0)

    def test_contextual_price_question_clarifies_without_reusing_old_title(self) -> None:
        tool_client = CatalogToolClient()
        service = self.make_service(tool_client=tool_client)
        first = service.chat("找一本概率论二手书")
        llm = NoCallLLM()
        service.llm_client = llm

        result = service.chat("多少钱", first.session_id)

        self.assertEqual(result.mode, "clarification")
        self.assertEqual(len(tool_client.calls), 1)
        self.assertEqual(llm.generate_calls, 0)
        self.assertEqual(llm.stream_calls, 0)

    def test_contextual_tool_stream_uses_tool_events_not_native_stream(self) -> None:
        tool_client = CatalogToolClient()
        service = self.make_service(tool_client=tool_client)
        first = service.chat("找一本概率论二手书")
        llm = CountingStreamingLLM()
        service.llm_client = llm

        plan = service.stream_chat("那概率论呢", first.session_id)
        updates = list(plan.updates)

        self.assertEqual(plan.mode, "tool")
        self.assertEqual(
            [update.kind for update in updates],
            ["tool_start", "tool_result", "delta"],
        )
        self.assertEqual(tool_client.calls[-1][1]["book_name"], "概率论")
        self.assertEqual(llm.stream_calls, 0)
        self.assertEqual(llm.generate_calls, 1)

    def test_transaction_clarification_stream_never_calls_model(self) -> None:
        llm = NoCallLLM()
        service = self.make_service(llm=llm, tool_client=CatalogToolClient())

        plan = service.stream_chat("有卖二手书吗")
        updates = list(plan.updates)

        self.assertEqual(plan.mode, "clarification")
        self.assertEqual([update.kind for update in updates], ["delta"])
        self.assertIn("补充书名", updates[0].text)
        self.assertEqual(llm.generate_calls, 0)
        self.assertEqual(llm.stream_calls, 0)

    def test_unrelated_short_question_does_not_inherit_book_tool(self) -> None:
        tool_client = CatalogToolClient()
        service = self.make_service(tool_client=tool_client)
        first = service.chat("找一本概率论二手书")

        service.chat("校园卡丢了", first.session_id)

        self.assertEqual(len(tool_client.calls), 1)

    def test_multi_agent_recommendation_uses_phase_five_coordinator(self) -> None:
        result = self.make_service().chat("用多Agent推荐一门周二下午没课且评分高的通识课")

        self.assertIn(result.mode, {"multi_agent", "task_error"})
        self.assertEqual(result.provider, "planner+executor+reviewer+mcp")
        self.assertEqual(
            [run["agent_id"] for run in result.agent_runs],
            ["planner", "executor", "reviewer"],
        )

    def test_tool_stream_contains_sanitised_trace_before_answer(self) -> None:
        tool_client = FakeToolClient(
            {
                "ok": True,
                "tool": "student_schedule_query",
                "data_mode": "demo",
                "items": [
                    {
                        "weekday": "周二",
                        "start_time": "08:30",
                        "end_time": "10:05",
                        "course_name": "程序设计基础",
                        "location": "实验楼（模拟）",
                        "weeks": "1-16周",
                    }
                ],
                "message": "查到 1 条模拟课程安排。",
                "warnings": ["模拟数据。"],
            }
        )

        plan = self.make_service(tool_client=tool_client).stream_chat(
            "查询 DEMO001 周二的课表"
        )
        updates = list(plan.updates)

        self.assertEqual(
            [update.kind for update in updates],
            ["tool_start", "tool_result", "delta"],
        )
        self.assertEqual(updates[0].payload["arguments"]["student_id"], "DEMO001")

    def test_ungrounded_remote_answer_falls_back_to_extract(self) -> None:
        service = self.make_service(FakeLLM("公共自习室允许校外人员使用。[来源1]"))
        result = service.chat("公共自习室允许校外人员使用吗？")

        self.assertEqual(result.provider, "local-grounding-fallback")
        self.assertIn("根据当前知识库", result.answer)
        self.assertTrue(any("没有逐字对应" in warning for warning in result.warnings))

    def test_exact_evidence_line_from_remote_model_is_accepted(self) -> None:
        hits = self.knowledge.search("公共自习室允许占座吗？")
        self.assertTrue(hits)
        evidence_line = next(
            line.strip()
            for line in hits[0].chunk.text.splitlines()
            if len(line.strip()) > 12 and not line.lstrip().startswith("#")
        )
        result = self.make_service(FakeLLM(f"{evidence_line} [来源1]")).chat(
            "公共自习室允许占座吗？"
        )

        self.assertEqual(result.provider, "fake:test-model")
        self.assertEqual(result.answer, f"{evidence_line} [来源1]")

    def test_supported_paraphrase_from_chat_model_is_accepted(self) -> None:
        result = self.make_service(
            FakeLLM("公共自习室不允许学生占用座位。[来源1]")
        ).chat("公共自习室允许占座吗？")

        self.assertEqual(result.provider, "fake:test-model")
        self.assertEqual(result.answer, "公共自习室不允许学生占用座位。[来源1]")

    def test_native_stream_is_used_and_saved_after_validation(self) -> None:
        llm = FakeStreamingLLM(
            ["公共自习室不允许学生", "占用座位。[来源1]"]
        )
        service = self.make_service(llm)

        plan = service.stream_chat("公共自习室允许占座吗？")
        updates = list(plan.updates)

        self.assertEqual([update.kind for update in updates], ["delta", "delta"])
        self.assertFalse(llm.generate_called)
        history = service.sessions.history(plan.session_id)
        self.assertEqual(history[-1].content, "公共自习室不允许学生占用座位。[来源1]")

    def test_invalid_native_stream_is_replaced_with_grounded_fallback(self) -> None:
        service = self.make_service(
            FakeStreamingLLM(["公共自习室允许校外人员使用。[来源1]"])
        )

        plan = service.stream_chat("公共自习室允许校外人员使用吗？")
        updates = list(plan.updates)

        self.assertEqual(updates[-1].kind, "replace")
        self.assertIn("根据当前知识库", updates[-1].text)


if __name__ == "__main__":
    unittest.main()
