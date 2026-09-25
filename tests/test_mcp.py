from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from campus_agent.mcp.client import MCPToolClient, dependency_available
from campus_agent.mcp.repositories import CampusToolRepository
from campus_agent.rag.service import KnowledgeBase
from tests.helpers import FakeEmbeddings, PROJECT_ROOT, make_config


class CampusToolRepositoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        cls.config = make_config(
            PROJECT_ROOT,
            source_dir=PROJECT_ROOT / "data" / "knowledge" / "source",
            index_file=root / "index.json",
        )
        cls.knowledge = KnowledgeBase(cls.config, embeddings=FakeEmbeddings())
        cls.knowledge.ensure_ready()
        cls.repository = CampusToolRepository(cls.config, cls.knowledge)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_secondhand_search_filters_price_and_hides_contact_data(self) -> None:
        result = self.repository.secondhand_book_search("高等数学", 30)

        self.assertTrue(result["ok"])
        self.assertEqual(result["data_mode"], "demo")
        self.assertTrue(result["items"])
        self.assertTrue(all(item["price"] <= 30 for item in result["items"]))
        self.assertTrue(all("contact" not in item for item in result["items"]))

    def test_secondhand_demo_contains_exactly_100_safe_records(self) -> None:
        payload = json.loads(
            (PROJECT_ROOT / "data" / "tools" / "secondhand_books.json").read_text(
                encoding="utf-8"
            )
        )
        records = payload["books"]
        listing_ids = [record["listing_id"] for record in records]

        self.assertEqual(len(records), 100)
        self.assertEqual(len(set(listing_ids)), 100)
        self.assertTrue(
            all(record["seller_alias"].startswith("演示卖家") for record in records)
        )
        self.assertTrue(all("contact" not in record for record in records))

    def test_probability_book_exists_and_is_searchable(self) -> None:
        result = self.repository.secondhand_book_search("概率论")

        self.assertTrue(result["ok"])
        self.assertTrue(
            any(item["listing_id"] == "BOOK-DEMO-005" for item in result["items"])
        )
        self.assertTrue(
            any(item["book"] == "概率论与数理统计" for item in result["items"])
        )

    def test_common_course_book_abbreviations_are_searchable(self) -> None:
        cases = {
            "高数": "高等数学",
            "线代": "线性代数",
            "概统": "概率论与数理统计",
            "大物": "大学物理",
            "计网": "计算机网络",
            "计组": "计算机组成原理",
            "模电": "模拟电子技术基础",
            "数电": "数字电子技术基础",
            "马原": "马克思主义基本原理",
            "毛概": "毛泽东思想和中国特色社会主义理论体系概论",
        }

        for query, expected_title in cases.items():
            with self.subTest(query=query):
                result = self.repository.secondhand_book_search(query)
                self.assertTrue(result["items"])
                self.assertTrue(
                    any(expected_title in item["book"] for item in result["items"])
                )

    def test_schedule_only_accepts_demo_aliases(self) -> None:
        rejected = self.repository.student_schedule_query("202612345678", "周二")
        accepted = self.repository.student_schedule_query("DEMO001", "周二")

        self.assertFalse(rejected["ok"])
        self.assertIn("真实学号", rejected["message"])
        self.assertTrue(accepted["ok"])
        self.assertTrue(all(item["weekday"] == "周二" for item in accepted["items"]))

    def test_course_review_is_aggregate_demo_data(self) -> None:
        result = self.repository.course_review_search("电影艺术赏析")

        self.assertEqual(len(result["items"]), 1)
        self.assertEqual(result["items"][0]["rating"], 4.7)
        self.assertIn("模拟", result["warnings"][0])

    def test_knowledge_tool_uses_existing_local_rag(self) -> None:
        result = self.repository.campus_knowledge_search("公共自习室允许占座吗？", 2)

        self.assertTrue(result["ok"])
        self.assertEqual(result["data_mode"], "local_knowledge_base")
        self.assertTrue(result["items"])


@unittest.skipUnless(dependency_available(), "official MCP SDK is not installed")
class MCPProtocolIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from campus_agent.mcp.server import create_mcp_server

        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        config = make_config(
            PROJECT_ROOT,
            source_dir=PROJECT_ROOT / "data" / "knowledge" / "source",
            index_file=root / "index.json",
        )
        knowledge = KnowledgeBase(config, embeddings=FakeEmbeddings())
        knowledge.ensure_ready()
        cls.client = MCPToolClient(create_mcp_server(config, knowledge))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_server_advertises_five_whitelisted_tools(self) -> None:
        tools = self.client.list_tools()

        self.assertEqual(
            {tool["name"] for tool in tools},
            {
                "campus_knowledge_search",
                "secondhand_book_search",
                "student_schedule_query",
                "course_review_search",
                "course_offering_search",
            },
        )
        self.assertTrue(all(tool["input_schema"] for tool in tools))

    def test_official_client_searches_course_offerings(self) -> None:
        result = self.client.call_tool(
            "course_offering_search", {"course_name": "摄影", "campus": "五山校区"}
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "course_offering_search")
        self.assertEqual(len(result["items"]), 2)
        self.assertTrue(all(item["course_id"] == "COURSE-DEMO-007" for item in result["items"]))

    def test_client_calls_tool_and_receives_structured_envelope(self) -> None:
        result = self.client.call_tool(
            "secondhand_book_search",
            {"book_name": "高等数学", "price_limit": 30},
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "secondhand_book_search")
        self.assertGreater(len(result["items"]), 0)

    def test_official_client_finds_probability_book_in_demo_catalog(self) -> None:
        result = self.client.call_tool(
            "secondhand_book_search",
            {"book_name": "概率论"},
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "secondhand_book_search")
        matching_items = [
            item
            for item in result["items"]
            if item["listing_id"] == "BOOK-DEMO-005"
        ]
        self.assertEqual(len(matching_items), 1)
        self.assertEqual(matching_items[0]["book"], "概率论与数理统计")
        self.assertEqual(matching_items[0]["price"], 20)


if __name__ == "__main__":
    unittest.main()
