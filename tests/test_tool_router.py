from __future__ import annotations

import unittest

from campus_agent.tool_router import (
    is_book_transaction_request,
    is_phase4_request,
    route_tool_request,
)


class ToolRouterTests(unittest.TestCase):
    def test_routes_secondhand_book_and_extracts_price(self) -> None:
        route = route_tool_request("帮我找一本30元以内的高等数学二手书。")

        self.assertIsNotNone(route)
        assert route is not None
        self.assertEqual(route.name, "secondhand_book_search")
        self.assertEqual(route.arguments["book_name"], "高等数学")
        self.assertEqual(route.arguments["price_limit"], 30.0)

    def test_routes_secondhand_book_with_trailing_price_qualifier(self) -> None:
        route = route_tool_request("帮我找一本高等数学30元以内的二手书。")

        self.assertIsNotNone(route)
        assert route is not None
        self.assertEqual(route.arguments["book_name"], "高等数学")
        self.assertEqual(route.arguments["price_limit"], 30.0)

    def test_routes_secondhand_book_with_budget_prefix(self) -> None:
        route = route_tool_request("二手书：高等数学，预算30元以内")

        self.assertIsNotNone(route)
        assert route is not None
        self.assertEqual(route.arguments["book_name"], "高等数学")
        self.assertEqual(route.arguments["price_limit"], 30.0)

    def test_quoted_book_name_keeps_title_number(self) -> None:
        route = route_tool_request("帮我找一本《高等数学2》二手书，30元以内。")

        self.assertIsNotNone(route)
        assert route is not None
        self.assertEqual(route.arguments["book_name"], "高等数学2")
        self.assertEqual(route.arguments["price_limit"], 30.0)

    def test_routes_have_for_sale_wording_without_using_particle_as_title(self) -> None:
        route = route_tool_request("有卖概率论二手书吗？")

        self.assertIsNotNone(route)
        assert route is not None
        self.assertEqual(route.name, "secondhand_book_search")
        self.assertEqual(route.arguments["book_name"], "概率论")

    def test_routes_rich_book_transaction_wording(self) -> None:
        cases = {
            "有没有卖线性代数二手书？": "线性代数",
            "有人卖大学物理吗": "大学物理",
            "大学英语在售吗": "大学英语",
            "出售数据结构": "数据结构",
            "出一本计算机网络": "计算机网络",
            "求购操作系统": "操作系统",
            "我想买离散数学": "离散数学",
            "高等数学哪里买？": "高等数学",
            "能买到大学化学吗？": "大学化学",
            "哪儿可以买到C语言教材？": "C语言",
        }

        for question, expected_name in cases.items():
            with self.subTest(question=question):
                route = route_tool_request(question)
                self.assertIsNotNone(route)
                assert route is not None
                self.assertEqual(route.name, "secondhand_book_search")
                self.assertEqual(route.arguments["book_name"], expected_name)

    def test_rejects_missing_or_particle_only_book_name(self) -> None:
        for question in (
            "二手书吗？",
            "有卖二手教材吗",
            "旧书呢",
            "求购一本书",
            "还有货吗",
            "多少钱",
            "有什么二手书吗",
            "有哪些二手教材呢",
            "有啥旧书",
        ):
            with self.subTest(question=question):
                self.assertIsNone(route_tool_request(question))
                if question != "多少钱":
                    self.assertTrue(is_book_transaction_request(question))

    def test_context_hint_still_rejects_follow_up_without_a_book_name(self) -> None:
        for question in ("还有货吗", "多少钱", "什么价", "怎么卖"):
            with self.subTest(question=question):
                self.assertIsNone(
                    route_tool_request(question, tool_hint="secondhand_book_search")
                )

    def test_routes_omitted_book_wording_with_recent_book_tool_hint(self) -> None:
        cases = {
            "概率论有吗？": "概率论",
            "那高数呢？": "高数",
            "还有数据结构吗": "数据结构",
        }

        for question, expected_name in cases.items():
            with self.subTest(question=question):
                route = route_tool_request(question, tool_hint="secondhand_book_search")
                self.assertIsNotNone(route)
                assert route is not None
                self.assertEqual(route.name, "secondhand_book_search")
                self.assertEqual(route.arguments["book_name"], expected_name)

    def test_book_hint_does_not_override_explicit_other_tool_intent(self) -> None:
        route = route_tool_request(
            "概率论这门课怎么样？", tool_hint="secondhand_book_search"
        )

        self.assertIsNotNone(route)
        assert route is not None
        self.assertEqual(route.name, "course_review_search")

    def test_contextual_book_guard_excludes_unrelated_or_other_tool_questions(self) -> None:
        self.assertTrue(
            is_book_transaction_request("那高数呢？", contextual=True)
        )
        self.assertTrue(is_book_transaction_request("多少钱", contextual=True))
        self.assertTrue(is_book_transaction_request("还有货吗", contextual=True))
        self.assertFalse(is_book_transaction_request("谢谢", contextual=True))
        self.assertFalse(is_book_transaction_request("校园卡丢了", contextual=True))
        self.assertIsNone(
            route_tool_request("校园卡丢了", tool_hint="secondhand_book_search")
        )
        self.assertIsNone(
            route_tool_request("线性代数", tool_hint="secondhand_book_search")
        )
        self.assertFalse(
            is_book_transaction_request("概率论这门课怎么样？", contextual=True)
        )
        self.assertFalse(is_book_transaction_request("查询周二课表", contextual=True))

    def test_routes_anonymous_schedule_with_weekday(self) -> None:
        route = route_tool_request("查询 DEMO002 星期二的课表")

        self.assertIsNotNone(route)
        assert route is not None
        self.assertEqual(route.name, "student_schedule_query")
        self.assertEqual(route.arguments, {"student_id": "DEMO002", "weekday": "周二"})

    def test_routes_course_review(self) -> None:
        route = route_tool_request("电影艺术赏析的课程评价怎么样？")

        self.assertIsNotNone(route)
        assert route is not None
        self.assertEqual(route.name, "course_review_search")
        self.assertEqual(route.arguments["course_name"], "电影艺术赏析")

    def test_plain_campus_question_stays_on_existing_rag_path(self) -> None:
        self.assertIsNone(route_tool_request("新生如何开通校园网？"))

    def test_multi_tool_course_recommendation_is_phase_four(self) -> None:
        question = "推荐一门周二下午没课、评分高且不点名的通识课"

        self.assertTrue(is_phase4_request(question))


if __name__ == "__main__":
    unittest.main()
