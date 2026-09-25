from __future__ import annotations

import unittest

from campus_agent.sessions import SessionStore


class SessionStoreToolContextTests(unittest.TestCase):
    def test_recent_tool_context_keeps_only_tool_name(self) -> None:
        store = SessionStore()
        session_id = store.normalise_id(None)

        store.append_exchange(
            session_id,
            "找一本概率论二手书",
            "找到一本模拟二手书。",
            tool_name="secondhand_book_search",
        )

        self.assertEqual(
            store.recent_tool_name(session_id), "secondhand_book_search"
        )

    def test_recent_tool_context_expires_after_three_intervening_turns(self) -> None:
        store = SessionStore()
        session_id = store.normalise_id(None)
        store.append_exchange(
            session_id,
            "找一本概率论二手书",
            "找到一本模拟二手书。",
            tool_name="secondhand_book_search",
        )
        for index in range(3):
            store.append_exchange(session_id, f"其他问题 {index}", "普通回答")

        self.assertIsNone(
            store.recent_tool_name(session_id, max_intervening_turns=2)
        )

    def test_clear_removes_tool_context_with_session(self) -> None:
        store = SessionStore()
        session_id = store.normalise_id(None)
        store.append_exchange(
            session_id,
            "找一本概率论二手书",
            "找到一本模拟二手书。",
            tool_name="secondhand_book_search",
        )

        self.assertTrue(store.clear(session_id))
        self.assertIsNone(store.recent_tool_name(session_id))


if __name__ == "__main__":
    unittest.main()
