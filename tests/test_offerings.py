from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from campus_agent.mcp.client import MCPToolClient, dependency_available
from campus_agent.mcp.repositories import CampusToolRepository, ToolDataError, normalize_book_query
from campus_agent.mcp.schemas import DATA_MODE_DEMO
from campus_agent.mcp.server import _safe_tool_call, create_mcp_server
from tests.helpers import PROJECT_ROOT, make_config


class NoopKnowledge:
    def ensure_ready(self) -> None:
        pass


class OfferingRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.config = replace(make_config(PROJECT_ROOT), tool_data_dir=self.directory)
        self.repository = CampusToolRepository(self.config, NoopKnowledge())
        self.payload = json.loads(
            (PROJECT_ROOT / "data" / "tools" / "course_offerings.json").read_text(encoding="utf-8")
        )
        self._write_offerings()

    def _write_offerings(self) -> None:
        (self.directory / "course_offerings.json").write_text(
            json.dumps(self.payload, ensure_ascii=False), encoding="utf-8"
        )

    def test_demo_has_twelve_distinct_offerings_with_safe_boundary_records(self) -> None:
        result = self.repository.course_offering_search()
        self.assertTrue(result["ok"])
        self.assertEqual(len(self.payload["offerings"]), 12)
        self.assertEqual(len(result["items"]), 11)  # AI is not a general-education course.
        self.assertEqual(len({row["offering_id"] for row in self.payload["offerings"]}), 12)
        self.assertTrue(any(not row["meetings"] for row in result["items"]))
        self.assertTrue(any("尚未公布" in message for message in result["warnings"]))
        self.assertTrue(all("contact" not in row for row in result["items"]))

    def test_name_and_exact_campus_filter_preserve_each_class_meeting(self) -> None:
        result = self.repository.course_offering_search("摄影", "五山")
        self.assertEqual(len(result["items"]), 2)
        self.assertEqual(
            {item["meetings"][0]["start_time"] for item in result["items"]},
            {"14:00", "16:00"},
        )
        self.assertFalse(self.repository.course_offering_search("摄影", "大学城")["items"])

    def test_missing_student_fails_but_existing_empty_day_succeeds(self) -> None:
        (self.directory / "student_schedules.json").write_text(
            json.dumps({"students": [{"student_alias": "DEMO001", "semester": "2026-2027-1", "courses": []}]}),
            encoding="utf-8",
        )
        self.assertFalse(self.repository.student_schedule_query("DEMO999", "周二")["ok"])
        empty = self.repository.student_schedule_query("DEMO001", "周二")
        self.assertTrue(empty["ok"])
        self.assertEqual(empty["items"], [])

    def test_invalid_time_or_week_range_fails_even_outside_name_filter(self) -> None:
        cases = (
            {"start_time": "25:00"}, {"end_time": "14:00"}, {"start_time": None},
            {"week_start": 0}, {"week_end": 31}, {"week_start": True},
            {"week_start": 10, "week_end": 9}, {"weekday": "周八"},
        )
        original = copy.deepcopy(self.payload)
        for change in cases:
            with self.subTest(change=change):
                self.payload = copy.deepcopy(original)
                self.payload["offerings"][0]["meetings"][0].update(change)
                self._write_offerings()
                with self.assertRaises(ToolDataError):
                    self.repository.course_offering_search("摄影")
                safe = _safe_tool_call(
                    "course_offering_search", DATA_MODE_DEMO,
                    lambda: self.repository.course_offering_search("摄影"),
                )
                self.assertFalse(safe["ok"])
                self.assertEqual(safe["items"], [])

    def test_duplicate_ids_or_missing_required_fields_are_data_errors(self) -> None:
        self.payload["offerings"][1]["offering_id"] = self.payload["offerings"][0]["offering_id"]
        self._write_offerings()
        with self.assertRaises(ToolDataError):
            self.repository.course_offering_search()
        self.payload["offerings"][1]["offering_id"] = "unique"
        del self.payload["offerings"][0]["meetings"]
        self._write_offerings()
        with self.assertRaises(ToolDataError):
            self.repository.course_offering_search()

    def test_results_are_capped_at_thirty_without_claiming_full_coverage(self) -> None:
        prototype = self.payload["offerings"][0]
        self.payload["offerings"] = [
            dict(copy.deepcopy(prototype), offering_id=f"OFFERING-TEST-{index:03}")
            for index in range(35)
        ]
        self._write_offerings()
        result = self.repository.course_offering_search()
        self.assertEqual(len(result["items"]), 30)
        self.assertTrue(any("不代表完整" in message for message in result["warnings"]))

    def test_nonperiodic_attendance_is_not_no_attendance(self) -> None:
        config = make_config(PROJECT_ROOT)
        repository = CampusToolRepository(config, NoopKnowledge())
        music = repository.course_review_search("音乐")["items"][0]
        self.assertEqual(music["attendance_requirement"], "occasional")
        for query in ("摄影", "诗词"):
            review = repository.course_review_search(query)["items"][0]
            self.assertEqual(review["attendance_requirement"], "none")
            self.assertEqual(review["workload"], "较低")
            self.assertIn("通识", review["tags"])
        self.assertEqual(normalize_book_query("概统"), "概率论与数理统计")

    @unittest.skipUnless(dependency_available(), "official MCP SDK is not installed")
    def test_official_sdk_returns_failure_for_malformed_offering_data(self) -> None:
        self.payload["offerings"][0]["meetings"][0]["end_time"] = "99:99"
        self._write_offerings()
        client = MCPToolClient(create_mcp_server(self.config, NoopKnowledge()))
        result = client.call_tool("course_offering_search", {"course_name": "通识"})
        self.assertFalse(result["ok"])
        self.assertEqual(result["items"], [])


if __name__ == "__main__":
    unittest.main()
