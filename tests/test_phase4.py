from __future__ import annotations

import copy
import json
import threading
import unittest
from urllib.request import Request, urlopen

from campus_agent.agent import ChatService
from campus_agent.mcp.repositories import CampusToolRepository
from campus_agent.mcp.client import MCPToolClient, dependency_available
from campus_agent.mcp.server import create_mcp_server
from campus_agent.planning import TaskExecution, build_task_plan, filter_course_offerings, meetings_conflict
from campus_agent.server import create_server
from tests.helpers import PROJECT_ROOT, make_config


class EmptyKnowledge:
    def ensure_ready(self):
        pass

    def search(self, *_args, **_kwargs):
        return []

    def status(self):
        return {"documents": 0, "chunks": 0, "warnings": []}


class RepositoryClient:
    def __init__(self):
        self.repository = CampusToolRepository(make_config(PROJECT_ROOT), EmptyKnowledge())
        self.calls = []
        self.fail_name = ""

    def call_tool(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        if name == self.fail_name:
            return {"ok": False, "tool": name, "data_mode": "demo", "items": [], "message": "fixture failure", "warnings": []}
        return getattr(self.repository, name)(**arguments)


def service(client=None):
    client = client or RepositoryClient()
    return ChatService(make_config(PROJECT_ROOT), knowledge_base=EmptyKnowledge(), tool_client=client), client


class TaskPlanTests(unittest.TestCase):
    def test_course_plan_extracts_constraints(self):
        plan = build_task_plan("DEMO002 推荐一门大学城周二下午不冲突、评分至少4.6分、不点名、作业少的通识课")
        self.assertFalse(plan.clarification)
        self.assertEqual(plan.constraints["student_id"], "DEMO002")
        self.assertEqual(plan.constraints["min_rating"], 4.6)
        self.assertTrue(plan.constraints["no_attendance"])
        self.assertTrue(plan.constraints["low_workload"])
        self.assertEqual(plan.constraints["campus"], "大学城校区")
        self.assertEqual(len(plan.steps), 5)

    def test_high_rating_default_and_demo_are_explicit(self):
        plan = build_task_plan("推荐一个周二下午没课且评价高的通识课")
        self.assertTrue(plan.constraints["demo_student_defaulted"])
        self.assertEqual(plan.constraints["min_rating"], 4.5)
        self.assertIn("不代表", plan.constraints["time_interpretation"])

    def test_book_plan_keeps_atomic_title_and_budget(self):
        plan = build_task_plan("比较概率论与数理统计和线代二手书，总预算50元")
        self.assertEqual(plan.constraints["book_names"], ["概率论与数理统计", "线代"])
        self.assertEqual(plan.constraints["total_budget"], 50)

    def test_documented_each_one_book_request_only_queries(self):
        plan = build_task_plan("帮我买高数、线代、概统各一本，总预算80元")
        self.assertEqual(plan.constraints["book_names"], ["高数", "线代", "概统"])
        app, client = service()
        result = app.chat("帮我买高数、线代、概统各一本，总预算80元")
        self.assertEqual(result.task_status, "completed")
        self.assertEqual(len(client.calls), 3)
        self.assertIn("不下单", result.answer)

    def test_book_quoted_titles_and_per_book_limit(self):
        plan = build_task_plan("比较《高数》和《线代》二手书，每本20元以内，总预算40元")
        self.assertEqual(plan.constraints["book_names"], ["高数", "线代"])
        self.assertEqual(plan.constraints["price_limit"], 20)

    def test_invalid_or_unsupported_constraint_clarifies(self):
        for question in ("推荐评分6分的通识课", "推荐周二下午和周三晚上的通识课", "推荐2学分不考试的通识课", "比较二手书", "比较《高数》《线代》《大物》《计网》《数电》二手书"):
            with self.subTest(question=question):
                self.assertTrue(build_task_plan(question).clarification)

    def test_single_book_request_is_not_a_workflow(self):
        for question in ("有卖概率论二手书吗", "找《高等数学》二手书，30元以内", "我想买概率论与数理统计"):
            self.assertIsNone(build_task_plan(question))

    def test_avoid_time_is_not_within_time(self):
        plan = build_task_plan("推荐避开周二下午的通识课")
        self.assertEqual(plan.constraints["time_mode"], "avoid")

    def test_quoted_comparison_without_explicit_book_noun_works(self):
        plan = build_task_plan("比较《高等数学》和《线性代数》，总预算50元")
        self.assertEqual(plan.task_type, "book_comparison")
        self.assertFalse(plan.clarification)

    def test_per_budget_is_not_total_and_particles_are_cleaned(self):
        plan = build_task_plan("比较二手高数和线代吗，每本预算20元")
        self.assertIsNone(plan.constraints["total_budget"])
        self.assertEqual(plan.constraints["price_limit"], 20)
        self.assertEqual(plan.constraints["book_names"], ["高数", "线代"])

    def test_unprefixed_numeric_limit_is_explicit_total_budget(self):
        plan = build_task_plan("比较二手高数和线代，50元以内")
        self.assertEqual(plan.constraints["total_budget"], 50)
        self.assertFalse(plan.clarification)
        self.assertFalse(build_task_plan("比较二手高数和线代，总预算50元以内").clarification)

    def test_unparsed_negative_or_reverse_budget_clarifies(self):
        for suffix in ("总预算-1元", "总预算五十元", "总预算不低于30元", "每本预算二十元", "总预算50元或总预算100元", "总预算50元，60元以内", "50元以内或60元以内"):
            with self.subTest(suffix=suffix):
                self.assertTrue(build_task_plan("比较二手《高数》和《线代》，" + suffix).clarification)

    def test_unimplemented_book_campus_author_constraints_clarify(self):
        for question in ("在五山校区比较《高数》和《线代》，总预算50元", "比较二手《高数》和《线代》，作者分别是同济大学和李永乐"):
            self.assertTrue(build_task_plan(question).clarification)

    def test_other_hard_constraints_are_never_silently_ignored(self):
        for question in ("推荐下学期周三下午的通识课", "推荐第二周周三下午的通识课", "推荐五山和大学城校区的通识课", "DEMO001 DEMO002 推荐通识课", "推荐评分不高于4的通识课", "推荐评分10分的通识课", "推荐周三下午的通识课，不要作业少的", "推荐通识课并找高数二手书", "比较二手高数和线代然后查课表", "推荐两门同时选修的通识课", "推荐一门周二上午14:00-16:00不点名的通识课"):
            with self.subTest(question=question):
                self.assertTrue(build_task_plan(question).clarification)

    def test_two_independent_alternatives_and_supported_greater_equal(self):
        plan = build_task_plan("推荐评分>=4.5分的两门通识课")
        self.assertEqual(plan.constraints["limit"], 2)
        self.assertEqual(plan.constraints["min_rating"], 4.5)
        self.assertFalse(plan.clarification)

    def test_named_course_is_not_ignored_without_quotes(self):
        plan = build_task_plan("DEMO001 推荐一门电影艺术赏析课程")
        self.assertEqual(plan.constraints["course_query"], "电影艺术赏析")
        app, _ = service()
        result = app.chat("DEMO001 推荐一门电影艺术赏析课程")
        self.assertIn("电影艺术赏析", result.answer)
        self.assertNotIn("音乐鉴赏", result.answer)


class CourseFilterTests(unittest.TestCase):
    def setUp(self):
        self.client = RepositoryClient()
        repository = self.client.repository
        self.schedule = repository.student_schedule_query("DEMO001")["items"]
        self.offerings = repository.course_offering_search("通识", "五山校区")["items"]
        self.reviews = repository.course_review_search("通识")["items"]
        self.constraints = build_task_plan("DEMO001 推荐周二下午评分高且不点名的通识课").constraints

    def filter(self, offerings=None, reviews=None):
        return filter_course_offerings(self.schedule, self.offerings if offerings is None else offerings,
                                      self.reviews if reviews is None else reviews, self.constraints, "五山校区")

    def test_strict_recommendation_excludes_conflict_and_unknowns(self):
        selected, reasons = self.filter()
        ids = {item["offering_id"] for item in selected}
        self.assertIn("OFFERING-DEMO-007", ids)
        self.assertNotIn("OFFERING-DEMO-008", ids)
        self.assertNotIn("OFFERING-DEMO-011", ids)
        self.assertNotIn("OFFERING-DEMO-012", ids)
        self.assertTrue(reasons)

    def test_occasional_attendance_is_not_no_attendance(self):
        offering = copy.deepcopy(next(item for item in self.offerings if item["offering_id"] == "OFFERING-DEMO-007"))
        review = copy.deepcopy(next(item for item in self.reviews if item["course_id"] == offering["course_id"]))
        review["attendance_requirement"] = "occasional"
        selected, reasons = self.filter([offering], [review])
        self.assertFalse(selected)
        self.assertIn("非明确不点名", reasons)

    def test_one_conflicting_meeting_excludes_entire_offering(self):
        offering = copy.deepcopy(next(item for item in self.offerings if item["offering_id"] == "OFFERING-DEMO-007"))
        offering["meetings"].append({"weekday": "周二", "start_time": "16:00", "end_time": "17:00", "week_start": 1, "week_end": 16})
        self.assertFalse(self.filter([offering])[0])

    def test_missing_review_time_location_or_wrong_semester_excluded(self):
        original = next(item for item in self.offerings if item["offering_id"] == "OFFERING-DEMO-007")
        for field, value in (("meetings", []), ("meetings", [{}]), ("location", ""), ("semester", "2025-2026-1"), ("campus", "大学城校区"), ("course_id", "missing")):
            with self.subTest(field=field, value=value):
                offering = copy.deepcopy(original)
                offering[field] = value
                self.assertFalse(self.filter([offering])[0])

    def test_missing_or_invalid_rating_never_matches(self):
        offering = next(item for item in self.offerings if item["offering_id"] == "OFFERING-DEMO-007")
        original = next(item for item in self.reviews if item["course_id"] == offering["course_id"])
        for value in (None, float("nan"), True, 6, -1):
            with self.subTest(value=value):
                review = {**original, "rating": value}
                self.assertFalse(self.filter([offering], [review])[0])

    def test_conflicts_use_half_open_times_and_week_ranges(self):
        base = {"weekday": "周二", "start_time": "14:00", "end_time": "15:35", "week_start": 1, "week_end": 8}
        self.assertFalse(meetings_conflict(base, {**base, "start_time": "15:35", "end_time": "16:00"}))
        self.assertFalse(meetings_conflict(base, {**base, "week_start": 9, "week_end": 16}))
        self.assertTrue(meetings_conflict(base, {**base, "week_start": 8, "week_end": 16}))

    def test_malformed_student_schedule_fails_not_free_time(self):
        self.schedule[0]["weeks"] = "未知"
        with self.assertRaisesRegex(Exception, "课表周次"):
            self.filter()


class WorkflowIntegrationTests(unittest.TestCase):
    def test_course_three_tools_then_recommendation(self):
        app, client = service()
        result = app.chat("DEMO001 推荐一门周二下午评分高、不点名的通识课")
        self.assertEqual(result.task_status, "completed")
        self.assertEqual([name for name, _ in client.calls], ["student_schedule_query", "course_offering_search", "course_review_search"])
        self.assertIn("OFFERING-DEMO-007", result.answer)
        self.assertNotIn("OFFERING-DEMO-008", result.answer)
        self.assertIn("模拟数据", result.answer)
        self.assertTrue(all(step["status"] == "completed" for step in result.task_plan["steps"]))

    def test_demo002_uses_own_campus_and_conflicts(self):
        app, _ = service()
        result = app.chat("DEMO002 推荐一门周二下午评分高、不点名的通识课")
        self.assertEqual(result.task_status, "completed")
        self.assertIn("OFFERING-DEMO-009", result.answer)
        self.assertNotIn("OFFERING-DEMO-010", result.answer)
        self.assertIn("大学城校区", result.answer)

    def test_default_demo_declared_and_no_result_does_not_relax(self):
        app, _ = service()
        result = app.chat("推荐周二下午评分至少5分且不点名的通识课")
        self.assertEqual(result.task_status, "no_result")
        self.assertIn("默认 DEMO001", result.answer)
        self.assertIn("没有找到", result.answer)

    def test_unknown_student_stops_after_schedule(self):
        app, client = service()
        result = app.chat("DEMO999 推荐周二下午评分高的通识课")
        self.assertEqual(result.mode, "task_error")
        self.assertEqual(len(client.calls), 1)
        self.assertTrue(any(step["status"] == "skipped" for step in result.task_plan["steps"]))

    def test_tool_failure_stops_without_partial_recommendation(self):
        client = RepositoryClient()
        client.fail_name = "course_review_search"
        app, _ = service(client)
        result = app.chat("DEMO001 推荐周二下午评分高的通识课")
        self.assertEqual(result.task_status, "failed")
        self.assertIn("任务没有完成", result.answer)
        self.assertNotIn("摄影与视觉表达", result.answer)
        self.assertEqual(result.tool_calls[-1]["status"], "failed")

    def test_book_comparison_and_total_budget(self):
        app, client = service()
        result = app.chat("比较高数和线代二手书，总预算50元")
        self.assertEqual(result.task_status, "completed")
        self.assertEqual(len(client.calls), 2)
        self.assertIn("最低组合总价", result.answer)
        self.assertIn("符合预算", result.answer)

    def test_book_over_budget_is_no_result(self):
        app, _ = service()
        result = app.chat("比较《高数》和《线代》二手书，总预算1元")
        self.assertEqual(result.task_status, "no_result")
        self.assertIn("没有符合预算的组合", result.answer)

    def test_book_missing_title_does_not_complete_combination(self):
        app, _ = service()
        result = app.chat("比较《高数》和《完全不存在的测试书》二手书，总预算50元")
        self.assertEqual(result.task_status, "no_result")
        self.assertIn("没有找到完整组合", result.answer)

    def test_duplicate_book_alias_does_not_buy_twice(self):
        app, client = service()
        result = app.chat("比较《高数》和《高等数学》二手书")
        self.assertEqual(result.task_status, "clarification")
        self.assertFalse(client.calls)

    def test_invalid_plan_clarification_without_tool_calls(self):
        app, client = service()
        result = app.chat("推荐周二下午和周三晚上的通识课")
        self.assertEqual(result.mode, "clarification")
        self.assertFalse(client.calls)

    def test_phase_five_executes_but_real_student_id_is_guarded_before_calls(self):
        app, client = service()
        result = app.chat("用多Agent推荐一门通识课")
        self.assertEqual(result.mode, "multi_agent")
        self.assertEqual([run["agent_id"] for run in result.agent_runs], ["planner", "executor", "reviewer"])
        self.assertEqual(len(client.calls), 3)

        app, client = service()
        result = app.chat("123456789012 推荐周二下午通识课")
        self.assertEqual(result.mode, "sensitive_input_blocked")
        self.assertFalse(client.calls)

    def test_plan_and_tool_start_are_lazy_before_actual_call(self):
        app, client = service()
        plan = app.stream_chat("DEMO001 推荐周二下午评分高的通识课", "lazy-test")
        self.assertFalse(client.calls)
        updates = iter(plan.updates)
        self.assertEqual(next(updates).kind, "agent_start")
        self.assertEqual(next(updates).kind, "agent_result")
        self.assertEqual(next(updates).kind, "plan")
        self.assertEqual(next(updates).kind, "agent_start")
        self.assertFalse(client.calls)
        self.assertEqual(next(updates).kind, "step_start")
        self.assertEqual(next(updates).kind, "tool_start")
        self.assertFalse(client.calls)
        rest = list(updates)
        self.assertEqual(len(client.calls), 3)
        self.assertTrue(any(update.kind == "task_result" and update.payload["status"] == "completed" for update in rest))
        self.assertTrue(app.sessions.history("lazy-test"))

    def test_cancelled_workflow_does_not_save_half_result(self):
        app, _ = service()
        plan = app.stream_chat("比较高数和线代二手书，总预算50元", "cancel-test")
        updates = iter(plan.updates)
        next(updates)
        next(updates)
        updates.close()
        self.assertFalse(app.sessions.history("cancel-test"))

    def test_stream_privacy_guard_does_not_emit_plan(self):
        app, client = service()
        plan = app.stream_chat("123456789012 推荐周二下午通识课")
        updates = list(plan.updates)
        self.assertEqual(plan.mode, "sensitive_input_blocked")
        self.assertFalse(client.calls)
        self.assertFalse(any(update.kind == "plan" for update in updates))

    def test_unsupported_cross_tool_request_does_not_do_partial_work(self):
        app, client = service()
        result = app.chat("查询DEMO001课表然后查询课程评价")
        self.assertEqual(result.mode, "not_implemented")
        self.assertFalse(client.calls)

    @unittest.skipUnless(dependency_available(), "official MCP SDK is not installed")
    def test_complete_workflows_use_official_sdk(self):
        knowledge = EmptyKnowledge()
        config = make_config(PROJECT_ROOT)
        client = MCPToolClient(create_mcp_server(config, knowledge))
        app = ChatService(config, knowledge_base=knowledge, tool_client=client)
        course = app.chat("DEMO001 推荐一门周二下午不点名且评分高的通识课")
        self.assertEqual(course.task_status, "completed")
        self.assertEqual(len(course.tool_calls), 3)
        self.assertIn("OFFERING-DEMO-007", course.answer)
        books = app.chat("比较《高数》和《线代》二手书，总预算50元")
        self.assertEqual(books.task_status, "completed")
        self.assertEqual(len(books.tool_calls), 2)


class PhaseFourHTTPTests(unittest.TestCase):
    def test_http_emits_plan_steps_tools_and_final_status(self):
        app, client = service()
        server = create_server(app.config, app=app, host="127.0.0.1", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}"
            body = json.dumps({"message": "DEMO001 推荐周二下午评分高、不点名的通识课", "session_id": "http-plan"}).encode()
            request = Request(url + "/api/chat", data=body, headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=5) as response:
                events = [json.loads(line) for line in response]
            types = [event["type"] for event in events]
            self.assertEqual(types[:4], ["start", "agent_start", "agent_result", "plan"])
            self.assertEqual(
                [event["agent_id"] for event in events if event["type"] == "agent_result"],
                ["planner", "executor", "reviewer"],
            )
            self.assertEqual(types.count("tool_result"), 3)
            self.assertEqual(types[-1], "done")
            self.assertTrue(any(event["type"] == "task_result" and event["status"] == "completed" for event in events))
            self.assertIn("OFFERING-DEMO-007", "".join(event.get("text", "") for event in events))
            self.assertTrue(all("items" not in event for event in events))
            with urlopen(url + "/api/health", timeout=5) as response:
                health = json.load(response)
            self.assertEqual(health["stage"], [1, 2, 3, 4, 5])
            self.assertEqual(health["tooling"]["tool_count"], 5)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
