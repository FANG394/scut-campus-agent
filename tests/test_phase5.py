from __future__ import annotations

import copy
import json
import threading
import unittest
from urllib.request import Request, urlopen

from campus_agent.agent import ChatService
from campus_agent.mcp.repositories import CampusToolRepository
from campus_agent.multi_agent import (
    ExecutorAgent,
    MultiAgentCoordinator,
    PlannerAgent,
    ReviewerAgent,
    validate_plan,
)
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
    """Repository-backed tool double kept independent from test_phase4."""

    def __init__(self):
        self.repository = CampusToolRepository(make_config(PROJECT_ROOT), EmptyKnowledge())
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.fail_name = ""

    def call_tool(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        if name == self.fail_name:
            return {
                "ok": False,
                "tool": name,
                "data_mode": "demo",
                "items": [],
                "message": "fixture failure",
                "warnings": [],
            }
        return getattr(self.repository, name)(**arguments)


def service(client=None):
    client = client or RepositoryClient()
    app = ChatService(
        make_config(PROJECT_ROOT),
        knowledge_base=EmptyKnowledge(),
        tool_client=client,
    )
    return app, client


def planned(question: str):
    decision = PlannerAgent().plan(question, None)
    if decision.plan is None:
        raise AssertionError(f"expected a plan, got: {decision}")
    if decision.clarification:
        raise AssertionError(f"unexpected clarification: {decision.clarification}")
    return decision.plan


def executed(question: str, client: RepositoryClient | None = None):
    client = client or RepositoryClient()
    plan = planned(question)
    execution = ExecutorAgent().create_execution(plan, client.call_tool)
    updates = list(execution.updates())
    return plan, execution, updates, client


class AgentRoleAndPlanTests(unittest.TestCase):
    def test_planner_only_plans_and_executor_only_executes(self):
        client = RepositoryClient()
        decision = PlannerAgent().plan(
            "DEMO001 推荐一门周二下午评分高、不点名的通识课", None
        )

        self.assertIsNotNone(decision.plan)
        self.assertFalse(decision.clarification)
        self.assertFalse(client.calls, "Planner must not invoke tools")

        execution = ExecutorAgent().create_execution(decision.plan, client.call_tool)
        self.assertFalse(client.calls, "creating an execution must remain lazy")
        list(execution.updates())
        self.assertEqual(
            [name for name, _arguments in client.calls],
            [
                "student_schedule_query",
                "course_offering_search",
                "course_review_search",
            ],
        )

        before_review = list(client.calls)
        review = ReviewerAgent().review(decision.plan, execution)
        self.assertTrue(review.approved)
        self.assertEqual(client.calls, before_review, "Reviewer must use captured evidence only")

    def test_plan_validator_accepts_only_bounded_known_templates(self):
        original = planned("DEMO001 推荐一门周二下午评分高、不点名的通识课")
        validate_plan(original)

        mutations = []
        unknown_task = copy.deepcopy(original)
        unknown_task.task_type = "arbitrary_shell_task"
        mutations.append(unknown_task)

        unknown_tool = copy.deepcopy(original)
        unknown_tool.steps[0]["tool"] = "run_arbitrary_code"
        mutations.append(unknown_tool)

        reordered = copy.deepcopy(original)
        reordered.steps[0], reordered.steps[1] = reordered.steps[1], reordered.steps[0]
        mutations.append(reordered)

        duplicate_step = copy.deepcopy(original)
        duplicate_step.steps[1]["step_id"] = duplicate_step.steps[0]["step_id"]
        mutations.append(duplicate_step)

        too_many_calls = copy.deepcopy(original)
        too_many_calls.steps = [
            {
                "step_id": f"step-{index}",
                "title": "invalid extra call",
                "tool": "course_review_search",
                "status": "pending",
            }
            for index in range(1, 8)
        ]
        mutations.append(too_many_calls)

        for plan in mutations:
            with self.subTest(task_type=plan.task_type, steps=plan.steps):
                with self.assertRaises(Exception):
                    validate_plan(plan)

    def test_planner_clarification_never_creates_an_execution(self):
        decision = PlannerAgent().plan("推荐周二下午和周三晚上的通识课", None)
        self.assertTrue(decision.recognized)
        self.assertTrue(decision.clarification)


class ReviewerTests(unittest.TestCase):
    def test_course_and_book_workflows_are_independently_approved(self):
        questions = (
            "DEMO001 推荐一门周二下午评分高、不点名的通识课",
            "比较《高数》和《线代》二手书，总预算50元",
        )
        for question in questions:
            with self.subTest(question=question):
                plan, execution, _updates, _client = executed(question)
                review = ReviewerAgent().review(plan, execution)
                self.assertTrue(review.approved, review.issues)
                self.assertEqual(review.status, "approved")
                self.assertTrue(review.summary)

    def test_truthful_no_result_is_approved_without_relaxing_constraints(self):
        plan, execution, _updates, _client = executed(
            "DEMO001 推荐一门周二下午评分至少5分且不点名的通识课"
        )
        self.assertEqual(execution.status, "no_result")
        self.assertFalse(execution.items)

        review = ReviewerAgent().review(plan, execution)

        self.assertTrue(review.approved, review.issues)
        self.assertEqual(review.status, "approved")

    def test_truthful_tool_failure_is_reviewed_and_not_presented_as_success(self):
        client = RepositoryClient()
        client.fail_name = "course_review_search"
        plan, execution, _updates, _client = executed(
            "DEMO001 推荐一门周二下午评分高的通识课", client
        )
        self.assertEqual(execution.status, "failed")
        self.assertFalse(execution.items)

        review = ReviewerAgent().review(plan, execution)

        self.assertTrue(review.approved, review.issues)
        self.assertEqual(review.status, "approved")
        self.assertNotIn("OFFERING-DEMO-007", execution.answer)

    def test_reviewer_rejects_forged_course_id_and_rating(self):
        plan, execution, _updates, _client = executed(
            "DEMO001 推荐一门周二下午评分高、不点名的通识课"
        )
        original = copy.deepcopy(execution.items[0])

        execution.items[0]["offering_id"] = "OFFERING-FORGED-999"
        review = ReviewerAgent().review(plan, execution)
        self.assertFalse(review.approved)
        self.assertTrue(review.issues)

        execution.items[0] = original
        old_rating = f"{float(original['rating']):g}/5"
        execution.answer = execution.answer.replace(old_rating, "5/5")
        review = ReviewerAgent().review(plan, execution)
        self.assertFalse(review.approved)
        self.assertTrue(review.issues)

    def test_reviewer_rejects_conflicting_or_attendance_ineligible_course(self):
        plan, execution, _updates, client = executed(
            "DEMO001 推荐一门周二下午评分高、不点名的通识课"
        )
        offerings = client.repository.course_offering_search("通识", "五山校区")["items"]
        reviews = client.repository.course_review_search("通识")["items"]
        review_by_course = {item["course_id"]: item for item in reviews}

        for offering_id in ("OFFERING-DEMO-008", "OFFERING-DEMO-001"):
            with self.subTest(offering_id=offering_id):
                offering = copy.deepcopy(
                    next(item for item in offerings if item["offering_id"] == offering_id)
                )
                source_review = review_by_course[offering["course_id"]]
                execution.items = [
                    {
                        **offering,
                        "rating": source_review["rating"],
                        "review_count": source_review["review_count"],
                        "workload": source_review["workload"],
                        "attendance_policy": source_review["attendance_policy"],
                    }
                ]
                review = ReviewerAgent().review(plan, execution)
                self.assertFalse(review.approved)
                self.assertTrue(review.issues)

    def test_reviewer_rejects_duplicate_listing_and_over_budget_selection(self):
        plan, execution, _updates, _client = executed(
            "比较《高数》和《线代》二手书，总预算50元"
        )
        original_items = copy.deepcopy(execution.items)

        execution.items[1]["listing_id"] = execution.items[0]["listing_id"]
        review = ReviewerAgent().review(plan, execution)
        self.assertFalse(review.approved)
        self.assertTrue(review.issues)

        execution.items = original_items
        plan.constraints["total_budget"] = 1.0
        review = ReviewerAgent().review(plan, execution)
        self.assertFalse(review.approved)
        self.assertTrue(review.issues)

    def test_reviewer_rejects_forged_book_id_and_total(self):
        plan, execution, _updates, _client = executed(
            "比较《高数》和《线代》二手书，总预算50元"
        )
        execution.items[0]["listing_id"] = "BOOK-FORGED-999"
        review = ReviewerAgent().review(plan, execution)
        self.assertFalse(review.approved)

        plan, execution, _updates, _client = executed(
            "比较《高数》和《线代》二手书，总预算50元"
        )
        total = sum(float(item["price"]) for item in execution.items)
        execution.answer = execution.answer.replace(
            f"最低组合总价：{total:g} 元", "最低组合总价：1 元"
        )
        review = ReviewerAgent().review(plan, execution)
        self.assertFalse(review.approved)
        self.assertTrue(review.issues)


class CoordinatorTests(unittest.TestCase):
    def test_coordinator_exposes_three_roles_execution_and_outcome(self):
        client = RepositoryClient()
        coordinator = MultiAgentCoordinator(
            "DEMO001 推荐一门周二下午评分高、不点名的通识课",
            None,
            client.call_tool,
        )

        outcome = coordinator.run()

        self.assertIs(outcome, coordinator.outcome)
        self.assertIsNotNone(coordinator.execution)
        self.assertIsNotNone(outcome.review)
        self.assertTrue(outcome.review.approved, outcome.review.issues)
        self.assertEqual([run["agent_id"] for run in coordinator.agent_runs], [
            "planner", "executor", "reviewer"
        ])
        self.assertEqual([run["role"] for run in coordinator.agent_runs], [
            "规划", "执行", "复核"
        ])
        self.assertTrue(all(run["status"] == "completed" for run in coordinator.agent_runs))

    def test_stream_waits_for_review_before_result_or_answer(self):
        client = RepositoryClient()
        coordinator = MultiAgentCoordinator(
            "DEMO001 推荐一门周二下午评分高、不点名的通识课",
            None,
            client.call_tool,
        )

        updates = list(coordinator.updates())
        kinds = [update.kind for update in updates]
        review_index = next(
            index for index, update in enumerate(updates)
            if update.kind == "agent_result" and update.payload["agent_id"] == "reviewer"
        )

        self.assertNotIn("delta", kinds[:review_index])
        self.assertNotIn("task_result", kinds[:review_index])
        self.assertLess(review_index, kinds.index("task_result"))
        self.assertLess(kinds.index("task_result"), kinds.index("delta"))
        self.assertEqual(
            [update.payload["agent_id"] for update in updates if update.kind == "agent_start"],
            ["planner", "executor", "reviewer"],
        )
        self.assertEqual(
            [update.payload["agent_id"] for update in updates if update.kind == "agent_result"],
            ["planner", "executor", "reviewer"],
        )

    def test_reviewer_exception_fails_closed(self):
        class ExplodingReviewer:
            def review(self, *_args, **_kwargs):
                raise RuntimeError("reviewer unavailable")

        client = RepositoryClient()
        coordinator = MultiAgentCoordinator(
            "DEMO001 推荐一门周二下午评分高、不点名的通识课",
            None,
            client.call_tool,
            reviewer=ExplodingReviewer(),
        )

        updates = list(coordinator.updates())

        self.assertIsNotNone(coordinator.outcome)
        self.assertIsNotNone(coordinator.outcome.review)
        self.assertFalse(coordinator.outcome.review.approved)
        self.assertTrue(coordinator.outcome.review.issues)
        self.assertFalse(any(
            update.kind == "delta" and "OFFERING-DEMO-007" in update.text
            for update in updates
        ))
        self.assertTrue(any(
            update.kind == "task_result" and update.payload["status"] == "review_failed"
            for update in updates
        ))


class ChatServicePhaseFiveTests(unittest.TestCase):
    def test_phase_five_wording_is_no_longer_blocked_by_scope_guard(self):
        app, client = service()

        result = app.chat(
            "请让 Planner、Executor 和 Reviewer 三个 Agent 协作，"
            "DEMO001 推荐一门周二下午评分高、不点名的通识课"
        )

        self.assertNotEqual(result.mode, "not_implemented")
        self.assertEqual(result.task_status, "completed")
        self.assertTrue(result.review["approved"])
        self.assertEqual([run["agent_id"] for run in result.agent_runs], [
            "planner", "executor", "reviewer"
        ])
        self.assertEqual(len(client.calls), 3)

    def test_multi_agent_for_prefix_does_not_become_course_query(self):
        app, client = service()

        result = app.chat(
            "用多 Agent 为 DEMO001 推荐一门周二下午评分高且不点名的通识课"
        )

        self.assertEqual(result.task_status, "completed")
        self.assertIn("OFFERING-DEMO-007", result.answer)
        self.assertEqual(client.calls[1][1]["course_name"], "通识")
        self.assertEqual(client.calls[2][1]["course_name"], "通识")

    def test_cancelled_multi_agent_stream_does_not_save_history(self):
        app, _client = service()
        stream = app.stream_chat(
            "DEMO001 推荐一门周二下午评分高、不点名的通识课", "phase5-cancel"
        )
        updates = iter(stream.updates)
        first = next(updates)
        self.assertEqual(first.kind, "agent_start")
        updates.close()

        self.assertFalse(app.sessions.history("phase5-cancel"))


class PhaseFiveHTTPTests(unittest.TestCase):
    def test_http_exposes_only_sanitised_agent_trace(self):
        app, _client = service()
        server = create_server(app.config, app=app, host="127.0.0.1", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}"
            body = json.dumps({
                "message": "请让 Planner、Executor 和 Reviewer 协作，"
                           "DEMO001 推荐一门周二下午评分高、不点名的通识课",
                "session_id": "phase5-http",
            }, ensure_ascii=False).encode("utf-8")
            request = Request(
                url + "/api/chat", body, {"Content-Type": "application/json"}
            )
            with urlopen(request, timeout=5) as response:
                events = [json.loads(line) for line in response if line.strip()]

            event_types = [event["type"] for event in events]
            self.assertEqual(event_types[0], "start")
            self.assertEqual(event_types[-1], "done")
            review_index = next(
                index for index, event in enumerate(events)
                if event["type"] == "agent_result" and event["agent_id"] == "reviewer"
            )
            self.assertFalse(any(kind in {"delta", "task_result"} for kind in event_types[:review_index]))
            for event in events:
                if event["type"] in {"agent_start", "agent_result", "tool_result"}:
                    self.assertNotIn("items", event)
                    self.assertNotIn("raw_items", event)
                    self.assertNotIn("evidence", event)
                    self.assertNotIn("answer", event)

            with urlopen(url + "/api/health", timeout=5) as response:
                health = json.load(response)
            self.assertEqual(health["stage"], [1, 2, 3, 4, 5])
            self.assertEqual(
                health["multi_agent"]["roles"],
                ["planner", "executor", "reviewer"],
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
