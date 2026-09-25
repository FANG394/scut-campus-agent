from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator
from uuid import uuid4

from campus_agent.mcp.client import TOOL_NAMES
from campus_agent.mcp.repositories import normalize_book_query
from campus_agent.mcp.schemas import ToolEnvelope
from campus_agent.models import ChatStreamUpdate
from campus_agent.planning import (
    MAX_BOOKS,
    MAX_TOOL_CALLS,
    TaskExecution,
    TaskPlan,
    WorkflowError,
    build_task_plan,
    filter_course_offerings,
)


class MultiAgentError(RuntimeError):
    """Raised when a phase-five hand-off violates its bounded contract."""


class PlanValidationError(MultiAgentError):
    """Raised when Planner Agent produced anything outside the allow-list."""


@dataclass(slots=True)
class PlanningDecision:
    recognized: bool
    plan: TaskPlan | None = None
    clarification: str = ""


@dataclass(slots=True)
class ReviewResult:
    approved: bool
    summary: str
    issues: list[str] = field(default_factory=list)
    status: str = "approved"

    def as_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "status": self.status,
            "summary": self.summary,
            "issues": list(self.issues),
        }


@dataclass(slots=True)
class AgentRun:
    run_id: str
    agent_id: str
    display_name: str
    role: str
    status: str
    message: str
    approved: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "display_name": self.display_name,
            "role": self.role,
            "status": self.status,
            "message": self.message,
        }
        if self.approved is not None:
            payload["approved"] = self.approved
        return payload


@dataclass(slots=True)
class MultiAgentOutcome:
    answer: str
    status: str
    execution: TaskExecution | None
    review: ReviewResult | None
    agent_runs: list[dict[str, Any]]


_PLAN_ID = re.compile(r"plan-[0-9a-f]{12}\Z")
_DEMO_ID = re.compile(r"DEMO\d{3}\Z")
_CLOCK = re.compile(r"(?:[01]\d|2[0-3]):[0-5]\d\Z")
_COURSE_TITLE = "按课表和评价推荐模拟课程"
_BOOK_TITLE = "多本模拟二手书比价与预算筛选"
_FORBIDDEN_EVIDENCE_KEYS = {
    "contact", "contact_info", "email", "phone", "seller_contact",
    "student_name", "real_name", "password", "token", "credential",
}
_ROLES = {
    "planner": ("Planner Agent", "规划"),
    "executor": ("Executor Agent", "执行"),
    "reviewer": ("Reviewer Agent", "复核"),
}
_THREE_ROLE_PHRASE = re.compile(
    r"(?:请)?(?:让|用|使用|通过)?\s*Planner(?:\s*Agent)?\s*[、,，]\s*"
    r"Executor(?:\s*Agent)?\s*(?:和|与|及|、)\s*Reviewer(?:\s*Agent)?"
    r"\s*(?:三个\s*Agent)?\s*(?:协作|共同(?:完成)?)?\s*(?:来|为)?[，,:：]?",
    re.IGNORECASE,
)
_MULTI_AGENT_PHRASE = re.compile(
    r"(?:请)?(?:让|用|使用|通过)?\s*(?:多\s*Agent|多智能体|三个\s*Agent)"
    r"\s*(?:协作|共同(?:完成)?)?\s*(?:来|为)?[，,:：]?",
    re.IGNORECASE,
)


def _business_question(question: str) -> str:
    """Remove orchestration wording without changing business constraints."""

    cleaned = _THREE_ROLE_PHRASE.sub(" ", question)
    cleaned = _MULTI_AGENT_PHRASE.sub(" ", cleaned)
    return " ".join(cleaned.split())


def _course_step_specs() -> list[tuple[str, str | None]]:
    return [
        ("查询匿名学生完整课表", "student_schedule_query"),
        ("查询候选课程开课时间", "course_offering_search"),
        ("查询候选课程聚合评价", "course_review_search"),
        ("校验时间、周次、校区及评价约束", None),
        ("按评分生成有依据的推荐", None),
    ]


def _book_step_specs(names: list[str]) -> list[tuple[str, str | None]]:
    return [
        *[(f"查询《{name}》模拟二手书", "secondhand_book_search") for name in names],
        ("校验书籍与预算并比较最低组合", None),
        ("汇总模拟比价结果", None),
    ]


def _finite_number(value: object, *, minimum: float, maximum: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and minimum <= float(value) <= maximum
    )


def _validate_steps(plan: TaskPlan, specs: list[tuple[str, str | None]], *, require_pending: bool) -> None:
    if len(plan.steps) != len(specs) or len(plan.steps) > MAX_TOOL_CALLS + 2:
        raise PlanValidationError("步骤数量不符合固定工作流。")
    allowed_statuses = {"pending"} if require_pending else {"pending", "running", "completed", "failed", "skipped"}
    for index, (step, spec) in enumerate(zip(plan.steps, specs), 1):
        if not isinstance(step, dict) or set(step) != {"step_id", "title", "tool", "status"}:
            raise PlanValidationError("步骤结构不符合固定工作流。")
        if step["step_id"] != f"step-{index}" or (step["title"], step["tool"]) != spec:
            raise PlanValidationError("步骤顺序或工具白名单不符合固定工作流。")
        if step["status"] not in allowed_statuses:
            raise PlanValidationError("步骤状态无效。")
        if step["tool"] is not None and step["tool"] not in TOOL_NAMES:
            raise PlanValidationError("计划包含未授权工具。")


def validate_plan(plan: TaskPlan, *, require_pending: bool = True) -> None:
    """Validate the complete Planner→Executor hand-off, including exact tool order."""

    if not isinstance(plan, TaskPlan) or not _PLAN_ID.fullmatch(plan.plan_id):
        raise PlanValidationError("计划编号无效。")
    if plan.clarification:
        raise PlanValidationError("需要澄清的计划不得交给 Executor Agent。")
    if plan.task_type == "course_recommendation":
        if plan.title != _COURSE_TITLE:
            raise PlanValidationError("课程计划标题不符合固定模板。")
        required = {
            "student_id", "demo_student_defaulted", "course_query", "min_rating",
            "no_attendance", "low_workload", "limit", "time_mode", "time_interpretation",
        }
        optional = {"campus", "weekday", "period", "time_range"}
        if set(plan.constraints) - required - optional or not required.issubset(plan.constraints):
            raise PlanValidationError("课程计划含缺失或未授权约束。")
        constraints = plan.constraints
        if not isinstance(constraints["student_id"], str) or not _DEMO_ID.fullmatch(constraints["student_id"]):
            raise PlanValidationError("只允许匿名演示编号。")
        if type(constraints["demo_student_defaulted"]) is not bool:
            raise PlanValidationError("演示编号标记无效。")
        if not isinstance(constraints["course_query"], str) or not 1 <= len(constraints["course_query"]) <= 100:
            raise PlanValidationError("课程查询词无效。")
        if not _finite_number(constraints["min_rating"], minimum=0, maximum=5):
            raise PlanValidationError("最低评分无效。")
        if type(constraints["no_attendance"]) is not bool or type(constraints["low_workload"]) is not bool:
            raise PlanValidationError("课程布尔约束无效。")
        if isinstance(constraints["limit"], bool) or not isinstance(constraints["limit"], int) or not 1 <= constraints["limit"] <= 3:
            raise PlanValidationError("课程结果数量无效。")
        if constraints["time_mode"] not in {"avoid", "within"} or not isinstance(constraints["time_interpretation"], str):
            raise PlanValidationError("时间解释无效。")
        if "campus" in constraints and constraints["campus"] not in {"五山校区", "大学城校区", "广州国际校区"}:
            raise PlanValidationError("校区约束无效。")
        if "weekday" in constraints and constraints["weekday"] not in {f"周{day}" for day in "一二三四五六日"}:
            raise PlanValidationError("星期约束无效。")
        if "period" in constraints and constraints["period"] not in {"上午", "下午", "晚上"}:
            raise PlanValidationError("时段约束无效。")
        if "time_range" in constraints:
            interval = constraints["time_range"]
            if not isinstance(interval, list) or len(interval) != 2 or not all(isinstance(item, str) and _CLOCK.fullmatch(item) for item in interval):
                raise PlanValidationError("具体时间约束无效。")
        _validate_steps(plan, _course_step_specs(), require_pending=require_pending)
    elif plan.task_type == "book_comparison":
        if plan.title != _BOOK_TITLE or set(plan.constraints) != {"book_names", "total_budget", "price_limit", "selection"}:
            raise PlanValidationError("二手书计划结构不符合固定模板。")
        constraints = plan.constraints
        names = constraints["book_names"]
        if not isinstance(names, list) or not 2 <= len(names) <= MAX_BOOKS:
            raise PlanValidationError("二手书计划必须包含2至4个书名。")
        if any(not isinstance(name, str) or not 1 <= len(name) <= 100 for name in names):
            raise PlanValidationError("书名无效。")
        if len({normalize_book_query(name) for name in names}) != len(names):
            raise PlanValidationError("书名重复或等价。")
        for key in ("total_budget", "price_limit"):
            value = constraints[key]
            if value is not None and not _finite_number(value, minimum=0, maximum=10000):
                raise PlanValidationError("预算约束无效。")
        if constraints["selection"] != "每个书名各选一本，比较可检索模拟记录的最低组合价格":
            raise PlanValidationError("二手书选择规则不符合固定模板。")
        _validate_steps(plan, _book_step_specs(names), require_pending=require_pending)
    else:
        raise PlanValidationError("任务类型不在阶段五白名单中。")

    tool_count = sum(step["tool"] is not None for step in plan.steps)
    if tool_count > MAX_TOOL_CALLS:
        raise PlanValidationError("计划超过工具调用上限。")


def _plan_fingerprint(plan: TaskPlan) -> str:
    payload = {
        "plan_id": plan.plan_id,
        "task_type": plan.task_type,
        "title": plan.title,
        "constraints": plan.constraints,
        "steps": plan.steps,
        "clarification": plan.clarification,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class PlannerAgent:
    """Parse user intent into one of two deterministic, allow-listed plans."""

    def plan(self, question: str, tool_hint: str | None = None) -> PlanningDecision:
        candidate = build_task_plan(_business_question(question), tool_hint)
        if candidate is None:
            return PlanningDecision(recognized=False)
        if candidate.clarification:
            return PlanningDecision(recognized=True, plan=candidate, clarification=candidate.clarification)
        validate_plan(candidate)
        return PlanningDecision(recognized=True, plan=candidate)


class ExecutorAgent:
    """Execute only a validated plan; it never decides which workflow to run."""

    def create_execution(
        self,
        plan: TaskPlan,
        call_tool: Callable[[str, dict[str, Any]], ToolEnvelope],
    ) -> TaskExecution:
        validate_plan(plan)
        # TaskExecution updates step states and resolves a campus. Give it a
        # private copy so the Planner Agent's signed hand-off remains unchanged.
        return TaskExecution(deepcopy(plan), call_tool)


def _contains_forbidden_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(str(key).lower() in _FORBIDDEN_EVIDENCE_KEYS or _contains_forbidden_key(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def _expected_tool_names(plan: TaskPlan) -> list[str]:
    return [step["tool"] for step in plan.steps if isinstance(step.get("tool"), str)]


def _validate_execution_copy(plan: TaskPlan, execution: TaskExecution) -> list[str]:
    issues: list[str] = []
    try:
        validate_plan(plan)
        runtime = execution.plan
        if runtime.plan_id != plan.plan_id or runtime.task_type != plan.task_type or runtime.title != plan.title:
            issues.append("Executor Agent 更改了计划身份。")
        original_steps = [(item["step_id"], item["title"], item["tool"]) for item in plan.steps]
        runtime_steps = [(item.get("step_id"), item.get("title"), item.get("tool")) for item in runtime.steps]
        if runtime_steps != original_steps:
            issues.append("Executor Agent 更改了步骤或工具顺序。")
        base_constraints = dict(runtime.constraints)
        base_constraints.pop("resolved_campus", None)
        if base_constraints != plan.constraints:
            issues.append("Executor Agent 更改了 Planner Agent 的约束。")
        runtime_for_validation = deepcopy(runtime)
        runtime_for_validation.constraints.pop("resolved_campus", None)
        validate_plan(runtime_for_validation, require_pending=False)
    except (PlanValidationError, KeyError, TypeError, AttributeError) as exc:
        issues.append(f"执行计划结构无效：{exc}")
    return issues


def _trace_and_evidence_issues(plan: TaskPlan, execution: TaskExecution) -> list[str]:
    issues: list[str] = []
    expected = _expected_tool_names(plan)
    traces = execution.tool_calls
    evidence = execution.tool_results
    if len(traces) != len(evidence):
        issues.append("工具调用记录与原始证据数量不一致。")
        return issues
    actual = [trace.get("name") for trace in traces]
    if actual != expected[:len(actual)] or len(actual) > len(expected):
        issues.append("Executor Agent 未按计划顺序调用工具。")
    if execution.status in {"completed", "no_result"} and len(actual) != len(expected):
        issues.append("任务声称完成但工具证据不完整。")
    for index, (trace, result) in enumerate(zip(traces, evidence)):
        name = actual[index]
        if not isinstance(result, dict) or result.get("tool") != name:
            issues.append("工具证据与调用名称不匹配。")
            continue
        if result.get("data_mode") != "demo" or trace.get("data_mode") != "demo":
            issues.append("阶段五仅允许有明确标记的模拟数据。")
        if trace.get("status") != ("completed" if result.get("ok") is True else "failed"):
            issues.append("工具状态与原始证据不一致。")
        if trace.get("item_count") != len(result.get("items", [])):
            issues.append("工具结果数量与原始证据不一致。")
        if _contains_forbidden_key(result):
            issues.append("工具证据包含禁止进入工作流的敏感字段。")
    return issues


def _expected_course_arguments(plan: TaskPlan, execution: TaskExecution) -> list[dict[str, Any]] | None:
    constraints = plan.constraints
    campus = execution.plan.constraints.get("resolved_campus")
    if not isinstance(campus, str):
        return None
    return [
        {"student_id": constraints["student_id"]},
        {"course_name": constraints["course_query"], "campus": campus},
        {"course_name": constraints["course_query"]},
    ]


def _expected_book_arguments(plan: TaskPlan) -> list[dict[str, Any]]:
    expected: list[dict[str, Any]] = []
    for name in plan.constraints["book_names"]:
        arguments: dict[str, Any] = {"book_name": name}
        if plan.constraints["price_limit"] is not None:
            arguments["price_limit"] = plan.constraints["price_limit"]
        expected.append(arguments)
    return expected


def _arguments_match(execution: TaskExecution, expected: list[dict[str, Any]]) -> bool:
    return [trace.get("arguments") for trace in execution.tool_calls] == expected[:len(execution.tool_calls)]


def _safe_failed_execution(execution: TaskExecution) -> ReviewResult:
    issues: list[str] = []
    if execution.items:
        issues.append("失败任务仍携带推荐项。")
    if "任务没有完成" not in execution.answer or "不会根据失败或缺失的数据生成推荐" not in execution.answer:
        issues.append("失败任务没有使用安全的停止说明。")
    statuses = [step.get("status") for step in execution.plan.steps]
    if "failed" not in statuses:
        issues.append("失败任务缺少失败步骤。")
    if issues:
        return ReviewResult(False, "失败结果的安全性复核未通过。", issues, "rejected")
    return ReviewResult(True, "执行失败已被如实标记，且没有输出推荐。", [], "approved")


def _review_course(plan: TaskPlan, execution: TaskExecution) -> ReviewResult:
    if len(execution.tool_results) != 3 or any(result.get("ok") is not True for result in execution.tool_results):
        return ReviewResult(False, "课程证据不完整。", ["缺少三个成功的课程工具证据。"], "rejected")
    expected_arguments = _expected_course_arguments(plan, execution)
    if expected_arguments is None or not _arguments_match(execution, expected_arguments):
        return ReviewResult(False, "课程工具参数复核未通过。", ["工具参数与 Planner Agent 的约束不一致。"], "rejected")
    schedule, offerings, reviews = (result["items"] for result in execution.tool_results)
    student_id = plan.constraints["student_id"]
    if any(item.get("student_alias") != student_id for item in schedule):
        return ReviewResult(False, "匿名学生证据不匹配。", ["课表证据属于不同演示编号。"], "rejected")
    campus = execution.plan.constraints.get("resolved_campus")
    if plan.constraints.get("campus") is not None and campus != plan.constraints["campus"]:
        return ReviewResult(False, "校区解析结果不一致。", ["执行时更改了明确校区。"], "rejected")
    if plan.constraints.get("campus") is None:
        schedule_campuses = {item.get("campus") for item in schedule}
        if len(schedule_campuses) != 1 or campus != next(iter(schedule_campuses), None):
            return ReviewResult(False, "无法独立确认校区。", ["课表不能唯一解析出执行所用校区。"], "rejected")
    try:
        constraints = deepcopy(plan.constraints)
        constraints["resolved_campus"] = campus
        selected, _reasons = filter_course_offerings(schedule, offerings, reviews, constraints, campus)
        expected_items = selected[:constraints["limit"]]
    except (WorkflowError, KeyError, TypeError, ValueError) as exc:
        if execution.status == "failed":
            return _safe_failed_execution(execution)
        return ReviewResult(False, "课程证据无法独立复算。", [str(exc)], "rejected")
    expected_status = "completed" if expected_items else "no_result"
    issues: list[str] = []
    if execution.status != expected_status:
        issues.append(f"独立复算状态应为 {expected_status}，执行结果却为 {execution.status}。")
    if execution.items != expected_items:
        issues.append("课程推荐项与独立筛选结果不一致。")
    if "模拟数据" not in execution.answer:
        issues.append("课程答案缺少模拟数据声明。")
    for item in expected_items:
        facts = [
            item["offering_id"], f"{float(item['rating']):g}/5", str(item["review_count"]),
            item["location"], item["attendance_policy"], f"工作量{item['workload']}",
        ]
        facts.extend(
            f"{meeting['weekday']} {meeting['start_time']}–{meeting['end_time']}"
            for meeting in item["meetings"]
        )
        if any(str(fact) not in execution.answer for fact in facts):
            issues.append(f"课程 {item['offering_id']} 的答案事实与工具证据不一致。")
    if not expected_items and "没有找到同时满足全部条件" not in execution.answer:
        issues.append("空结果没有被如实说明。")
    if issues:
        return ReviewResult(False, "课程推荐未通过独立复核。", issues, "rejected")
    return ReviewResult(True, f"已独立复算并确认 {len(expected_items)} 条课程结果。", [], "approved")


def _valid_book_pool(items: object, price_limit: float | None) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        raise ValueError("二手书证据 items 不是列表。")
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items[:10]:
        if not isinstance(item, dict):
            raise ValueError("二手书证据记录结构无效。")
        listing_id, book, price = item.get("listing_id"), item.get("book"), item.get("price")
        if not isinstance(listing_id, str) or not listing_id or listing_id in seen:
            raise ValueError("二手书证据编号缺失或重复。")
        if not isinstance(book, str) or not book or not _finite_number(price, minimum=0, maximum=1_000_000):
            raise ValueError("二手书证据含无效书名或价格。")
        seen.add(listing_id)
        if price_limit is None or float(price) <= price_limit:
            pool.append(dict(item))
    return pool


def _review_books(plan: TaskPlan, execution: TaskExecution) -> ReviewResult:
    names = plan.constraints["book_names"]
    if len(execution.tool_results) != len(names) or any(result.get("ok") is not True for result in execution.tool_results):
        return ReviewResult(False, "二手书证据不完整。", ["并非每个书名都有成功的工具证据。"], "rejected")
    expected_arguments = _expected_book_arguments(plan)
    if not _arguments_match(execution, expected_arguments):
        return ReviewResult(False, "二手书工具参数复核未通过。", ["查询书名或价格上限被更改。"], "rejected")
    try:
        pools = [_valid_book_pool(result["items"], plan.constraints["price_limit"]) for result in execution.tool_results]
        choices = [
            choice for choice in itertools.product(*pools)
            if len({item["listing_id"] for item in choice}) == len(names)
        ]
        cheapest = min(
            choices,
            key=lambda choice: (
                sum(float(item["price"]) for item in choice),
                tuple(item["listing_id"] for item in choice),
            ),
        ) if choices else None
        cost = sum(float(item["price"]) for item in cheapest) if cheapest else None
        budget = plan.constraints["total_budget"]
        affordable = cheapest is not None and (budget is None or cost <= budget)
        expected_items = list(cheapest) if affordable else []
    except (KeyError, TypeError, ValueError) as exc:
        if execution.status == "failed":
            return _safe_failed_execution(execution)
        return ReviewResult(False, "二手书证据无法独立复算。", [str(exc)], "rejected")
    expected_status = "completed" if expected_items else "no_result"
    issues: list[str] = []
    if execution.status != expected_status:
        issues.append(f"独立复算状态应为 {expected_status}，执行结果却为 {execution.status}。")
    if execution.items != expected_items:
        issues.append("二手书选择项不是证据中的最低有效组合。")
    if "模拟数据" not in execution.answer:
        issues.append("二手书答案缺少模拟数据声明。")
    if cheapest:
        for name, item in zip(names, cheapest):
            if item["listing_id"] not in execution.answer or f"{float(item['price']):g} 元" not in execution.answer:
                issues.append(f"《{name}》的答案价格或编号与证据不一致。")
        if f"最低组合总价：{float(cost):g} 元" not in execution.answer:
            issues.append("答案中的最低组合总价与独立复算不一致。")
        if budget is not None and affordable and f"剩余 {float(budget - cost):g} 元" not in execution.answer:
            issues.append("答案中的预算余额与独立复算不一致。")
        if budget is not None and not affordable and f"超出 {float(cost - budget):g} 元" not in execution.answer:
            issues.append("答案中的超预算金额与独立复算不一致。")
    elif "没有找到完整组合" not in execution.answer:
        issues.append("空结果没有被如实说明。")
    if issues:
        return ReviewResult(False, "二手书比价未通过独立复核。", issues, "rejected")
    return ReviewResult(True, f"已独立复算并确认 {len(expected_items)} 本书的组合结果。", [], "approved")


class ReviewerAgent:
    """Recompute the outcome from raw evidence without calling any tool or model."""

    def review(self, plan: TaskPlan, execution: TaskExecution) -> ReviewResult:
        issues = [*_validate_execution_copy(plan, execution), *_trace_and_evidence_issues(plan, execution)]
        if _plan_fingerprint(plan) != getattr(execution, "planner_fingerprint", _plan_fingerprint(plan)):
            issues.append("Planner Agent 的计划指纹不匹配。")
        if issues:
            return ReviewResult(False, "执行链结构复核未通过。", issues, "rejected")
        if execution.status == "failed":
            # A complete, successful evidence set should have produced a
            # recomputable outcome. Review it normally to catch a forged failure.
            all_expected = len(execution.tool_results) == len(_expected_tool_names(plan))
            all_success = all(result.get("ok") is True for result in execution.tool_results)
            if not (all_expected and all_success):
                return _safe_failed_execution(execution)
        if plan.task_type == "course_recommendation":
            return _review_course(plan, execution)
        if plan.task_type == "book_comparison":
            return _review_books(plan, execution)
        return ReviewResult(False, "任务类型不在复核白名单中。", [plan.task_type], "rejected")


class MultiAgentCoordinator:
    """Orchestrate Planner → Executor → Reviewer with a fail-closed final gate."""

    def __init__(
        self,
        question: str,
        tool_hint: str | None,
        call_tool: Callable[[str, dict[str, Any]], ToolEnvelope],
        *,
        planner: PlannerAgent | None = None,
        executor: ExecutorAgent | None = None,
        reviewer: ReviewerAgent | None = None,
    ) -> None:
        self.question = question
        self.tool_hint = tool_hint
        self.call_tool = call_tool
        self.planner = planner or PlannerAgent()
        self.executor_agent = executor or ExecutorAgent()
        self.reviewer = reviewer or ReviewerAgent()
        self.run_id = f"run-{uuid4().hex[:12]}"
        self.decision: PlanningDecision | None = None
        self.plan: TaskPlan | None = None
        self.execution: TaskExecution | None = None
        self.review: ReviewResult | None = None
        self.outcome: MultiAgentOutcome | None = None
        self.agent_runs: list[dict[str, Any]] = []
        self._planner_fingerprint = ""
        self._prepared = False
        self._finished = False

    def _run_record(
        self,
        agent_id: str,
        status: str,
        message: str,
        approved: bool | None = None,
    ) -> AgentRun:
        display_name, role = _ROLES[agent_id]
        return AgentRun(self.run_id, agent_id, display_name, role, status, message, approved)

    def prepare(self) -> PlanningDecision:
        if self._prepared:
            assert self.decision is not None
            return self.decision
        self._prepared = True
        try:
            self.decision = self.planner.plan(self.question, self.tool_hint)
        except Exception:
            self.decision = PlanningDecision(
                recognized=True,
                clarification="Planner Agent 未能生成符合白名单的安全计划，请改写需求后重试。",
            )
        if self.decision.recognized:
            self.plan = self.decision.plan
            if self.plan is not None and not self.decision.clarification:
                self._planner_fingerprint = _plan_fingerprint(self.plan)
        return self.decision

    def planner_run(self) -> dict[str, Any]:
        decision = self.prepare()
        if not decision.recognized:
            return {}
        if decision.clarification:
            message = "需要用户澄清后才能生成安全执行计划。"
            status = "completed" if decision.plan is not None else "failed"
        else:
            assert decision.plan is not None
            message = f"已生成并校验 {len(decision.plan.steps)} 步白名单计划。"
            status = "completed"
        return self._run_record("planner", status, message).as_dict()

    def run(self) -> MultiAgentOutcome:
        for _update in self.updates():
            pass
        assert self.outcome is not None
        return self.outcome

    def updates(self) -> Iterator[ChatStreamUpdate]:
        if self._finished:
            raise MultiAgentError("同一次多 Agent 协作不能重复执行。")
        decision = self.prepare()
        if not decision.recognized or decision.plan is None or decision.clarification:
            raise MultiAgentError("只有可执行的阶段五计划才能启动协调器。")
        plan = decision.plan
        planner_result = self.planner_run()
        yield ChatStreamUpdate(kind="agent_start", payload=self._run_record("planner", "running", "正在解析约束并生成白名单计划。").as_dict())
        self.agent_runs.append(planner_result)
        yield ChatStreamUpdate(kind="agent_result", payload=planner_result)
        if _plan_fingerprint(plan) != self._planner_fingerprint:
            raise MultiAgentError("Planner Agent 的计划在交接前被更改。")
        yield ChatStreamUpdate(kind="plan", payload=plan.as_dict())

        yield ChatStreamUpdate(kind="agent_start", payload=self._run_record("executor", "running", "正在按固定顺序调用模拟数据工具。").as_dict())
        executor_status = "failed"
        executor_message = "Executor Agent 未能启动。"
        try:
            self.execution = self.executor_agent.create_execution(plan, self.call_tool)
            self.execution.planner_fingerprint = self._planner_fingerprint
            yield from self.execution.updates(emit_plan=False, emit_result=False, emit_answer=False)
            executor_status = "failed" if self.execution.status == "failed" else "completed"
            executor_message = (
                "执行失败，已停止后续步骤并保留证据供复核。"
                if self.execution.status == "failed"
                else f"执行完成，形成 {len(self.execution.items)} 条候选结果，等待独立复核。"
            )
        except Exception:
            self.execution = None
            executor_message = "执行链结构异常，未输出候选结果。"
        executor_result = self._run_record("executor", executor_status, executor_message).as_dict()
        self.agent_runs.append(executor_result)
        yield ChatStreamUpdate(kind="agent_result", payload=executor_result)

        yield ChatStreamUpdate(kind="agent_start", payload=self._run_record("reviewer", "running", "正在从原始工具证据独立复算结果。").as_dict())
        if self.execution is None:
            self.review = ReviewResult(False, "没有可复核的执行报告。", ["Executor Agent 未产生执行报告。"], "rejected")
        else:
            try:
                self.review = self.reviewer.review(plan, self.execution)
            except Exception:
                self.review = ReviewResult(False, "Reviewer Agent 运行异常。", ["复核未完成，已按失败关闭。"], "rejected")
        reviewer_status = "completed" if self.review.approved else "failed"
        reviewer_result = self._run_record(
            "reviewer", reviewer_status, self.review.summary, self.review.approved
        ).as_dict()
        self.agent_runs.append(reviewer_result)
        yield ChatStreamUpdate(kind="agent_result", payload=reviewer_result)

        if self.review.approved and self.execution is not None:
            final_status = self.execution.status
            final_answer = self.execution.answer
            item_count = len(self.execution.items)
        else:
            final_status = "review_failed"
            final_answer = (
                "Reviewer Agent 未通过结果校验，已停止输出未经确认的推荐。"
                "请检查模拟数据和执行链后重试。"
            )
            item_count = 0
        self.outcome = MultiAgentOutcome(
            answer=final_answer,
            status=final_status,
            execution=self.execution,
            review=self.review,
            agent_runs=list(self.agent_runs),
        )
        self._finished = True
        yield ChatStreamUpdate(kind="task_result", payload={
            "plan_id": plan.plan_id,
            "status": final_status,
            "review_status": self.review.status,
            "approved": self.review.approved,
            "item_count": item_count,
            "message": "Reviewer Agent 已独立复核；仅使用本地模拟数据。",
        })
        yield ChatStreamUpdate(kind="delta", text=final_answer)
