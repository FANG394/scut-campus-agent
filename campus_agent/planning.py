from __future__ import annotations

import itertools
import math
import re
from copy import deepcopy
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from campus_agent.mcp.client import MCPDependencyError, MCPToolError, _validate_envelope
from campus_agent.mcp.schemas import ToolEnvelope
from campus_agent.models import ChatStreamUpdate, ToolCallTrace
from campus_agent.tool_router import ToolRoute, extract_book_title


MAX_BOOKS = 4
MAX_TOOL_CALLS = 6
_COURSE_REQUEST = re.compile(r"(?:推荐|挑选|选择|筛选|帮我选|选一门|挑一门|找).{0,45}(?:通识(?:课|课程)?|通选课|公选课|选修课|专业拓展课|课程)|(?:课程|选课)推荐")
_BOOK_REQUEST = re.compile(r"二手|旧书|书籍|教材|比价|各一本|(?:想|要)买")
_COMPARISON = re.compile(r"比较|对比|比价|各一本|都买|一起买|总预算|合计|分别|[、，,]|(?:和|以及)")
_DAY = re.compile(r"(?:周|星期)([一二三四五六日天1-7])")
_ALIAS = re.compile(r"(?<![A-Za-z0-9])DEMO\d{3}(?![A-Za-z0-9])", re.I)
_CAMPUS = re.compile(r"五山(?:校区)?|大学城(?:校区)?|广州国际(?:校区)?")
_QUOTED = re.compile(r'[《“"]([^》”"]{1,100})[》”"]')
_WINDOWS = {"上午": (480, 720), "下午": (780, 1080), "晚上": (1080, 1380)}
_CLOCK = re.compile(r"(?:[01]\d|2[0-3]):[0-5]\d\Z")
_ATOMIC_BOOKS = (
    "概率论与数理统计", "思想道德与法治",
    "毛泽东思想和中国特色社会主义理论体系概论",
)


@dataclass(slots=True)
class TaskPlan:
    task_type: str
    title: str
    constraints: dict[str, Any]
    steps: list[dict[str, Any]]
    plan_id: str = field(default_factory=lambda: f"plan-{uuid4().hex[:12]}")
    clarification: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id, "task_type": self.task_type,
            "title": self.title, "constraints": dict(self.constraints),
            "steps": [dict(step) for step in self.steps],
        }


def build_task_plan(question: str, tool_hint: str | None = None) -> TaskPlan | None:
    """Build bounded phase-four workflows, never execute arbitrary model plans."""

    if _COURSE_REQUEST.search(question):
        return _course_plan(question)
    quoted_comparison = len(_QUOTED.findall(question)) >= 2 and re.search(r"比较|对比|比价", question)
    contextual_comparison = tool_hint == "secondhand_book_search" and re.search(r"比较|对比|比价|各一本|都买|总预算", question) and not re.search(r"课程|课表|评价|评分|知识库", question)
    if (_BOOK_REQUEST.search(question) or quoted_comparison or contextual_comparison) and _COMPARISON.search(question):
        plan = _book_plan(question)
        if plan is not None:
            return plan
    return None


def _steps(*specs: tuple[str, str | None]) -> list[dict[str, Any]]:
    return [
        {"step_id": f"step-{index}", "title": title, "tool": tool, "status": "pending"}
        for index, (title, tool) in enumerate(specs, 1)
    ]


def _course_plan(question: str) -> TaskPlan:
    constraints: dict[str, Any] = {
        "student_id": (_ALIAS.search(question).group(0).upper() if _ALIAS.search(question) else "DEMO001"),
        "demo_student_defaulted": _ALIAS.search(question) is None,
        "course_query": "专业拓展" if "专业拓展" in question else "通识",
        "min_rating": 4.5 if re.search(r"评分(?:较)?高|评价(?:较)?高|高评分|高评价|高分|评价好|口碑好", question) else 0.0,
        "no_attendance": bool(re.search(r"不点名|不要点名|无需(?:点名|签到)|免签到|不签到|不要签到", question)),
        "low_workload": bool(re.search(r"作业少|工作量低|工作量少|轻松|低工作量", question)),
        "limit": 1 if re.search(r"一(?:门|个)", question) else 3,
        "time_mode": "avoid" if re.search(r"(?:不要|避开|不安排|不在|排除).{0,6}(?:周|星期|上午|下午|晚上|\d{1,2}:)", question) else "within",
    }
    campus = _CAMPUS.search(question)
    if campus:
        constraints["campus"] = campus.group(0).removesuffix("校区") + "校区"
    day_aliases = dict(zip("1234567", "一二三四五六日"))
    days = [day_aliases.get(day, "日" if day == "天" else day) for day in _DAY.findall(question)]
    periods = [period for period in _WINDOWS if period in question]
    if days:
        constraints["weekday"] = "周" + ("日" if days[0] == "天" else days[0])
    if periods:
        constraints["period"] = periods[0]
    quantity = re.search(r"([一二两三四五六七八九]|\d+)\s*(?:门|个)(?:.{0,10})?(?:通识课|选修课|课程)?", question)
    if quantity:
        counts = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
        constraints["limit"] = counts.get(quantity.group(1), int(quantity.group(1)) if quantity.group(1).isdigit() else 3)
    rating = re.search(r"(?:评分|评价|分数)\s*(?:至少|不低于|大于等于|≥|>=)?\s*(\d+(?:\.\d+)?)\s*(?:分|/5)?", question)
    if rating:
        constraints["min_rating"] = float(rating.group(1))
    quoted = _QUOTED.findall(question)
    if quoted:
        constraints["course_query"] = quoted[0].strip()
    time_range = re.search(r"(\d{1,2}:\d{2})\s*[-–~至到]\s*(\d{1,2}:\d{2})", question)
    if time_range:
        constraints["time_range"] = [part.zfill(5) for part in time_range.groups()]
    if not quoted:
        subject = question
        for pattern in (_ALIAS, _DAY, _CAMPUS):
            subject = pattern.sub(" ", subject)
        for match in (rating, time_range):
            if match:
                subject = subject.replace(match.group(0), " ")
        subject = re.sub(r"(?:用|通过)?\s*(?:多\s*Agent|多智能体|Planner\s*Agent|Executor\s*Agent|Reviewer\s*Agent)|(?:课程|选课)推荐|请问|请|帮我选|帮我|推荐(?:一下)?|挑选|选择|筛选|选一门|挑一门|找(?:一下)?|我想|我|(?:一|二|两|三|\d+)(?:门|个)|上午|下午|晚上|通识(?:课程|课)?|通选课|公选课|选修课|专业拓展课|这门|这节|评分(?:较)?高|评价(?:较)?高|高评分|高评价|高分|评价好|口碑好|不要点名|不点名|无需(?:点名|签到)|免签到|不要签到|不签到|作业少|工作量低|工作量少|轻松|低工作量|没有课|没课|不冲突|空闲|有空|避开|不安排|不在|排除|不要", " ", subject, flags=re.IGNORECASE)
        subject = re.sub(r"(?<!\S)(?:且|和|与|同时|的|在|一个)(?!\S)", " ", subject)
        subject = "".join(subject.split()).strip(" 的在，。！？、:：；?")
        subject = subject.removesuffix("课程").removesuffix("课")
        if subject:
            constraints["course_query"] = subject[:100]
    constraints["time_interpretation"] = (
        "排除指定时段的开课班，同时检查其余上课时间是否与已选课程冲突。"
        if constraints["time_mode"] == "avoid" else
        "在指定时段寻找可选且不冲突的开课班；不代表该时段整段没有已选课程。"
    )
    plan = TaskPlan(
        "course_recommendation", "按课表和评价推荐模拟课程", constraints,
        _steps(("查询匿名学生完整课表", "student_schedule_query"),
               ("查询候选课程开课时间", "course_offering_search"),
               ("查询候选课程聚合评价", "course_review_search"),
               ("校验时间、周次、校区及评价约束", None),
               ("按评分生成有依据的推荐", None)),
    )
    if len(set(days)) > 1 or len(periods) > 1 or len(quoted) > 1 or len(_CAMPUS.findall(question)) > 1 or len(_ALIAS.findall(question)) > 1:
        plan.clarification = "请先指定一个星期和一个时段，或一门课程；多个时段的复杂组合暂不支持。"
    elif not 0 <= constraints["min_rating"] <= 5:
        plan.clarification = "模拟课程评分采用 0～5 分，请提供该范围内的最低评分。"
    elif time_range:
        try:
            start, end = (_minutes(text) for text in constraints["time_range"])
            if start >= end:
                raise ValueError("invalid interval")
            period_window = _WINDOWS.get(constraints.get("period"))
            if period_window and not period_window[0] <= start < end <= period_window[1]:
                plan.clarification = "具体起止时间与上午／下午／晚上条件冲突，请保留一致的时间要求。"
        except ValueError:
            plan.clarification = "请提供有效的起止时间，例如 14:00–16:00。"
    if not 1 <= constraints["limit"] <= 3 or re.search(r"同时(?:选|修|上)|一起(?:选|修|上)|课程组合|组合选课", question):
        plan.clarification = "当前提供最多3条独立课程备选，不生成多门同时选修的组合；请明确要几条备选。"
    if re.search(r"(?:不要|不需要|排除|不是).{0,3}(?:不点名|不签到|作业少|轻松|低工作量)|(?:评分|评价|分数).{0,3}(?:不高于|不超过|最高|至多|低于|大于(?!等于)|高于|小于|≤|<=|>(?!=))", question):
        plan.clarification = "当前支持最低评分（不低于）、明确不点名和低工作量；暂不支持反向或严格大于／小于的评价条件，请重新明确要求。"
    if re.search(r"(?:评分|评价|分数)\s*(?:至少|不低于|≥|>=)?\s*[零一二两三四五六七八九十\d]", question) and not rating:
        plan.clarification = "请用明确的数字最低评分，例如‘评分不低于4.5分’。"
    # Avoid pretending to apply unsupported hard constraints.
    if re.search(r"学分|老师|教师|线上|线下|考试形式|无考试|不开考|不考试|期末|闭卷|开卷|学期|(?:单双|单|双)周|第.{0,5}周|\d+(?:点|节)|至少\d+门|中午|早上|夜间", question):
        plan.clarification = "当前推荐支持时间、校区、评分、不点名和低工作量；你提出的其他约束缺少可靠模拟字段，请先去掉或补齐对应数据。"
    if re.search(r"二手|旧书|校园知识|知识库|校园网|主动提醒|不怎么点名|很少点名", question):
        plan.clarification = "当前课程推荐只联合课表、开课时间和评价；跨业务组合或模糊点名条件暂不支持，请拆开或明确需求。"
    if len(re.findall(r"(?:评分|评价|分数)\s*(?:至少|不低于|大于等于|≥|>=)?\s*\d+(?:\.\d+)?", question)) > 1 or len(re.findall(r"\d{1,2}:\d{2}\s*[-–~至到]\s*\d{1,2}:\d{2}", question)) > 1:
        plan.clarification = "请先提供一个最低评分和一个连续时间段，避免歧义。"
    return plan


def _book_plan(question: str) -> TaskPlan | None:
    quoted = _QUOTED.findall(question)
    budget_pattern = r"(?:总预算|合计预算|预算|总价|合计)\s*(?:不超过|最多|以内|为|是|[：:])?\s*(-?\d+(?:\.\d+)?)\s*元?"
    per_pattern = r"(?:每本|每一本|单本)\s*(?:预算|价格)?\s*(?:不超过|最多|[：:])?\s*(-?\d+(?:\.\d+)?)\s*元?(?:以内|以下)?"
    per = re.search(per_pattern, question)
    without_per = re.sub(per_pattern, "", question)
    total = re.search(budget_pattern, without_per) or re.search(r"(-?\d+(?:\.\d+)?)\s*元\s*(?:以内|以下)", without_per)
    if len(quoted) >= 2:
        names = [name.strip() for name in quoted]
    else:
        text = re.sub(budget_pattern, "", without_per)
        text = re.sub(r"-?\d+(?:\.\d+)?\s*元\s*(?:以内|以下)", "", text)
        text = re.sub(r"(?:用|通过|让)?\s*(?:多\s*Agent|多智能体|Planner\s*Agent|Executor\s*Agent|Reviewer\s*Agent)|请问|请|帮我|替我|我想|我|比较|对比|比价|分别(?:查找|查询|找|查)?|查找|查询|搜索|购买|想买|要买|找|二手(?:书|教材)?|旧(?:书|教材)|教材|各(?:买)?一本|都买|一起买|两本|最低价|最便宜|便宜的|合计|总预算|多少钱|以内|以下|推荐|价格|预算", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^\s*(?:买|收)\s*", "", text)
        protected: dict[str, str] = {}
        for index, title in enumerate(_ATOMIC_BOOKS):
            key = f"BOOKATOM{index}"
            if title in text:
                protected[key] = title
                text = text.replace(title, key)
        names = [part.strip(" 《》“”\"，。！？:：；;、?") for part in re.split(r"以及|和|与|及|[、，,；;]", text)]
        names = [protected.get(name, name) for name in names if name]
    names = [extract_book_title(name) or "" for name in names]
    if len(names) < 2:
        # Leave ordinary one-title requests on the compatible phase-three route.
        if not re.search(r"比较|对比|比价|各一本|都买|一起买", question):
            return None
        names = names[:1]
    constraints = {"book_names": names, "total_budget": float(total.group(1)) if total else None,
                   "price_limit": float(per.group(1)) if per else None,
                   "selection": "每个书名各选一本，比较可检索模拟记录的最低组合价格"}
    plan = TaskPlan("book_comparison", "多本模拟二手书比价与预算筛选", constraints,
                    _steps(*[(f"查询《{name}》模拟二手书", "secondhand_book_search") for name in names],
                           ("校验书籍与预算并比较最低组合", None), ("汇总模拟比价结果", None)))
    if not 2 <= len(names) <= MAX_BOOKS:
        plan.clarification = f"请明确提供 2～{MAX_BOOKS} 个书名，例如：比较《高等数学》和《线性代数》，总预算50元。"
    elif any(len(name) > 100 or not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", name) for name in names):
        plan.clarification = "请提供可检索的明确书名，不要只写‘这本’或‘那本’。"
    elif any(name in {"这本", "那本", "书", "的", "还有", "吗", "呢"} for name in names):
        plan.clarification = "请重新写出要比较的具体书名，不会复用旧的工具参数。"
    elif any(value is not None and not 0 <= value <= 10000 for value in (constraints["total_budget"], constraints["price_limit"])):
        plan.clarification = "预算必须在 0～10000 元之间。"
    if (re.search(r"总预算|合计预算|总价|预算|合计|元(?:以内|以下)", without_per) and not total) or (re.search(r"每本|每一本|单本", question) and not per):
        plan.clarification = "未能可靠识别预算，请使用数字，例如‘总预算50元，每本不超过30元’。"
    if re.search(r"(?:预算|总价|合计|每本|单本).{0,5}(?:不低于|至少|高于|低于|大于|小于)|成新|版本|第.{1,3}版|包邮|邮寄|卖家|联系方式|电话|成色", question):
        plan.clarification = "当前多书比价只支持书名和数字预算上限；其他条件请先改用单本查询核验。"
    total_spans = [match.span() for match in re.finditer(budget_pattern, without_per)]
    total_spans.extend(match.span() for match in re.finditer(r"-?\d+(?:\.\d+)?\s*元\s*(?:以内|以下)", without_per)
                       if not any(max(start, match.start()) < min(end, match.end()) for start, end in total_spans))
    if len(total_spans) > 1 or len(re.findall(per_pattern, question)) > 1:
        plan.clarification = "请提供一个总预算上限和一个单本预算上限，不要同时给出多组预算。"
    if _CAMPUS.search(question) or re.search(r"校区|作者|出版社|出版日期|交接地点|取书地点", question):
        plan.clarification = "当前多书比价只比较书名和预算，尚未筛选校区、作者或交接条件；请去掉这些条件或改用单本查询核验。"
    if re.search(r"课表|课程安排|课程推荐|课程评价|校园知识|知识库|主动提醒", question):
        plan.clarification = "当前多书比价不联合其他业务工具，请把课程／知识查询拆开。"
    return plan


class WorkflowError(RuntimeError):
    pass


class TaskExecution:
    """Bounded workflow executor used by the phase-five Executor Agent."""

    def __init__(self, plan: TaskPlan, call_tool: Callable[[str, dict[str, Any]], ToolEnvelope]) -> None:
        self.plan = plan
        self.call_tool = call_tool
        self.tool_calls: list[dict[str, Any]] = []
        # Reviewer Agent receives immutable snapshots of the evidence returned by
        # each tool instead of trusting the Executor Agent's prose or selection.
        self.tool_results: list[ToolEnvelope] = []
        self.warnings: list[str] = ["以下结果全部来自模拟数据，不代表真实开课、库存或学校规定。"]
        self.answer = ""
        self.status = "pending"
        self.items: list[dict[str, Any]] = []
        self._calls = 0

    def updates(
        self,
        *,
        emit_plan: bool = True,
        emit_result: bool = True,
        emit_answer: bool = True,
    ) -> Iterator[ChatStreamUpdate]:
        if emit_plan:
            yield ChatStreamUpdate(kind="plan", payload=self.plan.as_dict())
        self.status = "running"
        try:
            if self.plan.task_type == "course_recommendation":
                yield from self._recommend_courses()
            elif self.plan.task_type == "book_comparison":
                yield from self._compare_books()
            else:
                raise WorkflowError("任务类型不在允许执行的白名单中。")
            self.status = "completed" if self.items else "no_result"
        except Exception as exc:
            self.status = "failed"
            self.items = []
            detail = str(exc) if isinstance(exc, WorkflowError) else "执行数据异常或工具服务不可用。"
            self.answer = f"以下为模拟数据。任务没有完成：{detail}不会根据失败或缺失的数据生成推荐，请修复数据后重试。"
            self.warnings.append(detail)
            for step in self.plan.steps:
                if step["status"] == "running":
                    yield self._finish(step, "failed", detail)
                elif step["status"] == "pending":
                    yield self._finish(step, "skipped", "依赖步骤失败，未继续执行。")
        if emit_result:
            yield ChatStreamUpdate(kind="task_result", payload={
                "plan_id": self.plan.plan_id, "status": self.status,
                "item_count": len(self.items), "message": "仅使用本地模拟数据",
            })
        if emit_answer:
            yield ChatStreamUpdate(kind="delta", text=self.answer)

    def _start(self, step: dict[str, Any]) -> ChatStreamUpdate:
        step["status"] = "running"
        return ChatStreamUpdate(kind="step_start", payload={"plan_id": self.plan.plan_id, **step})

    def _finish(self, step: dict[str, Any], status: str, message: str) -> ChatStreamUpdate:
        step["status"] = status
        return ChatStreamUpdate(kind="step_result", payload={"plan_id": self.plan.plan_id, **step, "message": message})

    def _tool(self, step: dict[str, Any], route: ToolRoute) -> Iterator[ChatStreamUpdate]:
        self._calls += 1
        if self._calls > MAX_TOOL_CALLS:
            raise WorkflowError("工具调用达到上限，已停止执行。")
        yield self._start(step)
        call_id = f"call-{uuid4().hex[:12]}"
        yield ChatStreamUpdate(kind="tool_start", payload={"call_id": call_id, "name": route.name,
            "display_name": route.display_name, "arguments": route.public_arguments()})
        try:
            result = _validate_envelope(self.call_tool(route.name, route.arguments), expected_tool=route.name)
        except (MCPDependencyError, MCPToolError):
            result = {"ok": False, "tool": route.name, "data_mode": "demo", "items": [],
                      "message": "MCP 工具未成功执行，请检查 SDK、服务或本地数据。", "warnings": []}
        except Exception:
            result = {"ok": False, "tool": route.name, "data_mode": "demo", "items": [],
                      "message": "工具服务异常，请检查本地数据。", "warnings": []}
        self.tool_results.append(deepcopy(result))
        trace = ToolCallTrace(call_id, route.name, route.display_name, route.public_arguments(),
                              "completed" if result["ok"] else "failed", result["data_mode"],
                              len(result["items"]), result["message"]).as_dict()
        self.tool_calls.append(trace)
        self.warnings.extend(result["warnings"])
        yield ChatStreamUpdate(kind="tool_result", payload={key: value for key, value in trace.items() if key != "arguments"})
        yield self._finish(step, trace["status"], result["message"])
        if not result["ok"]:
            raise WorkflowError(f"{route.display_name}失败，后续筛选已停止。")
        if result["data_mode"] != "demo":
            raise WorkflowError("当前任务只支持经过标注的模拟数据，工具数据模式不匹配。")
        return result

    def _recommend_courses(self) -> Iterator[ChatStreamUpdate]:
        constraints = self.plan.constraints
        steps = self.plan.steps
        schedule = yield from self._tool(steps[0], ToolRoute("student_schedule_query", {"student_id": constraints["student_id"]}))
        if any(item.get("student_alias") != constraints["student_id"] for item in schedule["items"]):
            raise WorkflowError("工具返回的匿名学生编号与请求不符。")
        if constraints["demo_student_defaulted"]:
            self.warnings.append("未指定演示编号，明确使用 DEMO001；不代表用户本人的课表。")
        campus = constraints.get("campus")
        schedule_campuses = {item.get("campus") for item in schedule["items"]}
        if campus is None:
            if len(schedule_campuses) != 1 or not next(iter(schedule_campuses), None):
                raise WorkflowError("无法从完整课表确定校区，请明确提供演示编号和校区。")
            campus = next(iter(schedule_campuses))
        self.plan.constraints["resolved_campus"] = campus
        offerings = yield from self._tool(steps[1], ToolRoute("course_offering_search", {"course_name": constraints["course_query"], "campus": campus}))
        reviews = yield from self._tool(steps[2], ToolRoute("course_review_search", {"course_name": constraints["course_query"]}))
        yield self._start(steps[3])
        selected, reasons = filter_course_offerings(schedule["items"], offerings["items"], reviews["items"], constraints, campus)
        self.items = selected[:constraints["limit"]]
        filtered_count = len(offerings["items"]) - len(selected)
        yield self._finish(steps[3], "completed", f"核验 {len(offerings['items'])} 个开课班，{len(selected)} 个符合，排除 {filtered_count} 个。")
        yield self._start(steps[4])
        lines = [f"以下为模拟数据，使用 {constraints['student_id']} 的演示课表，校区：{campus}。",
                 "以下是独立备选，不是可同时选修的组合；候选之间是否冲突尚未验证。",
                 constraints["time_interpretation"],
                 f"筛选条件：最低评分 {constraints['min_rating']:g}/5；" + ("明确不点名；" if constraints["no_attendance"] else "未限制点名；") + ("低工作量。" if constraints["low_workload"] else "未限制工作量。")]
        if constraints["demo_student_defaulted"]:
            lines.insert(1, "你没有提供演示编号，本次默认 DEMO001，并非你的真实课表。")
        if self.items:
            for index, item in enumerate(self.items, 1):
                times = "；".join(f"{meeting['weekday']} {meeting['start_time']}–{meeting['end_time']}（{meeting['week_start']}–{meeting['week_end']}周）" for meeting in item["meetings"])
                lines.append(f"{index}. {item['course_name']}（{item['offering_id']}）：{item['rating']:g}/5，{item['review_count']} 条模拟评价；{times}；{item['location']}；{item['attendance_policy']}；工作量{item['workload']}。已校验与该演示课表全部上课时间及周次不冲突。")
        else:
            lines.append("没有找到同时满足全部条件的模拟开课班；不会放宽约束或把缺失数据当作符合。")
        if reasons:
            lines.append("排除原因：" + "；".join(f"{key} {count} 个" for key, count in sorted(reasons.items())) + "。")
        lines.append("以上仅比较已检索的模拟开课班，不代表全部可选课程；实际选课还需核验学校系统。")
        self.answer = "\n".join(lines)
        yield self._finish(steps[4], "completed", f"形成 {len(self.items)} 条模拟推荐。")

    def _compare_books(self) -> Iterator[ChatStreamUpdate]:
        from campus_agent.mcp.repositories import normalize_book_query

        names = self.plan.constraints["book_names"]
        canonical = [normalize_book_query(name) for name in names]
        if len(set(canonical)) != len(canonical):
            raise WorkflowError("多个书名实际是同一本教材的简称，请提供不同书名。")
        pools: list[list[dict[str, Any]]] = []
        for index, name in enumerate(names):
            arguments: dict[str, Any] = {"book_name": name}
            price_limit = self.plan.constraints["price_limit"]
            if price_limit is not None:
                arguments["price_limit"] = price_limit
            result = yield from self._tool(self.plan.steps[index], ToolRoute("secondhand_book_search", arguments))
            pool: list[dict[str, Any]] = []
            for item in result["items"][:10]:
                price = _number(item.get("price"))
                if price < 0 or not isinstance(item.get("listing_id"), str) or not item["listing_id"]:
                    raise WorkflowError("二手书记录缺少有效价格或唯一编号。")
                if not isinstance(item.get("book"), str) or not item["book"]:
                    raise WorkflowError("二手书记录缺少有效书名。")
                if price_limit is None or price <= price_limit:
                    pool.append(dict(item))
            pools.append(pool)
        step = self.plan.steps[-2]
        yield self._start(step)
        combinations = [choice for choice in itertools.product(*pools)
                        if len({item["listing_id"] for item in choice}) == len(names)]
        cheapest = min(combinations, key=lambda choice: (sum(float(item["price"]) for item in choice), tuple(item["listing_id"] for item in choice))) if combinations else None
        cost = sum(float(item["price"]) for item in cheapest) if cheapest else None
        budget = self.plan.constraints["total_budget"]
        affordable = cheapest is not None and (budget is None or cost <= budget)
        self.items = list(cheapest) if affordable else []
        yield self._finish(step, "completed", "已比较各选一本的可检索最低组合。")
        yield self._start(self.plan.steps[-1])
        lines = ["以下为模拟数据，每个书名各选一本；比较的是工具实际返回的记录，不代表真实市场最低价。",
                 "书名采用近名检索，可能包含辅导书；请核对列出的书名、版本及教材类型。本次仅查询比价，不下单或联系卖家。"]
        if cheapest:
            for name, item in zip(names, cheapest):
                lines.append(f"- {name} → {item['book']}，{item.get('edition') or '版本未注明'}：{item['price']:g} 元（{item['listing_id']}）；{item.get('campus') or '校区未注明'}。")
            lines.append(f"最低组合总价：{cost:g} 元。")
            if budget is not None:
                lines.append(f"总预算：{budget:g} 元；" + (f"符合预算，剩余 {budget - cost:g} 元。" if affordable else f"超出 {cost - budget:g} 元，没有符合预算的组合。"))
        else:
            missing = [name for name, pool in zip(names, pools) if not pool]
            lines.append("没有找到完整组合。" + ("未命中的模拟书名：" + "、".join(missing) + "。" if missing else "候选记录不能重复用于不同书名。"))
        self.answer = "\n".join(lines)
        yield self._finish(self.plan.steps[-1], "completed", "比价完成。" if affordable else "没有完整或符合预算的组合。")


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise WorkflowError("模拟数据含无效数值，未继续推荐。")
    return float(value)


def _minutes(value: object) -> int:
    if not isinstance(value, str) or not _CLOCK.fullmatch(value):
        raise ValueError("invalid clock")
    hour, minute = (int(part) for part in value.split(":"))
    return hour * 60 + minute


def _meeting(record: dict[str, Any]) -> tuple[str, int, int, int, int]:
    try:
        day = record["weekday"]
        start, end = _minutes(record["start_time"]), _minutes(record["end_time"])
        first, last = record["week_start"], record["week_end"]
        if day not in {"周一", "周二", "周三", "周四", "周五", "周六", "周日"} or start >= end:
            raise ValueError("invalid interval")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (first, last)) or not 1 <= first <= last <= 30:
            raise ValueError("invalid weeks")
        return day, start, end, first, last
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkflowError("课程时间或周次数据不完整，无法证明无冲突。") from exc


def _scheduled_meeting(record: dict[str, Any]) -> tuple[str, int, int, int, int]:
    weeks = record.get("weeks")
    match = re.fullmatch(r"(\d+)(?:[-–](\d+))?周", weeks) if isinstance(weeks, str) else None
    if not match:
        raise WorkflowError("学生课表周次格式不完整，无法证明无冲突。")
    return _meeting({**record, "week_start": int(match.group(1)), "week_end": int(match.group(2) or match.group(1))})


def meetings_conflict(left: dict[str, Any], right: dict[str, Any]) -> bool:
    a, b = _meeting(left), _meeting(right)
    return a[0] == b[0] and max(a[1], b[1]) < min(a[2], b[2]) and max(a[3], b[3]) <= min(a[4], b[4])


def filter_course_offerings(schedule: list[dict[str, Any]], offerings: list[dict[str, Any]],
                           reviews: list[dict[str, Any]], constraints: dict[str, Any],
                           campus: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Evidence-only filtering. Unknown candidate fields are not successful matches."""

    occupied = [_scheduled_meeting(item) for item in schedule]
    semesters = {item.get("semester") for item in schedule}
    if schedule and (len(semesters) != 1 or not next(iter(semesters), None)):
        raise WorkflowError("学生课表学期缺失或不一致。")
    semester = next(iter(semesters), None)
    enrolled = {item.get("course_name") for item in schedule}
    review_map: dict[str, dict[str, Any]] = {}
    for review in reviews:
        key = review.get("course_id")
        if not isinstance(key, str) or not key or key in review_map:
            raise WorkflowError("课程评价编号缺失或重复，无法可靠连接开课表。")
        review_map[key] = review
    matches: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    seen: set[str] = set()
    for offering in offerings:
        reason = ""
        offering_id = offering.get("offering_id")
        if not isinstance(offering_id, str) or not offering_id or offering_id in seen:
            raise WorkflowError("开课班编号缺失或重复。")
        seen.add(offering_id)
        review = review_map.get(offering.get("course_id"))
        meetings = offering.get("meetings")
        try:
            parsed = [_meeting(item) for item in meetings] if isinstance(meetings, list) and meetings and all(isinstance(item, dict) for item in meetings) else []
        except WorkflowError:
            parsed = []
        if not parsed:
            reason = "缺少有效开课时间"
        elif offering.get("campus") != campus:
            reason = "校区不符"
        elif not semester or offering.get("semester") != semester:
            reason = "学期无法匹配"
        elif offering.get("course_name") in enrolled:
            reason = "已选同名课程"
        elif not review or review.get("course_name") != offering.get("course_name"):
            reason = "缺少可匹配评价"
        elif not offering.get("location"):
            reason = "缺少上课地点"
        else:
            try:
                rating = _number(review.get("rating"))
                count = _number(review.get("review_count"))
                if not 0 <= rating <= 5 or count < 1 or not count.is_integer():
                    raise WorkflowError("invalid rating")
            except WorkflowError:
                reason = "无有效评分"
            if not reason and rating < constraints["min_rating"]:
                reason = "评分不足"
            if not reason and constraints["no_attendance"] and review.get("attendance_requirement") != "none":
                reason = "非明确不点名"
            if not reason and constraints["low_workload"] and review.get("workload") not in {"低", "较低"}:
                reason = "非低工作量"
            target_day = constraints.get("weekday")
            window = tuple(_minutes(text) for text in constraints["time_range"]) if constraints.get("time_range") else _WINDOWS.get(constraints.get("period"))
            if not reason and (target_day or window):
                if constraints["time_mode"] == "avoid":
                    if any((not target_day or meeting[0] == target_day) and (not window or max(meeting[1], window[0]) < min(meeting[2], window[1])) for meeting in parsed):
                        reason = "位于排除时段"
                elif not all((not target_day or meeting[0] == target_day) and (not window or (window[0] <= meeting[1] < meeting[2] <= window[1])) for meeting in parsed):
                    reason = "不在指定时段"
            if not reason and any(a[0] == b[0] and max(a[1], b[1]) < min(a[2], b[2]) and max(a[3], b[3]) <= min(a[4], b[4]) for a in parsed for b in occupied):
                reason = "与已选课程冲突"
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
        else:
            matches.append({**offering, "rating": rating, "review_count": int(count),
                            "workload": review.get("workload", "未知"),
                            "attendance_policy": review.get("attendance_policy", "点名要求未注明")})
    matches.sort(key=lambda item: (-item["rating"], -item["review_count"], item["offering_id"]))
    # Keep one alternative per course; do not imply multiple sections can be enrolled together.
    distinct: list[dict[str, Any]] = []
    seen_courses: set[str] = set()
    for item in matches:
        if item["course_id"] not in seen_courses:
            distinct.append(item)
            seen_courses.add(item["course_id"])
        else:
            reasons["同课程其他备选班"] = reasons.get("同课程其他备选班", 0) + 1
    return distinct, reasons
