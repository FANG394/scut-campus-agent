from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from campus_agent.config import AppConfig
from campus_agent.mcp.schemas import (
    DATA_MODE_DEMO,
    DATA_MODE_KNOWLEDGE,
    DEMO_WARNING,
    ToolEnvelope,
    envelope,
)
from campus_agent.rag.service import KnowledgeBase


MAX_DATA_FILE_BYTES = 2_000_000
_BOOK_QUERY_ALIASES = {
    "高数": "高等数学",
    "线代": "线性代数",
    "概统": "概率论与数理统计",
    "概率统计": "概率论与数理统计",
    "大物": "大学物理",
    "大英": "大学英语",
    "c语言": "c程序设计",
    "c语言程序设计": "c程序设计",
    "计网": "计算机网络",
    "计组": "计算机组成原理",
    "模电": "模拟电子技术",
    "数电": "数字电子技术",
    "自控": "自动控制原理",
    "马原": "马克思主义基本原理",
    "毛概": "毛泽东思想和中国特色社会主义理论体系概论",
    "近代史": "中国近现代史纲要",
    "思修": "思想道德与法治",
}


class ToolDataError(RuntimeError):
    """Raised when a fixed local demo dataset is unavailable or malformed."""


def normalize_book_query(value: str) -> str:
    """Return the same case-insensitive title/abbreviation used by book search."""

    query = _required_text(value, "book_name", maximum=100).casefold()
    return _BOOK_QUERY_ALIASES.get(query, query)


class CampusToolRepository:
    """Read-only access layer for the MCP campus tools and demo course planning."""

    def __init__(self, config: AppConfig, knowledge_base: KnowledgeBase) -> None:
        self.config = config
        self.knowledge_base = knowledge_base

    def campus_knowledge_search(self, query: str, limit: int = 4) -> ToolEnvelope:
        query = _required_text(query, "query", maximum=500)
        limit = _bounded_int(limit, default=4, minimum=1, maximum=8)
        hits = self.knowledge_base.search(query, top_k=limit)
        items = [hit.as_source(index) for index, hit in enumerate(hits, start=1)]
        return envelope(
            ok=True,
            tool="campus_knowledge_search",
            data_mode=DATA_MODE_KNOWLEDGE,
            items=items,
            message=(
                f"在本地校园知识库中找到 {len(items)} 条相关内容。"
                if items
                else "本地校园知识库中没有找到足以回答该问题的内容。"
            ),
            warnings=[] if items else ["未命中可靠知识片段，请勿根据常识猜测校园事实。"],
        )

    def secondhand_book_search(
        self, book_name: str, price_limit: float | None = None
    ) -> ToolEnvelope:
        query = normalize_book_query(book_name)
        price = _optional_price(price_limit)
        records = self._load_list("secondhand_books.json", "books")
        matches: list[dict[str, Any]] = []
        for record in records:
            searchable = " ".join(
                str(record.get(key, ""))
                for key in ("book", "edition", "author", "category")
            ).casefold()
            if query not in searchable:
                continue
            record_price = _record_number(record, "price")
            if price is not None and record_price > price:
                continue
            matches.append(_public_book(record))
        matches.sort(key=lambda item: (float(item["price"]), str(item["book"])))
        suffix = f"且价格不超过 {price:g} 元" if price is not None else ""
        return envelope(
            ok=True,
            tool="secondhand_book_search",
            data_mode=DATA_MODE_DEMO,
            items=matches[:10],
            message=(
                f"找到 {len(matches[:10])} 本符合“{book_name.strip()}”{suffix}的模拟二手书。"
                if matches
                else f"没有找到符合“{book_name.strip()}”{suffix}的模拟二手书。"
            ),
            warnings=[DEMO_WARNING],
        )

    def student_schedule_query(
        self, student_id: str, weekday: str | None = None
    ) -> ToolEnvelope:
        student_alias = _required_text(student_id, "student_id", maximum=30).upper()
        if not student_alias.startswith("DEMO"):
            return envelope(
                ok=False,
                tool="student_schedule_query",
                data_mode=DATA_MODE_DEMO,
                message="演示环境只接受 DEMO001、DEMO002 等匿名演示编号，请勿输入真实学号。",
                warnings=[DEMO_WARNING],
            )
        normalized_weekday = _normalise_weekday(weekday)
        students = self._load_list("student_schedules.json", "students")
        student = next(
            (
                record
                for record in students
                if str(record.get("student_alias", "")).upper() == student_alias
            ),
            None,
        )
        if student is None:
            return envelope(
                ok=False,
                tool="student_schedule_query",
                data_mode=DATA_MODE_DEMO,
                message=f"没有找到匿名演示编号 {student_alias} 的课表。",
                warnings=[DEMO_WARNING],
            )
        courses = student.get("courses")
        if not isinstance(courses, list):
            raise ToolDataError("student_schedules.json 中 courses 必须是数组。")
        items = [
            _public_schedule_item(student, course)
            for course in courses
            if isinstance(course, dict)
            and (
                normalized_weekday is None
                or str(course.get("weekday", "")) == normalized_weekday
            )
        ]
        items.sort(key=lambda item: (str(item["weekday"]), str(item["start_time"])))
        scope = f"{normalized_weekday}的" if normalized_weekday else "本学期的"
        return envelope(
            ok=True,
            tool="student_schedule_query",
            data_mode=DATA_MODE_DEMO,
            items=items,
            message=f"查到 {student_alias} {scope}{len(items)} 条模拟课程安排。",
            warnings=[DEMO_WARNING, "演示编号不对应任何真实学生。"],
        )

    def course_review_search(self, course_name: str) -> ToolEnvelope:
        query = _required_text(course_name, "course_name", maximum=100).casefold()
        reviews = self._load_list("course_reviews.json", "courses")
        matches = [
            _public_review(record)
            for record in reviews
            if query in str(record.get("course_name", "")).casefold()
            or query in " ".join(str(tag) for tag in record.get("tags", [])).casefold()
        ]
        matches.sort(key=lambda item: (-float(item["rating"]), str(item["course_name"])))
        return envelope(
            ok=True,
            tool="course_review_search",
            data_mode=DATA_MODE_DEMO,
            items=matches[:10],
            message=(
                f"找到 {len(matches[:10])} 条与“{course_name.strip()}”相关的模拟课程评价。"
                if matches
                else f"没有找到与“{course_name.strip()}”相关的模拟课程评价。"
            ),
            warnings=[DEMO_WARNING],
        )

    def course_offering_search(
        self, course_name: str = "通识", campus: str | None = None
    ) -> ToolEnvelope:
        query = _required_text(course_name, "course_name", maximum=100).casefold()
        campus_query = None
        if campus is not None:
            campus_query = _required_text(campus, "campus", maximum=50)
            campus_query = {
                "五山": "五山校区",
                "大学城": "大学城校区",
                "广州国际": "广州国际校区",
            }.get(campus_query, campus_query)
        records = self._load_list("course_offerings.json", "offerings")
        # Validate the whole dataset before filtering: malformed records must not
        # silently disappear or turn a failed availability query into an empty list.
        offerings = [_public_offering(record) for record in records]
        ids = [item["offering_id"] for item in offerings]
        if len(ids) != len(set(ids)):
            raise ToolDataError("course_offerings.json 中 offering_id 必须唯一。")
        matches = [
            item for item in offerings
            if (
                query in item["course_name"].casefold()
                or query in item["category"].casefold()
            ) and (campus_query is None or item["campus"] == campus_query)
        ]
        matches.sort(key=lambda item: (item["course_name"], item["offering_id"]))
        items = matches[:30]
        warnings = [DEMO_WARNING]
        if any(not item["meetings"] for item in items):
            warnings.append("部分模拟开课班尚未公布上课时间，不能据此判断无时间冲突。")
        if len(matches) > 30:
            warnings.append("只返回前 30 个模拟开课班，不代表完整开课清单。")
        return envelope(
            ok=True,
            tool="course_offering_search",
            data_mode=DATA_MODE_DEMO,
            items=items,
            message=f"找到 {len(items)} 个与“{course_name.strip()}”相关的模拟开课班。",
            warnings=warnings,
        )

    def _load_list(self, file_name: str, key: str) -> list[dict[str, Any]]:
        path = (self.config.tool_data_dir / file_name).resolve()
        try:
            path.relative_to(self.config.tool_data_dir.resolve())
        except ValueError as exc:
            raise ToolDataError("工具数据路径超出配置目录。") from exc
        if not path.is_file():
            raise ToolDataError(f"缺少工具模拟数据文件：{file_name}。")
        if path.stat().st_size > MAX_DATA_FILE_BYTES:
            raise ToolDataError(f"工具模拟数据文件过大：{file_name}。")
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ToolDataError(f"无法读取工具模拟数据：{file_name}。") from exc
        records = payload.get(key) if isinstance(payload, dict) else None
        if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
            raise ToolDataError(f"{file_name} 中 {key} 必须是对象数组。")
        return records


def _required_text(value: object, field: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"参数 {field} 不能为空。")
    compact = " ".join(value.strip().split())
    if len(compact) > maximum:
        raise ValueError(f"参数 {field} 不能超过 {maximum} 个字符。")
    return compact


def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return min(max(parsed, minimum), maximum)


def _optional_price(value: object) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("参数 price_limit 必须是数字。")
    try:
        price = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("参数 price_limit 必须是数字。") from exc
    if not 0 <= price <= 10000:
        raise ValueError("参数 price_limit 必须在 0 到 10000 之间。")
    return price


def _record_number(record: dict[str, Any], key: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolDataError(f"模拟数据字段 {key} 必须是数字。")
    return float(value)


def _public_book(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record.get(key)
        for key in (
            "listing_id",
            "book",
            "category",
            "edition",
            "author",
            "price",
            "condition",
            "campus",
            "pickup",
            "seller_alias",
            "posted_at",
        )
    }


def _normalise_weekday(value: object) -> str | None:
    if value is None or value == "":
        return None
    text = str(value).strip()
    aliases = {
        "1": "周一", "一": "周一", "星期一": "周一", "周一": "周一",
        "2": "周二", "二": "周二", "星期二": "周二", "周二": "周二",
        "3": "周三", "三": "周三", "星期三": "周三", "周三": "周三",
        "4": "周四", "四": "周四", "星期四": "周四", "周四": "周四",
        "5": "周五", "五": "周五", "星期五": "周五", "周五": "周五",
        "6": "周六", "六": "周六", "星期六": "周六", "周六": "周六",
        "7": "周日", "日": "周日", "天": "周日", "星期日": "周日", "星期天": "周日", "周日": "周日",
    }
    if text not in aliases:
        raise ValueError("参数 weekday 应为周一到周日。")
    return aliases[text]


def _public_schedule_item(
    student: dict[str, Any], course: dict[str, Any]
) -> dict[str, Any]:
    return {
        "student_alias": student.get("student_alias"),
        "semester": student.get("semester"),
        "course_name": course.get("course_name"),
        "weekday": course.get("weekday"),
        "period": course.get("period"),
        "start_time": course.get("start_time"),
        "end_time": course.get("end_time"),
        "location": course.get("location"),
        "campus": course.get("campus"),
        "weeks": course.get("weeks"),
    }


def _public_review(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record.get(key)
        for key in (
            "course_id",
            "course_name",
            "category",
            "rating",
            "review_count",
            "workload",
            "attendance_policy",
            "attendance_requirement",
            "assessment",
            "tags",
            "summary",
        )
    }


def _public_offering(record: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "offering_id", "course_id", "course_name", "category", "semester", "campus", "location"
    )
    public: dict[str, Any] = {}
    for key in fields:
        value = record.get(key)
        if not isinstance(value, str) or not value.strip() or len(value) > 200:
            raise ToolDataError(f"模拟开课数据字段 {key} 必须是非空文本。")
        public[key] = value.strip()
    if not re.fullmatch(r"\d{4}-\d{4}-[12]", public["semester"]):
        raise ToolDataError("模拟开课数据 semester 格式必须为 YYYY-YYYY-1/2。")
    meetings = record.get("meetings")
    if not isinstance(meetings, list) or len(meetings) > 12:
        raise ToolDataError("模拟开课数据 meetings 必须是最多 12 项的数组。")
    public_meetings = []
    for meeting in meetings:
        if not isinstance(meeting, dict):
            raise ToolDataError("模拟开课数据 meeting 必须是对象。")
        day = meeting.get("weekday")
        if day not in {"周一", "周二", "周三", "周四", "周五", "周六", "周日"}:
            raise ToolDataError("模拟开课数据 weekday 必须为周一到周日。")
        start = _offering_time(meeting.get("start_time"), "start_time")
        end = _offering_time(meeting.get("end_time"), "end_time")
        if start >= end:
            raise ToolDataError("模拟开课数据 start_time 必须早于 end_time。")
        first, last = meeting.get("week_start"), meeting.get("week_end")
        if (
            isinstance(first, bool) or isinstance(last, bool)
            or not isinstance(first, int) or not isinstance(last, int)
            or not 1 <= first <= last <= 30
        ):
            raise ToolDataError("模拟开课数据 week range 必须为 1 到 30 的递增整数。")
        public_meetings.append({
            "weekday": day, "start_time": meeting["start_time"],
            "end_time": meeting["end_time"], "week_start": first, "week_end": last,
        })
    public["meetings"] = public_meetings
    return public


def _offering_time(value: object, field: str) -> int:
    if not isinstance(value, str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
        raise ToolDataError(f"模拟开课数据 {field} 必须为有效 HH:MM 时间。")
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)
