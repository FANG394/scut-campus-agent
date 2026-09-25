from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from campus_agent.safety import redact_sensitive_text


TOOL_DISPLAY_NAMES = {
    "campus_knowledge_search": "校园知识查询",
    "secondhand_book_search": "二手书查询",
    "student_schedule_query": "课程表查询",
    "course_review_search": "课程评价查询",
    "course_offering_search": "开课时间查询",
}

_BOOK_NOUN = re.compile(r"二手(?:书|教材)|旧(?:书|教材)")
_BOOK_TRANSACTION = re.compile(
    r"(?:有(?:没有|没)?|有人|谁)(?:在)?卖|"
    r"(?:正在)?(?:出售|转卖|转让)|在售|有货|"
    r"(?:出|卖|收)(?:一)?(?:本|套)|求购|"
    r"(?:想|要|准备|打算)(?:买|收)|"
    r"(?:哪里|哪儿|哪)(?:能|可以|可)?买(?:到)?|"
    r"能(?:不能|否)?买到|可(?:不可以)?买到|买得到"
)
_BOOK_INTENT = re.compile(
    rf"{_BOOK_NOUN.pattern}|{_BOOK_TRANSACTION.pattern}"
)
_SCHEDULE_INTENT = re.compile(r"课表|课程安排|(?:周|星期)[一二三四五六日天].{0,10}(?:有|上|没).{0,3}课")
_REVIEW_INTENT = re.compile(r"课程评价|课程评分|选课评价|(?:这门|这节).{0,4}课.{0,5}(?:怎么样|评价|评分)|(?:评价|评分|口碑).{0,8}(?:课程|课)")
_KNOWLEDGE_TOOL_INTENT = re.compile(
    r"(?:用|调用).{0,10}(?:校园知识|知识库).{0,8}(?:工具|查询|搜索)|"
    r"(?:校园知识|学校信息).{0,4}(?:查询|搜索)"
)
_OFFERING_INTENT = re.compile(r"开课(?:时间|安排|班|信息)|(?:查|搜|看看).{0,20}(?:开课|可选课程)")
_PHASE4_EXPLICIT = re.compile(r"任务规划|多\s*Agent|多智能体|多工具|连续调用", re.I)
_COURSE_RECOMMENDATION = re.compile(r"(?:推荐|挑选|选择).{0,30}(?:通识课|课程|选修课)|课程推荐")
_SCHEDULE_CONSTRAINT = re.compile(r"没课|空闲|不冲突|周[一二三四五六日天]|星期[一二三四五六日天]|上午|下午|晚上")
_REVIEW_CONSTRAINT = re.compile(r"评分|评价|点名|作业|考核|轻松|给分|工作量")
_PRICE = re.compile(
    r"(?:(?:价格|预算).{0,3})?(\d+(?:\.\d+)?)\s*元?\s*(?:以内|以下|之内|封顶|不超过|最多)?"
)
_PRICE_WITH_LIMIT = re.compile(
    r"(?:(?:价格|预算)\s*[：:]?\s*)?(\d+(?:\.\d+)?)\s*元?\s*(?:以内|以下|之内|封顶|不超过|最多)"
    r"|(?:价格\s*)?(?:不超过|最多)\s*(\d+(?:\.\d+)?)\s*元?"
    r"|预算\s*[：:]?\s*(\d+(?:\.\d+)?)\s*元?"
)
_WEEKDAY = re.compile(r"(?:周|星期)([一二三四五六日天])")
_DEMO_ID = re.compile(r"DEMO\d{3}", re.I)
_QUOTED = re.compile(r"[《“\"]([^》”\"]{1,80})[》”\"]")
_BOOK_CONTEXT_FOLLOW_UP = re.compile(
    r"^(?:那|那么|换成|换一本|再(?:查|找|搜|看看))|"
    r"^还有.{1,30}(?:吗|呢)[？?！!。.]?$|"
    r"(?:有吗|有没有|还有吗|有货吗|多少钱|什么价|怎么卖|呢)[？?！!。.]?$"
)
_BOOK_CONTEXT_INCOMPLETE = re.compile(
    r"^(?:那(?:本|个)?(?:呢)?|还有(?:货)?(?:吗)?|有货吗|多少钱|什么价|怎么卖)"
    r"[？?！!。.]?$"
)
_TRAILING_QUESTION_PARTICLES = re.compile(
    r"(?:多少钱|什么价|怎么卖|还有货|有货|有卖|在售|有售|还有|有|卖)?"
    r"(?:吗|么|呢|呀|啊|嘛|吧|不|没有)?[？?！!。.]?$"
)
_INVALID_BOOK_NAMES = frozenset(
    {
        "",
        "吗",
        "么",
        "呢",
        "呀",
        "啊",
        "嘛",
        "吧",
        "有吗",
        "有没有",
        "有卖吗",
        "还有吗",
        "有货吗",
        "书",
        "一本书",
        "教材",
        "一本教材",
        "二手书",
        "二手教材",
        "旧书",
        "旧教材",
        "这个",
        "那个",
        "一本",
        "一套",
        "谢谢",
        "好的",
        "好",
        "算了",
        "还",
        "多少钱",
        "什么价",
        "怎么卖",
        "有货",
        "还有货",
        "什么",
        "有什么",
        "都有什么",
        "哪些",
        "有哪些",
        "有啥",
        "哪本",
        "哪一本",
    }
)


@dataclass(frozen=True, slots=True)
class ToolRoute:
    name: str
    arguments: dict[str, Any]

    @property
    def display_name(self) -> str:
        return TOOL_DISPLAY_NAMES[self.name]

    def public_arguments(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in self.arguments.items():
            if isinstance(value, str):
                result[key] = redact_sensitive_text(value)
            elif isinstance(value, (int, float, bool)) or value is None:
                result[key] = value
        return result


def is_phase4_request(question: str) -> bool:
    """Recognize planning requests, including unsupported phase-four workflows."""

    if _PHASE4_EXPLICIT.search(question):
        return True
    if re.search(r"然后|再|并|同时|以及|和", question):
        intents = [bool(_BOOK_INTENT.search(question)), bool(re.search(r"课表|课程安排", question)),
                   bool(_REVIEW_INTENT.search(question)), bool(_KNOWLEDGE_TOOL_INTENT.search(question))]
        if sum(intents) > 1:
            return True
    if not _COURSE_RECOMMENDATION.search(question):
        return False
    return bool(_SCHEDULE_CONSTRAINT.search(question) or _REVIEW_CONSTRAINT.search(question)) or bool(
        re.search(r"(?:通识课|课程|选修课)", question)
    )


def route_tool_request(question: str, tool_hint: str | None = None) -> ToolRoute | None:
    """Route one question, optionally using the most recent tool as context.

    ``tool_hint`` is deliberately only a fallback. An explicit schedule, course
    review, or knowledge-base request in the current question always wins over a
    stale hint from an earlier turn.
    """

    if _BOOK_INTENT.search(question):
        return _build_book_route(question, contextual=False)

    if _OFFERING_INTENT.search(question):
        quoted = _QUOTED.search(question)
        query = quoted.group(1).strip() if quoted else "通识"
        arguments: dict[str, Any] = {"course_name": query}
        campus = re.search(r"五山(?:校区)?|大学城(?:校区)?|广州国际(?:校区)?", question)
        if campus:
            arguments["campus"] = campus.group(0).removesuffix("校区") + "校区"
        return ToolRoute("course_offering_search", arguments)

    if _SCHEDULE_INTENT.search(question):
        student_match = _DEMO_ID.search(question)
        weekday_match = _WEEKDAY.search(question)
        arguments = {
            "student_id": student_match.group(0).upper() if student_match else "DEMO001"
        }
        if weekday_match:
            day = "日" if weekday_match.group(1) == "天" else weekday_match.group(1)
            arguments["weekday"] = f"周{day}"
        return ToolRoute("student_schedule_query", arguments)

    if _REVIEW_INTENT.search(question):
        return ToolRoute(
            "course_review_search",
            {"course_name": _extract_course_name(question)},
        )

    if _KNOWLEDGE_TOOL_INTENT.search(question):
        return ToolRoute(
            "campus_knowledge_search",
            {"query": _extract_knowledge_query(question), "limit": 4},
        )

    if tool_hint == "secondhand_book_search" and is_book_transaction_request(
        question, contextual=True
    ):
        return _build_book_route(question, contextual=True)
    return None


def is_book_transaction_request(question: str, *, contextual: bool = False) -> bool:
    """Return whether a question needs secondhand-book transaction data.

    With ``contextual=True``, short follow-ups such as ``那高数呢`` and
    ``概率论有吗`` are accepted. Explicit intents for other tools are excluded so
    this helper is safe to use as an Agent-side no-hallucination guard.
    """

    if _BOOK_INTENT.search(question):
        return True
    if not contextual:
        return False
    if (
        _SCHEDULE_INTENT.search(question)
        or _REVIEW_INTENT.search(question)
        or _KNOWLEDGE_TOOL_INTENT.search(question)
        or _OFFERING_INTENT.search(question)
    ):
        return False
    if len(question.strip()) > 50:
        return False
    if _BOOK_CONTEXT_INCOMPLETE.fullmatch(question.strip()):
        return True
    return bool(
        _BOOK_CONTEXT_FOLLOW_UP.search(question)
        and _extract_book_name(question, contextual=True)
    )


def _build_book_route(question: str, *, contextual: bool) -> ToolRoute | None:
    book_name = _extract_book_name(question, contextual=contextual)
    if not book_name:
        return None
    price_match = _PRICE_WITH_LIMIT.search(question)
    raw_price = (
        next((group for group in price_match.groups() if group), None)
        if price_match
        else None
    )
    arguments: dict[str, Any] = {"book_name": book_name}
    if raw_price is not None:
        arguments["price_limit"] = float(raw_price)
    return ToolRoute("secondhand_book_search", arguments)


def _extract_book_name(question: str, *, contextual: bool = False) -> str | None:
    quoted = _QUOTED.search(question)
    if quoted:
        candidate = _clean_subject(quoted.group(1))
        return candidate if _is_valid_book_name(candidate) else None

    cleaned = _PRICE_WITH_LIMIT.sub(" ", question)
    # Remove high-signal transaction phrases before individual request words.
    # Longest-first alternatives prevent a residual "买到" or "卖" becoming a
    # bogus part of the title.
    cleaned = re.sub(
        r"请问|麻烦(?:你)?|劳驾|帮我|替我|给我|我想问(?:一下)?|问一下",
        " ",
        cleaned,
    )
    cleaned = re.sub(
        r"有(?:没有|没)?(?:人)?(?:在)?卖|有人(?:在)?卖|谁(?:在)?卖|"
        r"哪里(?:能|可以|可)?买(?:到)?|哪儿(?:能|可以|可)?买(?:到)?|"
        r"哪(?:能|可以|可)?买(?:到)?|能(?:不能|否)?买到|"
        r"可(?:不可以)?买到|买得到|"
        r"(?:正在)?(?:出售|转卖|转让)|在售|有售|(?:还)?有货|"
        r"(?:想|要|准备|打算)(?:买|收)|求购|"
        r"(?:出|卖|收)(?:一)?(?:本|套)",
        " ",
        cleaned,
    )
    cleaned = _BOOK_NOUN.sub(" ", cleaned)
    cleaned = re.sub(
        r"(?:帮忙)?(?:查找|查询|搜索|搜寻|搜|找找|找|看看|看下|查|推荐)(?:一下)?|"
        r"(?:我)?(?:要|想要)|(?:一|这|那)(?:本|套)",
        " ",
        cleaned,
    )
    cleaned = re.sub(r"^\s*我\s*", " ", cleaned)
    cleaned = re.sub(r"(?:教材|课本|书籍|书)\s*(?=[，。！？；：:、,.!?]*$)", " ", cleaned)
    cleaned = re.sub(r"(?:预算|价格)\s*[：:]?\s*", " ", cleaned)
    cleaned = re.sub(
        r"^(?:那|那么|还有|换成|换一本|再(?:查|找|搜|看看))\s*",
        " ",
        cleaned,
    )
    cleaned = re.sub(r"^\s*的\s*|\s*的\s*$", " ", cleaned)
    cleaned = _TRAILING_QUESTION_PARTICLES.sub("", cleaned)
    candidate = _clean_subject(cleaned)
    candidate = re.sub(r"^(?:的\s*)+|(?:\s*的)+$", "", candidate).strip()
    if not _is_valid_book_name(candidate):
        return None
    if not contextual and not _BOOK_INTENT.search(question):
        return None
    return candidate


def extract_book_title(question: str) -> str | None:
    """Shared title cleanup for independently specified workflow titles."""
    return _extract_book_name(question, contextual=True)


def _is_valid_book_name(value: str | None) -> bool:
    if value is None:
        return False
    normalized = _clean_subject(value)
    if normalized in _INVALID_BOOK_NAMES:
        return False
    if re.fullmatch(
        r"(?:有|没|没有|卖|买|找|查|搜|要|想|问|的|这|那|哪|啥|什么|哪些)+",
        normalized,
    ):
        return False
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z0-9]", normalized))


def _extract_course_name(question: str) -> str:
    quoted = _QUOTED.search(question)
    if quoted:
        return quoted.group(1).strip()
    patterns = (
        re.compile(r"(?:查|看看|想知道|了解)?\s*(.{2,50}?)(?:这门|这节)?课?(?:的)?(?:课程评价|课程评分|评价|评分|口碑|怎么样)"),
        re.compile(r"(?:课程评价|课程评分|评价|评分|口碑)[：:\s]*(.{2,50})"),
    )
    for pattern in patterns:
        match = pattern.search(question)
        if match:
            candidate = _clean_subject(match.group(1))
            if candidate:
                return candidate
    cleaned = re.sub(
        r"帮我|请|查|查询|搜索|看看|想知道|了解|这门|这节|课程|课|的|"
        r"评价|评分|口碑|怎么样|如何|吗|呢",
        " ",
        question,
    )
    return _clean_subject(cleaned) or "通识"


def _extract_knowledge_query(question: str) -> str:
    cleaned = re.sub(
        r"请|帮我|用|调用|校园知识|学校信息|知识库|工具|查询|搜索|一下",
        " ",
        question,
    )
    return _clean_subject(cleaned) or question.strip()


def _clean_subject(value: str) -> str:
    return " ".join(value.strip(" ，。！？；：:、\t\r\n").split())[:100]
