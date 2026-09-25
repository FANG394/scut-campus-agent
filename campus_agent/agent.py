from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from difflib import SequenceMatcher
from typing import Any, Literal, Protocol
from uuid import uuid4

from campus_agent.config import AppConfig
from campus_agent.llm import LLMError, LLMResult, create_llm_client
from campus_agent.models import (
    ChatMessage,
    ChatResult,
    ChatStreamPlan,
    ChatStreamUpdate,
    SearchHit,
    ToolCallTrace,
)
from campus_agent.multi_agent import MultiAgentCoordinator, MultiAgentOutcome
from campus_agent.mcp.client import MCPDependencyError, MCPToolClient, MCPToolError
from campus_agent.mcp.schemas import DATA_MODE_DEMO, ToolEnvelope
from campus_agent.prompts import (
    SYSTEM_PROMPT,
    TOOL_RESULT_SYSTEM_PROMPT,
    build_grounded_input,
    build_tool_result_input,
    llm_messages,
)
from campus_agent.rag.service import KnowledgeBase
from campus_agent.rag.tokenizer import normalize, token_counts
from campus_agent.safety import detect_sensitive_disclosure
from campus_agent.sessions import SessionStore
from campus_agent.planning import TaskPlan
from campus_agent.tool_router import (
    ToolRoute,
    is_book_transaction_request,
    is_phase4_request,
    route_tool_request,
)


_CAMPUS_KEYWORDS = {
    "华工",
    "华南理工",
    "学校",
    "校园",
    "校区",
    "新生",
    "教务",
    "图书馆",
    "宿舍",
    "校园网",
    "统一认证",
    "一卡通",
    "食堂",
    "课程",
    "选课",
    "考试",
    "校历",
    "学籍",
    "学位",
    "学生证",
    "校园卡",
    "转专业",
    "奖学金",
    "体测",
    "体质测试",
    "自习室",
    "毕业",
    "处分",
    "校医院",
    "保卫处",
    "办事大厅",
}
_GREETING = re.compile(r"^(你好|您好|嗨|hi|hello|在吗|早上好|下午好|晚上好)[!！。,.，\s]*$", re.I)
_THANKS = re.compile(r"^(谢谢|感谢|多谢|好的|明白了|知道了)[!！。,.，\s]*$")
_IDENTITY = re.compile(r"(你是谁|你能做什么|有什么功能|介绍一下你)")
_LATER_CAPABILITY = re.compile(
    r"(?:主动提醒|定时提醒|到时提醒|自动提醒|提醒我|定时通知|主动通知)",
    re.IGNORECASE,
)
_CITATION = re.compile(r"\[来源(\d+)\]")
_FOLLOW_UP = re.compile(
    r"(呢|那|这个|它|上述|刚才|还有|具体|联系电话|电话|网址|入口|时间|费用|收费|地点|地址|怎么办)"
)
_CJK_TERM = re.compile(r"[\u3400-\u9fff]{2,3}")
_ASCII_TERM = re.compile(r"[a-z0-9]+(?:[-_.:/][a-z0-9]+)*", re.IGNORECASE)
_SPECIFIC_LITERAL = re.compile(
    r"https?://[^\s，。；！？]+|\d+(?:[.:/-]\d+)+|\d+(?:\.\d+)?|[a-z][a-z0-9_.:/-]{2,}",
    re.IGNORECASE,
)
_UNLABELLED_SCHEDULE_ID = re.compile(r"(?<!\d)\d{8,14}(?!\d)")
_SECONDHAND_BOOK_TOOL = "secondhand_book_search"
_BOOK_TRANSACTION_CLARIFICATION = (
    "这看起来是在询问二手书交易，但我还没有识别出可检索的明确书名。"
    "请补充书名，例如“有卖概率论二手书吗？”或“找一本30元以内的高等数学”。"
    "在取得二手书工具结果前，我不会猜测库存或价格。"
)
_NATURAL_ANSWER_UNAVAILABLE = (
    "本地聊天模型这次没有成功把检索内容整理成自然语言回答，请稍后重试。"
)
_GROUNDED_ANSWER_REJECTED = "生成的回答与知识库证据存在关键冲突，请换一种问法后重试。"
ValidationLevel = Literal["pass", "soft_fail", "hard_fail"]
_AFFIRMATIVE_PERMISSION = re.compile(r"(?<!不)(?<!未)(?:允许|可以|可供|有权)")
_NEGATIVE_PERMISSION = re.compile(r"不得|禁止|严禁|不允许|不可|不能|无权")
_TERM_NOISE = {
    "回答",
    "来源",
    "证据",
    "根据",
    "相关",
    "信息",
    "内容",
    "问题",
    "目前",
    "当前",
    "知识",
    "知识库",
}


class LLMClient(Protocol):
    """Structural type shared by remote and local chat-model adapters."""

    def generate(self, instructions: str, messages: list[ChatMessage]) -> LLMResult: ...


class ToolClient(Protocol):
    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolEnvelope: ...


class ChatService:
    def __init__(
        self,
        config: AppConfig,
        knowledge_base: KnowledgeBase | None = None,
        llm_client: LLMClient | None = None,
        sessions: SessionStore | None = None,
        tool_client: ToolClient | None = None,
    ) -> None:
        self.config = config
        self.knowledge_base = knowledge_base or KnowledgeBase(config)
        self.knowledge_base.ensure_ready()
        self.llm_client = llm_client if llm_client is not None else create_llm_client(config)
        self.sessions = sessions or SessionStore(max_messages=config.max_history_messages)
        self.tool_client = tool_client

    def chat(self, message: str, session_id: str | None = None) -> ChatResult:
        question = self._normalise_question(message)

        resolved_session = self.sessions.normalise_id(session_id)
        sensitive_categories = detect_sensitive_disclosure(question)
        if re.search(r"课表|课程安排|推荐|挑选|筛选", question) and _UNLABELLED_SCHEDULE_ID.search(question):
            sensitive_categories = [*sensitive_categories, "疑似真实学号"]
        if sensitive_categories:
            return ChatResult(
                answer=(
                    "检测到消息中可能包含敏感身份或认证信息。为保护隐私，这条消息不会保存、"
                    "不会发送给外部模型，也不会进入知识检索。请删除具体值后重新描述问题，"
                    "例如只问“如何修改统一认证密码？”。"
                ),
                session_id=resolved_session,
                mode="sensitive_input_blocked",
                provider="privacy-guard",
                knowledge_used=False,
                warnings=[f"已拦截的敏感信息类型：{', '.join(sensitive_categories)}。"],
            )
        if _LATER_CAPABILITY.search(question):
            result = ChatResult(
                answer=(
                    "主动提醒属于第五阶段之后的扩展，当前尚未实现。"
                    "目前的多 Agent 仅执行有边界的模拟课程推荐和多书比价，不会假装发送通知。"
                ),
                session_id=resolved_session,
                mode="not_implemented",
                provider="scope-guard",
                knowledge_used=False,
            )
            self.sessions.append_exchange(resolved_session, question, result.answer)
            return result
        history = self.sessions.history(resolved_session)
        tool_hint = self.sessions.recent_tool_name(resolved_session)
        coordinator = self._multi_agent(question, tool_hint)
        decision = coordinator.prepare()
        if decision.recognized:
            if decision.clarification or decision.plan is None:
                return self._task_clarification(
                    question, resolved_session, decision.plan, decision.clarification,
                    coordinator.planner_run(),
                )
            outcome = coordinator.run()
            return self._multi_agent_result(question, resolved_session, outcome)
        if is_phase4_request(question):
            result = ChatResult(
                answer="当前任务规划支持课程时间／评价联合推荐、多本二手书比价与预算筛选。这个组合任务还没有可校验的执行模板，请明确提供上述类型的需求；不会只执行部分工具后伪造完成结果。",
                session_id=resolved_session, mode="not_implemented", provider="scope-guard", knowledge_used=False,
            )
            self.sessions.append_exchange(resolved_session, question, result.answer)
            return result
        tool_route = route_tool_request(question, tool_hint=tool_hint)
        if tool_route is not None:
            return self._execute_tool(question, resolved_session, history, tool_route)
        if is_book_transaction_request(
            question, contextual=tool_hint == _SECONDHAND_BOOK_TOOL
        ):
            result = ChatResult(
                answer=_BOOK_TRANSACTION_CLARIFICATION,
                session_id=resolved_session,
                mode="clarification",
                provider="tool-routing-guard",
                knowledge_used=False,
                warnings=["未取得工具结果，已阻止模型猜测二手书库存或价格。"],
            )
            self.sessions.append_exchange(resolved_session, question, result.answer)
            return result
        retrieval_query = self._retrieval_query(question, history)
        hits = self.knowledge_base.search(retrieval_query)
        is_campus = self._is_campus_question(question, hits)
        warnings = self._configuration_warnings()

        if hits:
            answer, provider, selected_hits, generation_warnings = self._grounded_answer(
                question, history, hits
            )
            warnings.extend(generation_warnings)
            warnings.extend(_source_reliability_warnings(selected_hits))
            result = ChatResult(
                answer=answer,
                session_id=resolved_session,
                mode="rag",
                provider=provider,
                knowledge_used=True,
                sources=[hit.as_source(index) for index, hit in enumerate(selected_hits, start=1)],
                warnings=warnings,
            )
        elif is_campus:
            result = ChatResult(
                answer=(
                    "当前知识库没有检索到足以回答这个校园问题的可靠依据，因此我不能直接猜测。"
                    "请补充具体业务、校区或适用年级，或通过学校官方入口核验；也可以把对应官方文档加入知识库后重试。"
                ),
                session_id=resolved_session,
                mode="knowledge_gap",
                provider="policy-guard",
                knowledge_used=False,
                warnings=warnings,
            )
        else:
            answer, provider, chat_warnings = self._general_answer(question, history)
            warnings.extend(chat_warnings)
            result = ChatResult(
                answer=answer,
                session_id=resolved_session,
                mode="chat",
                provider=provider,
                knowledge_used=False,
                warnings=warnings,
            )

        self.sessions.append_exchange(resolved_session, question, result.answer)
        return result

    @staticmethod
    def _normalise_question(message: str) -> str:
        question = " ".join(message.strip().split())
        if not question:
            raise ValueError("消息不能为空。")
        if len(question) > 2000:
            raise ValueError("单条消息不能超过 2000 个字符。")
        return question

    def stream_chat(
        self, message: str, session_id: str | None = None
    ) -> ChatStreamPlan:
        """Prepare a native model stream while retaining a validated final answer."""

        question = self._normalise_question(message)
        resolved_session = self.sessions.normalise_id(session_id)
        if (detect_sensitive_disclosure(question)
                or (re.search(r"课表|课程安排|推荐|挑选|筛选", question) and _UNLABELLED_SCHEDULE_ID.search(question))
                or _LATER_CAPABILITY.search(question)):
            return _static_stream_plan(self.chat(question, resolved_session))
        tool_hint = self.sessions.recent_tool_name(resolved_session)
        coordinator = self._multi_agent(question, tool_hint)
        decision = coordinator.prepare()
        if decision.recognized:
            if decision.clarification or decision.plan is None:
                return _static_stream_plan(self._task_clarification(
                    question, resolved_session, decision.plan, decision.clarification,
                    coordinator.planner_run(),
                ))

            def task_updates() -> Iterator[ChatStreamUpdate]:
                yield from coordinator.updates()
                assert coordinator.outcome is not None
                self._multi_agent_result(question, resolved_session, coordinator.outcome)

            return ChatStreamPlan(session_id=resolved_session, mode="multi_agent", provider="planner+executor+reviewer+mcp",
                                  knowledge_used=False, updates=task_updates())
        tool_route = route_tool_request(question, tool_hint=tool_hint)
        needs_book_clarification = is_book_transaction_request(
            question, contextual=tool_hint == _SECONDHAND_BOOK_TOOL
        )
        stream_method = getattr(self.llm_client, "stream", None)
        if (
            detect_sensitive_disclosure(question)
            or _LATER_CAPABILITY.search(question)
            or is_phase4_request(question)
            or tool_route is not None
            or needs_book_clarification
            or not callable(stream_method)
        ):
            return _static_stream_plan(self.chat(question, resolved_session))

        history = self.sessions.history(resolved_session)
        retrieval_query = self._retrieval_query(question, history)
        hits = self.knowledge_base.search(retrieval_query)
        provider = self._stream_provider()

        if hits:
            current_input, selected_hits = build_grounded_input(
                question,
                hits,
                max_context_chars=self.config.rag_max_context_chars,
            )
            updates = self._stream_grounded_updates(
                question,
                resolved_session,
                history,
                current_input,
                selected_hits,
                stream_method,
            )
            return ChatStreamPlan(
                session_id=resolved_session,
                mode="rag",
                provider=provider,
                knowledge_used=True,
                updates=updates,
            )

        if self._is_campus_question(question, hits):
            return _static_stream_plan(self.chat(question, resolved_session))

        return ChatStreamPlan(
            session_id=resolved_session,
            mode="chat",
            provider=provider,
            knowledge_used=False,
            updates=self._stream_general_updates(
                question,
                resolved_session,
                history,
                stream_method,
            ),
        )

    def _stream_provider(self) -> str:
        client = self.llm_client
        client_config = getattr(client, "config", None)
        provider = getattr(client, "provider", None) or getattr(
            client_config, "provider", self.config.provider
        )
        model = getattr(client, "model_name", None) or getattr(
            client_config, "model", self.config.model
        )
        return f"{provider}:{model}" if model else str(provider)

    def _stream_grounded_updates(
        self,
        question: str,
        session_id: str,
        history: list[ChatMessage],
        current_input: str,
        selected_hits: list[SearchHit],
        stream_method: Callable[[str, list[ChatMessage]], Iterator[str]],
    ) -> Iterator[ChatStreamUpdate]:
        parts: list[str] = []
        final_answer = ""
        try:
            for chunk in stream_method(
                SYSTEM_PROMPT, llm_messages(history, current_input)
            ):
                if not isinstance(chunk, str) or not chunk:
                    continue
                parts.append(chunk)
                # Native deltas are provisional until the complete draft is
                # checked. A replace event below retracts an invalid draft.
                yield ChatStreamUpdate(kind="delta", text=chunk)
            candidate = "".join(parts).strip()
            if not candidate:
                raise LLMError("聊天模型没有返回文本内容。")
            validation_level, _warnings = _validate_grounded_output(
                candidate, selected_hits
            )
            if validation_level == "hard_fail":
                final_answer = _extractive_answer(selected_hits[:2])
                yield ChatStreamUpdate(kind="replace", text=final_answer)
            else:
                final_answer = candidate
        except Exception:
            final_answer = _extractive_answer(selected_hits[:2])
            yield ChatStreamUpdate(kind="replace", text=final_answer)
        # Never persist provisional tokens or an interrupted draft.
        self.sessions.append_exchange(session_id, question, final_answer)

    def _stream_general_updates(
        self,
        question: str,
        session_id: str,
        history: list[ChatMessage],
        stream_method: Callable[[str, list[ChatMessage]], Iterator[str]],
    ) -> Iterator[ChatStreamUpdate]:
        parts: list[str] = []
        final_answer = ""
        try:
            for chunk in stream_method(
                SYSTEM_PROMPT, llm_messages(history, question)
            ):
                if not isinstance(chunk, str) or not chunk:
                    continue
                parts.append(chunk)
                yield ChatStreamUpdate(kind="delta", text=chunk)
            final_answer = "".join(parts).strip()
            if not final_answer:
                raise LLMError("聊天模型没有返回文本内容。")
        except Exception:
            final_answer = "模型服务暂时不可用。你仍可以询问知识库已覆盖的校园问题。"
            yield ChatStreamUpdate(kind="replace", text=final_answer)
        self.sessions.append_exchange(session_id, question, final_answer)

    def _is_campus_question(self, question: str, hits: list[SearchHit]) -> bool:
        if hits:
            return True
        compact = question.lower().replace(" ", "")
        return any(keyword in compact for keyword in _CAMPUS_KEYWORDS)

    def _retrieval_query(self, question: str, history: list[ChatMessage]) -> str:
        """Resolve short follow-ups against the most recent user topic."""

        if not history or len(question) > 40 or not _FOLLOW_UP.search(question):
            return question
        compact = question.lower().replace(" ", "")
        explicit_topics = _CAMPUS_KEYWORDS - {"学校", "校园"}
        if any(keyword in compact for keyword in explicit_topics):
            return question
        previous_user = next(
            (message.content for message in reversed(history) if message.role == "user"),
            "",
        )
        return f"{previous_user} {question}".strip() if previous_user else question

    def _configuration_warnings(self) -> list[str]:
        if self.config.llm_requested and not self.config.llm_configured:
            return [f"聊天模型未启用：{self.config.provider_status['message']}。已使用本地演示能力。"]
        return []

    def _task_clarification(
        self,
        question: str,
        session_id: str,
        plan: TaskPlan | None,
        clarification: str,
        planner_run: dict[str, Any],
    ) -> ChatResult:
        result = ChatResult(answer=clarification, session_id=session_id, mode="clarification",
                            provider="multi-agent-planner", knowledge_used=False,
                            task_plan=plan.as_dict() if plan is not None else {},
                            task_status="clarification",
                            agent_runs=[planner_run] if planner_run else [])
        self.sessions.append_exchange(session_id, question, result.answer)
        return result

    def _multi_agent(self, question: str, tool_hint: str | None) -> MultiAgentCoordinator:
        # Resolve the official MCP client only after Planner Agent has emitted a
        # visible, validated plan and Executor Agent starts its first tool call.
        return MultiAgentCoordinator(
            question,
            tool_hint,
            lambda name, arguments: self._get_tool_client().call_tool(name, arguments),
        )

    def _multi_agent_result(
        self,
        question: str,
        session_id: str,
        outcome: MultiAgentOutcome,
    ) -> ChatResult:
        execution = outcome.execution
        review = outcome.review
        warnings = list(execution.warnings) if execution is not None else []
        if review is not None and not review.approved:
            warnings.extend(review.issues)
        result = ChatResult(
            answer=outcome.answer,
            session_id=session_id,
            mode="task_error" if outcome.status in {"failed", "review_failed"} else "multi_agent",
            provider="planner+executor+reviewer+mcp",
            knowledge_used=False,
            warnings=_unique_strings(warnings),
            tool_calls=list(execution.tool_calls) if execution is not None else [],
            task_plan=execution.plan.as_dict() if execution is not None else {},
            task_status=outcome.status,
            agent_runs=list(outcome.agent_runs),
            review=review.as_dict() if review is not None else {},
        )
        # Do not seed single-book context with a multi-book workflow's last arguments.
        self.sessions.append_exchange(session_id, question, result.answer)
        return result

    def _execute_tool(
        self,
        question: str,
        session_id: str,
        history: list[ChatMessage],
        route: ToolRoute,
    ) -> ChatResult:
        call_id = f"call-{uuid4().hex[:12]}"
        public_arguments = route.public_arguments()
        try:
            client = self._get_tool_client()
            tool_result = client.call_tool(route.name, route.arguments)
            answer, provider, generation_warnings = self._tool_answer(
                question,
                history,
                route,
                tool_result,
            )
            status = "completed" if tool_result["ok"] else "failed"
            warnings = [*tool_result["warnings"], *generation_warnings]
        except (MCPDependencyError, MCPToolError) as exc:
            tool_result = {
                "ok": False,
                "tool": route.name,
                "data_mode": DATA_MODE_DEMO,
                "items": [],
                "message": str(exc),
                "warnings": ["MCP 工具调用未完成。"],
            }
            answer = f"{route.display_name}没有执行成功：{exc}"
            provider = "mcp-error"
            status = "failed"
            warnings = list(tool_result["warnings"])

        trace = ToolCallTrace(
            call_id=call_id,
            name=route.name,
            display_name=route.display_name,
            arguments=public_arguments,
            status=status,
            data_mode=tool_result["data_mode"],
            item_count=len(tool_result["items"]),
            message=tool_result["message"],
        )
        sources = (
            list(tool_result["items"])
            if route.name == "campus_knowledge_search"
            else []
        )
        result = ChatResult(
            answer=answer,
            session_id=session_id,
            mode="tool" if status == "completed" else "tool_error",
            provider=provider,
            knowledge_used=route.name == "campus_knowledge_search" and bool(sources),
            sources=sources,
            warnings=_unique_strings(warnings),
            tool_calls=[trace.as_dict()],
        )
        self.sessions.append_exchange(
            session_id,
            question,
            result.answer,
            tool_name=route.name if status == "completed" else None,
        )
        return result

    def _get_tool_client(self) -> ToolClient:
        if self.tool_client is None:
            from campus_agent.mcp.server import create_mcp_server

            self.tool_client = MCPToolClient(
                create_mcp_server(self.config, self.knowledge_base)
            )
        return self.tool_client

    def _tool_answer(
        self,
        question: str,
        history: list[ChatMessage],
        route: ToolRoute,
        tool_result: ToolEnvelope,
    ) -> tuple[str, str, list[str]]:
        fallback = _format_tool_result(tool_result)
        if not tool_result["ok"] or not tool_result["items"] or self.llm_client is None:
            return fallback, f"mcp:{route.name}", []
        try:
            prompt = build_tool_result_input(question, route.name, tool_result)
            response = self.llm_client.generate(
                TOOL_RESULT_SYSTEM_PROMPT,
                llm_messages(history, prompt),
            )
        except LLMError as exc:
            return fallback, f"mcp:{route.name}", [f"工具结果自然语言整理失败：{exc}"]
        candidate = response.text.strip()
        if not _tool_answer_is_grounded(candidate, tool_result):
            return (
                fallback,
                f"mcp:{route.name}",
                ["模型整理结果未通过工具数据一致性检查，已使用结构化结果生成回答。"],
            )
        return candidate, f"mcp+{response.provider}:{response.model}", []

    def tooling_status(self) -> dict[str, Any]:
        from campus_agent.mcp.client import TOOL_NAMES, dependency_available

        return {
            "protocol": "MCP",
            "sdk": "official-python-sdk-v2",
            "configured": dependency_available(),
            "transport": ["in-memory", "stdio"],
            "data_mode": "demo",
            "tool_count": len(TOOL_NAMES),
            "tools": list(TOOL_NAMES),
            "message": (
                "MCP 工具系统可用"
                if dependency_available()
                else "尚未安装 MCP SDK；基础对话与 RAG 仍可使用"
            ),
        }

    def _grounded_answer(
        self,
        question: str,
        history: list[ChatMessage],
        hits: list[SearchHit],
    ) -> tuple[str, str, list[SearchHit], list[str]]:
        current_input, selected_hits = build_grounded_input(
            question,
            hits,
            max_context_chars=self.config.rag_max_context_chars,
        )
        if self.llm_client is not None:
            try:
                response = self.llm_client.generate(
                    SYSTEM_PROMPT,
                    llm_messages(history, current_input),
                )
                validation_level, validation_warnings = _validate_grounded_output(
                    response.text, selected_hits
                )
                if validation_level != "hard_fail":
                    return (
                        response.text,
                        f"{response.provider}:{response.model}",
                        selected_hits,
                        validation_warnings,
                    )
                repaired_answer = self._repair_grounded_answer(
                    history,
                    current_input,
                    response.text,
                    selected_hits,
                )
                if repaired_answer:
                    return (
                        repaired_answer,
                        f"{response.provider}:{response.model}",
                        selected_hits,
                        [],
                    )
                return (
                    _extractive_answer(selected_hits[:2]),
                    "local-grounding-fallback",
                    selected_hits[:2],
                    [
                        *validation_warnings,
                        "模型回答没有逐字对应或通过语义一致性检查的可靠依据，已使用有来源的抽取式降级回答。",
                    ],
                )
            except LLMError as exc:
                return (
                    _extractive_answer(selected_hits[:2]),
                    "local-extractive-fallback",
                    selected_hits[:2],
                    [f"聊天模型调用失败：{exc}。已使用有来源的抽取式降级回答。"],
                )
        return (
            _extractive_answer(selected_hits[:2]),
            "local-extractive-fallback",
            selected_hits[:2],
            [],
        )

    def _repair_grounded_answer(
        self,
        history: list[ChatMessage],
        current_input: str,
        draft: str,
        selected_hits: list[SearchHit],
    ) -> str | None:
        """Give the model one chance to repair facts; still enforce the same guard."""

        if self.llm_client is None:
            return None
        repair_input = f"""{current_input}

上一版回答没有通过证据一致性校验。请根据上面的知识库证据重新回答。

待改写草稿：
<draft>
{draft[:3000]}
</draft>

改写要求：
1. 只保留与用户问题直接相关的信息，先给结论，再用 1 至 3 句话说明。
2. 删除证据中没有支持的事实，纠正数字、网址、适用对象以及允许或禁止含义；必要时可以直接引用证据原文。
3. 每项事实保留对应的 [来源N]，不得编造证据中没有的信息。
4. 只输出改写后的最终回答，不解释改写过程。
"""
        try:
            response = self.llm_client.generate(
                SYSTEM_PROMPT,
                llm_messages(history, repair_input),
            )
        except LLMError:
            return None
        repaired = response.text.strip()
        validation_level, _warnings = _validate_grounded_output(repaired, selected_hits)
        return repaired if validation_level != "hard_fail" else None

    def _general_answer(
        self, question: str, history: list[ChatMessage]
    ) -> tuple[str, str, list[str]]:
        if self.llm_client is not None:
            try:
                response = self.llm_client.generate(
                    SYSTEM_PROMPT,
                    llm_messages(history, question),
                )
                return response.text, f"{response.provider}:{response.model}", []
            except LLMError as exc:
                return (
                    "模型服务暂时不可用。你仍可以询问知识库已覆盖的校园问题。",
                    "local-fallback",
                    [str(exc)],
                )
        if _GREETING.fullmatch(question):
            return (
                "你好，我是华工校园助手的阶段 1–5 Demo。你可以询问校园知识、查询模拟二手书／匿名课表／课程评价，也可以让 Planner、Executor、Reviewer 三个 Agent 协作完成课程推荐和多本二手书比价。",
                "local-demo",
                [],
            )
        if _THANKS.fullmatch(question):
            return "不客气。如果是校园事实问题，我会尽量给出可核验的知识库来源。", "local-demo", []
        if _IDENTITY.search(question):
            return (
                "我是一个以对话为入口的校园助手原型。目前已完成基础对话、本地 RAG 知识问答，"
                "MCP 工具调用、有边界的任务规划，以及由 Planner、Executor、Reviewer 组成且审核失败即停止输出的第五阶段多 Agent 协作。",
                "local-demo",
                [],
            )
        return (
            "当前处于离线演示模式，未配置真实大模型，因此开放域闲聊能力有限。"
            "配置 LLM 后可以正常对话；校园事实即使接入模型，也只会依据知识库回答。",
            "local-demo",
            [],
        )


def _static_stream_plan(result: ChatResult) -> ChatStreamPlan:
    def updates() -> Iterator[ChatStreamUpdate]:
        for run in result.agent_runs:
            start = dict(run)
            start["status"] = "running"
            start["message"] = f"{run.get('display_name', 'Agent')} 正在处理。"
            start.pop("approved", None)
            yield ChatStreamUpdate(kind="agent_start", payload=start)
            yield ChatStreamUpdate(kind="agent_result", payload=dict(run))
        for call in result.tool_calls:
            yield ChatStreamUpdate(
                kind="tool_start",
                payload={
                    "call_id": call.get("call_id"),
                    "name": call.get("name"),
                    "display_name": call.get("display_name"),
                    "arguments": call.get("arguments", {}),
                },
            )
            yield ChatStreamUpdate(
                kind="tool_result",
                payload={
                    "call_id": call.get("call_id"),
                    "name": call.get("name"),
                    "display_name": call.get("display_name"),
                    "status": call.get("status"),
                    "data_mode": call.get("data_mode"),
                    "item_count": call.get("item_count", 0),
                    "message": call.get("message", ""),
                },
            )
        yield ChatStreamUpdate(kind="delta", text=result.answer)

    return ChatStreamPlan(
        session_id=result.session_id,
        mode=result.mode,
        provider=result.provider,
        knowledge_used=result.knowledge_used,
        updates=updates(),
    )


def _format_tool_result(result: ToolEnvelope) -> str:
    prefix = "以下为模拟数据。" if result["data_mode"] == DATA_MODE_DEMO else ""
    if not result["ok"] or not result["items"]:
        return f"{prefix}{result['message']}".strip()

    lines = [prefix, result["message"]] if prefix else [result["message"]]
    for item in result["items"][:5]:
        if result["tool"] == "secondhand_book_search":
            lines.append(
                f"- {item.get('book')} {item.get('edition') or ''}：{item.get('price'):g} 元，"
                f"{item.get('condition')}；{item.get('campus')}，{item.get('pickup')}；"
                f"发布者 {item.get('seller_alias')}。"
            )
        elif result["tool"] == "student_schedule_query":
            lines.append(
                f"- {item.get('weekday')} {item.get('start_time')}–{item.get('end_time')}："
                f"{item.get('course_name')}，{item.get('location')}，{item.get('weeks')}。"
            )
        elif result["tool"] == "course_review_search":
            lines.append(
                f"- {item.get('course_name')}：{item.get('rating'):g}/5，"
                f"{item.get('review_count')} 条模拟评价；{item.get('attendance_policy')}；"
                f"{item.get('summary')}"
            )
        elif result["tool"] == "campus_knowledge_search":
            lines.append(
                f"- {item.get('title')}（{item.get('section') or '相关章节'}）："
                f"{item.get('excerpt')}"
            )
        elif result["tool"] == "course_offering_search":
            times = "；".join(f"{meeting.get('weekday')} {meeting.get('start_time')}–{meeting.get('end_time')}（{meeting.get('week_start')}–{meeting.get('week_end')}周）" for meeting in item.get("meetings", [])) or "时间待公布，不能判断无冲突"
            lines.append(f"- {item.get('course_name')}（{item.get('offering_id')}）：{times}；{item.get('campus')}，{item.get('location')}。")
    return "\n".join(line for line in lines if line)


def _tool_answer_is_grounded(answer: str, result: ToolEnvelope) -> bool:
    if not answer:
        return False
    if result["data_mode"] == DATA_MODE_DEMO and "模拟" not in answer:
        return False
    encoded = json.dumps(result, ensure_ascii=False)
    answer_numbers = set(re.findall(r"\d+(?:\.\d+)?", answer))
    allowed_numbers = set(re.findall(r"\d+(?:\.\d+)?", encoded))
    if not answer_numbers.issubset(allowed_numbers):
        return False
    identifiers = {
        str(item.get(key)).strip()
        for item in result["items"]
        for key in ("book", "course_name", "title")
        if item.get(key)
    }
    return not identifiers or any(identifier in answer for identifier in identifiers)


def _unique_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _clean_markdown_for_extract(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = re.sub(r"^#{1,6}\s+", "", line).strip()
        if stripped.startswith(">"):
            stripped = stripped[1:].strip()
        if stripped:
            lines.append(stripped)
    return "\n".join(lines)


def _extractive_answer(hits: list[SearchHit]) -> str:
    if not hits:
        return "当前知识库没有可用依据。"
    paragraphs = ["根据当前知识库，可确认的信息如下："]
    for ordinal, hit in enumerate(hits[:2], start=1):
        text = _clean_markdown_for_extract(hit.chunk.text)
        if len(text) > 900:
            text = text[:899].rstrip() + "…"
        paragraphs.append(f"{text} [来源{ordinal}]")
    volatile = any(hit.chunk.volatility in {"high", "very_high"} for hit in hits[:2])
    if volatile:
        paragraphs.append("这类信息时效性较高，请在办理前结合来源文件和学校最新正式通知再次核验。")
    return "\n\n".join(paragraphs)


def _source_reliability_warnings(hits: list[SearchHit]) -> list[str]:
    selected = hits
    if any(
        hit.chunk.authority != "official"
        or hit.chunk.verification_status in {"pending-review", "unverified"}
        for hit in selected
    ):
        return ["本回答使用了用户提供且待核验的整理资料，不应视为校方正式规定；请以学校最新正式通知为准。"]
    if any(hit.chunk.verification_status == "source-provided" for hit in selected):
        return ["依据来自用户提供的本地原始文档，尚未通过官网 URL 独立核验。"]
    return []


def _normalise_for_evidence(text: str) -> str:
    text = text.lower().replace("：", ":")
    text = re.sub(r"[–—−]", "-", text)
    return re.sub(r"\s+", "", text)


def _normalise_paraphrase_terms(text: str) -> str:
    """Fold a few high-risk policy paraphrases before lexical comparison."""

    value = normalize(text)
    value = re.sub(r"(?:不允许|不得|严禁|不可|不能)", "禁止", value)
    value = re.sub(r"(?:可以|可供|有权)", "允许", value)
    value = value.replace("占用座位", "占座").replace("占位", "占座")
    return value


def _substantive_terms(text: str) -> set[str]:
    counts = token_counts(_normalise_paraphrase_terms(text))
    return {
        term
        for term in counts
        if term not in _TERM_NOISE
        and (_CJK_TERM.fullmatch(term) is not None or _ASCII_TERM.fullmatch(term) is not None)
    }


def _permission_polarity(text: str) -> int:
    """Return -1 for a prohibition, 1 for permission, and 0 when unspecified/mixed."""

    has_negative = _NEGATIVE_PERMISSION.search(text) is not None
    has_affirmative = _AFFIRMATIVE_PERMISSION.search(text) is not None
    if has_negative == has_affirmative:
        return 0
    return -1 if has_negative else 1


def _best_evidence_passages(claim: str, evidence: str) -> list[str]:
    passages = [part.strip() for part in re.split(r"[\n。！？；]+", evidence) if part.strip()]
    if not passages:
        return [evidence]
    claim_terms = _substantive_terms(claim)
    ranked = sorted(
        passages,
        key=lambda passage: len(claim_terms & _substantive_terms(passage)),
        reverse=True,
    )
    best_score = len(claim_terms & _substantive_terms(ranked[0]))
    return [
        passage
        for passage in ranked
        if len(claim_terms & _substantive_terms(passage)) == best_score
    ][:3]


def _has_substantive_evidence_overlap(claim: str, evidence: str) -> bool:
    claim_terms = _substantive_terms(claim)
    if not claim_terms:
        return False
    evidence_terms = _substantive_terms(evidence)
    shared_terms = claim_terms & evidence_terms
    minimum_shared = 1 if len(claim_terms) <= 4 else 2
    if len(shared_terms) < minimum_shared:
        return False
    if len(claim_terms) > 6 and len(shared_terms) / len(claim_terms) < 0.35:
        return False
    return True


def _unsupported_specific_literals(claim: str, evidence: str) -> list[str]:
    evidence_compact = _normalise_for_evidence(evidence)
    return sorted(
        {
            literal
            for literal in _SPECIFIC_LITERAL.findall(claim)
            if _normalise_for_evidence(literal) not in evidence_compact
        }
    )


def _has_consistent_permission_polarity(claim: str, evidence: str) -> bool:
    claim_polarity = _permission_polarity(claim)
    if not claim_polarity:
        return True
    passage_polarities = {
        polarity
        for passage in _best_evidence_passages(claim, evidence)
        if (polarity := _permission_polarity(passage))
    }
    return not passage_polarities or claim_polarity in passage_polarities


def _has_excessive_verbatim_overlap(claim: str, evidence: str) -> bool:
    """Reject long copied passages while allowing exact names, numbers and URLs."""

    claim_compact = re.sub(
        r"[^\u3400-\u9fffa-z0-9]+", "", claim.lower(), flags=re.IGNORECASE
    )
    evidence_compact = re.sub(
        r"[^\u3400-\u9fffa-z0-9]+", "", evidence.lower(), flags=re.IGNORECASE
    )
    if len(claim_compact) < 40 or not evidence_compact:
        return False
    longest_match = SequenceMatcher(
        None,
        claim_compact,
        evidence_compact,
        autojunk=False,
    ).find_longest_match()
    return longest_match.size >= 32 and longest_match.size / len(claim_compact) >= 0.45

def _hit_evidence(hit: SearchHit) -> str:
    chunk = hit.chunk
    return " ".join(
        part
        for part in (
            chunk.title,
            chunk.section,
            chunk.text,
            chunk.source_url,
            chunk.published_at,
            chunk.verified_at,
            chunk.updated_at,
        )
        if part
    )


def _validate_grounded_output(
    answer: str, hits: list[SearchHit]
) -> tuple[ValidationLevel, list[str]]:
    """Separate hard factual conflicts from soft citation and style warnings."""

    if not hits:
        return "hard_fail", ["没有可用于校验模型回答的证据。"]
    claim = _CITATION.sub("", answer).strip()
    if len(_normalise_for_evidence(claim)) < 4:
        return "hard_fail", ["模型回答为空或只有引用。"]

    warnings: list[str] = []
    references = [int(value) for value in _CITATION.findall(answer)]
    if not references:
        return "hard_fail", ["模型回答没有使用标准的 [来源N] 引用格式。"]
    invalid = sorted({number for number in references if not 1 <= number <= len(hits)})
    valid_references = sorted(
        {number for number in references if 1 <= number <= len(hits)}
    )
    if invalid:
        return "hard_fail", [f"模型回答中含有无效引用编号：{invalid}。"]

    referenced_evidence = " ".join(
        _hit_evidence(hits[number - 1])
        for number in (valid_references or range(1, len(hits) + 1))
    )
    unsupported_literals = _unsupported_specific_literals(claim, referenced_evidence)
    if unsupported_literals:
        return (
            "hard_fail",
            [f"模型回答包含证据中不存在的数字、网址或标识：{unsupported_literals}。"],
        )
    if not _has_consistent_permission_polarity(claim, referenced_evidence):
        return "hard_fail", ["模型回答的允许或禁止含义与引用证据相反。"]
    if not _has_substantive_evidence_overlap(claim, referenced_evidence):
        return "hard_fail", ["模型回答与证据没有足够的实质性对应。"]
    # Validate facts independently: a supported sentence must not mask an
    # unrelated second claim, or borrow numbers from a different cited source.
    for sentence in re.findall(
        r"[^。！？；\n]+[。！？；]?(?:\s*\[来源\d+\])*", answer
    ):
        sentence_claim = _CITATION.sub("", sentence).strip()
        if len(_normalise_for_evidence(sentence_claim)) < 4:
            continue
        sentence_references = sorted(
            {int(value) for value in _CITATION.findall(sentence)}
        )
        if not sentence_references:
            return "hard_fail", ["模型回答中的事实句没有对应的来源引用。"]
        sentence_evidence = " ".join(
            _hit_evidence(hits[number - 1]) for number in sentence_references
        )
        if (
            _unsupported_specific_literals(sentence_claim, sentence_evidence)
            or not _has_consistent_permission_polarity(sentence_claim, sentence_evidence)
            or not _has_substantive_evidence_overlap(sentence_claim, sentence_evidence)
        ):
            return "hard_fail", ["模型回答中的事实句未通过其对应来源的证据校验。"]
    if _has_excessive_verbatim_overlap(claim, referenced_evidence):
        warnings.append("模型回答与知识库原文重复较多。")
    return ("soft_fail" if warnings else "pass"), warnings
