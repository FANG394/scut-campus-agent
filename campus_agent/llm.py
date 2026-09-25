from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from collections.abc import Iterator
from typing import Any

from campus_agent.config import AppConfig
from campus_agent.models import ChatMessage
from campus_agent.safety import redact_sensitive_text


class LLMError(RuntimeError):
    pass


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    """Do not forward bearer credentials through HTTP redirects."""

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


@dataclass(slots=True)
class LLMResult:
    text: str
    provider: str
    model: str


class RemoteLLMClient:
    """Small HTTP adapter for Responses and OpenAI-compatible Chat Completions."""

    def __init__(self, config: AppConfig) -> None:
        if not config.remote_llm_configured:
            raise ValueError(str(config.provider_status["message"]))
        self.config = config

    @property
    def endpoint(self) -> str:
        path = "/responses" if self.config.api_style == "responses" else "/chat/completions"
        return self.config.base_url.rstrip("/") + path

    def _payload(self, instructions: str, messages: list[ChatMessage]) -> dict[str, Any]:
        serialised_messages = [
            {"role": message.role, "content": redact_sensitive_text(message.content)}
            for message in messages
        ]
        if self.config.api_style == "responses":
            return {
                "model": self.config.model,
                "instructions": instructions,
                "input": serialised_messages,
                "max_output_tokens": self.config.llm_max_output_tokens,
                "store": False,
            }
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "system", "content": instructions}, *serialised_messages],
            "temperature": 0.2,
        }
        payload[self.config.llm_chat_token_field] = self.config.llm_max_output_tokens
        return payload

    def generate(self, instructions: str, messages: list[ChatMessage]) -> LLMResult:
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(self._payload(instructions, messages), ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "campus-agent-demo/0.2",
            },
            method="POST",
        )
        try:
            opener = urllib.request.build_opener(_RejectRedirects())
            with opener.open(request, timeout=self.config.llm_timeout_seconds) as response:
                raw = response.read(2_000_001)
                if len(raw) > 2_000_000:
                    raise LLMError("模型服务返回内容过大。")
        except urllib.error.HTTPError as exc:
            detail = _http_error_detail(exc)
            raise LLMError(f"模型服务返回 HTTP {exc.code}：{detail}") from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise LLMError(f"无法连接模型服务：{reason}") from exc
        except TimeoutError as exc:
            raise LLMError("模型服务调用超时。") from exc

        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMError("模型服务返回了无法解析的 JSON。") from exc
        if self.config.api_style == "responses":
            _validate_responses_status(payload)
            text = _responses_text(payload)
        else:
            _validate_chat_completion_status(payload)
            text = _chat_completions_text(payload)
        if not text.strip():
            raise LLMError("模型服务没有返回文本内容。")
        return LLMResult(text=text.strip(), provider=self.config.provider, model=self.config.model)


class ChatOllamaClient:
    """LangChain adapter for the locally deployed Ollama chat model."""

    def __init__(self, config: AppConfig, *, chat_model: Any | None = None) -> None:
        if not config.ollama_configured:
            raise ValueError(str(config.provider_status["message"]))
        self.config = config
        if chat_model is None:
            try:
                from langchain_ollama import ChatOllama
            except ImportError as exc:
                raise LLMError(
                    "本地聊天模型需要 langchain-ollama；请先安装 requirements.txt。"
                ) from exc
            chat_model = ChatOllama(
                model=config.model,
                base_url=config.ollama_base_url,
                temperature=0,
                num_predict=config.llm_max_output_tokens,
                reasoning=True,
                client_kwargs={"timeout": config.llm_timeout_seconds},
            )
        self.model = chat_model

    @staticmethod
    def _messages(
        instructions: str, messages: list[ChatMessage]
    ) -> list[tuple[str, str]]:
        return [
            ("system", instructions),
            *[
                (message.role, redact_sensitive_text(message.content))
                for message in messages
                if message.role in {"user", "assistant"}
            ],
        ]

    def generate(self, instructions: str, messages: list[ChatMessage]) -> LLMResult:
        try:
            response = self.model.invoke(self._messages(instructions, messages))
        except Exception as exc:
            raise LLMError(f"无法调用本地 Ollama 聊天模型：{str(exc)[:500]}") from exc
        text = _langchain_content_text(getattr(response, "content", response)).strip()
        if not text:
            raise LLMError("本地 Ollama 聊天模型没有返回文本内容。")
        return LLMResult(text=text, provider="ollama", model=self.config.model)

    def stream(self, instructions: str, messages: list[ChatMessage]) -> Iterator[str]:
        """Yield native ChatOllama chunks for callers that can consume live tokens."""

        try:
            for response in self.model.stream(self._messages(instructions, messages)):
                text = _langchain_content_text(getattr(response, "content", response))
                if text:
                    yield text
        except Exception as exc:
            raise LLMError(f"无法流式调用本地 Ollama 聊天模型：{str(exc)[:500]}") from exc


def _langchain_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if isinstance(value, str):
                    parts.append(value)
        return "".join(parts)
    return "" if content is None else str(content)


def _http_error_detail(error: urllib.error.HTTPError) -> str:
    try:
        raw = error.read(4096).decode("utf-8", errors="replace")
        payload = json.loads(raw)
        if isinstance(payload, dict):
            nested = payload.get("error")
            if isinstance(nested, dict) and nested.get("message"):
                return str(nested["message"])[:500]
            if payload.get("message"):
                return str(payload["message"])[:500]
        return raw[:500] or error.reason
    except Exception:
        return str(error.reason)


def _responses_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    parts: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                parts.append(content["text"])
    return "\n".join(parts)


def _validate_responses_status(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise LLMError("模型服务返回结构无效。")
    status = payload.get("status")
    if status in {None, "completed"}:
        return
    details = payload.get("incomplete_details")
    reason = details.get("reason") if isinstance(details, dict) else None
    suffix = f"（{reason}）" if reason else ""
    raise LLMError(f"模型响应未完整完成：{status}{suffix}。")


def _chat_completions_text(payload: Any) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) for item in content if isinstance(item, dict)
        )
    return ""


def _validate_chat_completion_status(payload: Any) -> None:
    try:
        finish_reason = payload["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise LLMError("模型服务返回结构无效。") from exc
    if finish_reason in {None, "stop"}:
        return
    if finish_reason == "length":
        raise LLMError("模型回答因达到输出长度上限而被截断。")
    raise LLMError(f"模型回答未正常结束：{finish_reason}。")


def create_remote_client(config: AppConfig) -> RemoteLLMClient | None:
    if not config.remote_llm_configured:
        return None
    return RemoteLLMClient(config)


def create_llm_client(
    config: AppConfig,
) -> ChatOllamaClient | RemoteLLMClient | None:
    if config.ollama_configured:
        return ChatOllamaClient(config)
    return create_remote_client(config)
