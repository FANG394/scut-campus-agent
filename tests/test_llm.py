from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from campus_agent.config import AppConfig
from campus_agent.llm import ChatOllamaClient, LLMError, RemoteLLMClient
from campus_agent.models import ChatMessage
from tests.helpers import make_config


class FakeResponse:
    def __init__(self, payload: object | bytes) -> None:
        self.body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")

    def read(self, size: int = -1) -> bytes:
        return self.body if size < 0 else self.body[:size]

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class FakeOpener:
    def __init__(self, payload: object | bytes) -> None:
        self.payload = payload
        self.request = None
        self.timeout = None

    def open(self, request: object, timeout: float) -> FakeResponse:
        self.request = request
        self.timeout = timeout
        return FakeResponse(self.payload)


class RemoteLLMClientTests(unittest.TestCase):
    def config(self, *, api_style: str = "responses"):
        root = Path(self.temporary.name)
        return make_config(
            root,
            provider="openai" if api_style == "responses" else "compatible",
            api_style=api_style,
            api_key="test-key",
            model="test-model",
            base_url="https://api.example.test/v1",
        )

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_responses_request_and_text_parsing(self) -> None:
        opener = FakeOpener({"status": "completed", "output_text": "你好"})
        client = RemoteLLMClient(self.config())

        with patch("urllib.request.build_opener", return_value=opener):
            result = client.generate(
                "system",
                [ChatMessage(role="user", content="我的密码是 SECRET_12345")],
            )

        self.assertEqual(result.text, "你好")
        self.assertEqual(result.provider, "openai")
        self.assertEqual(opener.request.full_url, "https://api.example.test/v1/responses")
        self.assertEqual(opener.request.get_header("Authorization"), "Bearer test-key")
        payload = json.loads(opener.request.data.decode("utf-8"))
        self.assertFalse(payload["store"])
        self.assertEqual(payload["max_output_tokens"], 300)
        self.assertNotIn("SECRET_12345", json.dumps(payload))
        self.assertIn("已脱敏", payload["input"][0]["content"])

    def test_default_configuration_selects_local_qwen_models(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = AppConfig.from_env(Path(self.temporary.name))

        self.assertEqual(config.provider, "ollama")
        self.assertEqual(config.model, "qwen3:4b")
        self.assertEqual(config.embedding_model, "qwen3-embedding:4b")
        self.assertEqual(config.ollama_base_url, "http://127.0.0.1:11434")

    def test_chat_completions_uses_configured_token_field(self) -> None:
        config = replace(self.config(api_style="chat_completions"), llm_chat_token_field="max_tokens")
        opener = FakeOpener(
            {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": "兼容回答"}}
                ]
            }
        )

        with patch("urllib.request.build_opener", return_value=opener):
            result = RemoteLLMClient(config).generate(
                "system", [ChatMessage(role="user", content="hello")]
            )

        self.assertEqual(result.text, "兼容回答")
        payload = json.loads(opener.request.data.decode("utf-8"))
        self.assertEqual(payload["max_tokens"], 300)
        self.assertNotIn("max_completion_tokens", payload)
        self.assertEqual(payload["messages"][0]["role"], "system")

    def test_incomplete_responses_output_is_rejected(self) -> None:
        opener = FakeOpener(
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output_text": "截断内容",
            }
        )
        with patch("urllib.request.build_opener", return_value=opener):
            with self.assertRaisesRegex(LLMError, "未完整完成"):
                RemoteLLMClient(self.config()).generate("system", [])

    def test_length_limited_chat_completion_is_rejected(self) -> None:
        opener = FakeOpener(
            {
                "choices": [
                    {"finish_reason": "length", "message": {"content": "截断内容"}}
                ]
            }
        )
        with patch("urllib.request.build_opener", return_value=opener):
            with self.assertRaisesRegex(LLMError, "输出长度上限"):
                RemoteLLMClient(self.config(api_style="chat_completions")).generate("system", [])

    def test_oversized_response_is_rejected(self) -> None:
        opener = FakeOpener(b"x" * 2_000_001)
        with patch("urllib.request.build_opener", return_value=opener):
            with self.assertRaisesRegex(LLMError, "返回内容过大"):
                RemoteLLMClient(self.config()).generate("system", [])

    def test_external_plain_http_base_url_is_not_configured(self) -> None:
        external = replace(self.config(), base_url="http://api.example.test/v1")
        loopback = replace(self.config(), base_url="http://127.0.0.1:11434/v1")

        self.assertFalse(external.remote_llm_configured)
        self.assertTrue(loopback.remote_llm_configured)
        with self.assertRaises(ValueError):
            RemoteLLMClient(external)

    def test_chat_ollama_uses_requested_local_models_and_hides_reasoning(self) -> None:
        created: dict[str, object] = {}

        class FakeChatOllama:
            def __init__(self, **kwargs: object) -> None:
                created.update(kwargs)

            def invoke(self, messages: list[tuple[str, str]]) -> object:
                created["messages"] = messages
                return types.SimpleNamespace(content="本地回答")

        module = types.ModuleType("langchain_ollama")
        module.ChatOllama = FakeChatOllama
        config = make_config(
            Path(self.temporary.name),
            provider="ollama",
            model="qwen3:4b",
            base_url="http://127.0.0.1:11434",
        )

        with patch.dict(sys.modules, {"langchain_ollama": module}):
            result = ChatOllamaClient(config).generate(
                "system", [ChatMessage(role="user", content="你好")]
            )

        self.assertEqual(result.text, "本地回答")
        self.assertEqual(result.provider, "ollama")
        self.assertEqual(created["model"], "qwen3:4b")
        self.assertEqual(created["base_url"], "http://127.0.0.1:11434")
        self.assertIs(created["reasoning"], True)

    def test_chat_ollama_stream_yields_content_without_reasoning_metadata(self) -> None:
        class FakeModel:
            def stream(self, _messages: object):
                yield types.SimpleNamespace(
                    content="", additional_kwargs={"reasoning_content": "内部推理"}
                )
                yield types.SimpleNamespace(content="最终")
                yield types.SimpleNamespace(content="回答")

        config = make_config(
            Path(self.temporary.name),
            provider="ollama",
            model="qwen3:4b",
            base_url="http://127.0.0.1:11434",
        )

        chunks = list(
            ChatOllamaClient(config, chat_model=FakeModel()).stream(
                "system", [ChatMessage(role="user", content="你好")]
            )
        )

        self.assertEqual(chunks, ["最终", "回答"])
        self.assertNotIn("内部推理", "".join(chunks))


if __name__ == "__main__":
    unittest.main()
