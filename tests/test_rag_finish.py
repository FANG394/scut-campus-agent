from __future__ import annotations

import unittest

from campus_agent.agent import ChatService, _validate_grounded_output
from campus_agent.llm import LLMError, LLMResult
from campus_agent.models import KnowledgeChunk, SearchHit
from tests.helpers import PROJECT_ROOT, make_config


def evidence(text: str, ordinal: int = 1) -> SearchHit:
    return SearchHit(
        chunk=KnowledgeChunk(
            chunk_id=f"test-{ordinal}",
            source_id=f"source-{ordinal}",
            relative_path=f"test-{ordinal}.md",
            title="公共自习室使用规定",
            section="使用要求",
            text=text,
            content_hash=f"hash-{ordinal}",
            authority="official",
            verification_status="verified",
        ),
        score=0.95,
        bm25_score=1.0,
        coverage=1.0,
    )


class StaticKnowledge:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits

    def ensure_ready(self) -> None:
        pass

    def search(self, _query: str) -> list[SearchHit]:
        return self.hits


class FailingLLM:
    def generate(self, *_args: object) -> LLMResult:
        raise LLMError("test model unavailable")


class InterruptingStream:
    provider = "fake"
    model_name = "test"

    def __init__(self, chunks: list[str], fail: bool = False) -> None:
        self.chunks = chunks
        self.fail = fail

    def stream(self, *_args: object):
        yield from self.chunks
        if self.fail:
            raise LLMError("test stream interrupted")

    def generate(self, *_args: object) -> LLMResult:
        raise AssertionError("native RAG stream must not call generate")


class RagFinishTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hits = [evidence("公共自习室学生不得以任何方式占座。公共自习室禁止校外人员使用。")]
        self.config = make_config(PROJECT_ROOT)

    def service(self, llm: object | None = None) -> ChatService:
        return ChatService(
            self.config,
            knowledge_base=StaticKnowledge(self.hits),
            llm_client=llm,
        )

    def test_no_model_uses_cited_extractive_evidence(self) -> None:
        result = self.service().chat("公共自习室允许占座吗？")
        self.assertEqual(result.provider, "local-extractive-fallback")
        self.assertIn("不得以任何方式占座", result.answer)
        self.assertIn("[来源1]", result.answer)
        self.assertEqual(len(result.sources), 1)

    def test_model_failure_preserves_cited_knowledge_answer(self) -> None:
        result = self.service(FailingLLM()).chat("公共自习室允许占座吗？")
        self.assertIn("[来源1]", result.answer)
        self.assertIn("不得以任何方式占座", result.answer)
        self.assertTrue(any("test model unavailable" in warning for warning in result.warnings))

    def test_interrupted_stream_replaces_draft_and_only_saves_final(self) -> None:
        service = self.service(InterruptingStream(["公共自习室允许占座"], fail=True))
        plan = service.stream_chat("公共自习室允许占座吗？")
        iterator = iter(plan.updates)
        first = next(iterator)
        self.assertEqual((first.kind, first.text), ("delta", "公共自习室允许占座"))
        self.assertEqual(service.sessions.history(plan.session_id), [])
        remaining = list(iterator)
        self.assertEqual(remaining[-1].kind, "replace")
        self.assertIn("不得以任何方式占座", remaining[-1].text)
        self.assertEqual(service.sessions.history(plan.session_id)[-1].content, remaining[-1].text)

    def test_empty_stream_replaces_with_cited_evidence(self) -> None:
        service = self.service(InterruptingStream([]))
        plan = service.stream_chat("公共自习室允许占座吗？")
        updates = list(plan.updates)
        self.assertEqual([update.kind for update in updates], ["replace"])
        self.assertIn("[来源1]", updates[0].text)

    def test_closed_stream_does_not_save_provisional_draft(self) -> None:
        service = self.service(InterruptingStream(["公共自习室学生不得", "占座。[来源1]"]))
        plan = service.stream_chat("公共自习室允许占座吗？")
        next(plan.updates)
        plan.updates.close()
        self.assertEqual(service.sessions.history(plan.session_id), [])

    def test_missing_or_mixed_invalid_citations_are_hard_failures(self) -> None:
        for answer in (
            "公共自习室不允许学生占用座位。",
            "公共自习室不允许学生占用座位。[来源1][来源99]",
        ):
            with self.subTest(answer=answer):
                self.assertEqual(_validate_grounded_output(answer, self.hits)[0], "hard_fail")

    def test_supported_paraphrase_still_passes(self) -> None:
        answer = "公共自习室不允许学生占用座位。[来源1]"
        self.assertEqual(_validate_grounded_output(answer, self.hits)[0], "pass")

    def test_permission_contradiction_is_rejected(self) -> None:
        answer = "公共自习室允许学生占用座位。[来源1]"
        self.assertEqual(_validate_grounded_output(answer, self.hits)[0], "hard_fail")

    def test_unrelated_sentence_cannot_hide_behind_supported_sentence(self) -> None:
        answer = "公共自习室不允许学生占用座位。[来源1]火星上已经建立城市。[来源1]"
        self.assertEqual(_validate_grounded_output(answer, self.hits)[0], "hard_fail")

    def test_each_claim_must_use_its_own_cited_numbers(self) -> None:
        hits = [
            evidence("公共自习室开放时间为08:00。"),
            evidence("公共自习室关闭时间为19:00。", 2),
        ]
        answer = "公共自习室开放时间为19:00。[来源1]公共自习室关闭时间为19:00。[来源2]"
        self.assertEqual(_validate_grounded_output(answer, hits)[0], "hard_fail")


if __name__ == "__main__":
    unittest.main()
