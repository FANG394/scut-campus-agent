from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from campus_agent.models import KnowledgeDocument
from campus_agent.rag.chunker import chunk_documents
from campus_agent.rag.index import InvalidIndexError, load_index
from campus_agent.rag.loaders import load_documents
from campus_agent.rag.service import KnowledgeBase
from tests.helpers import (
    FakeEmbeddings,
    make_config,
    write_markdown,
    write_minimal_docx,
    write_sidecar,
)


class LoaderTests(unittest.TestCase):
    def test_loads_crlf_front_matter_gb18030_and_docx(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            markdown = root / "guide.md"
            markdown.write_bytes(
                (
                    "---\r\n"
                    "title: 校园网指南\r\n"
                    "authority: official\r\n"
                    "auth_required: false\r\n"
                    "tags: 网络, 新生\r\n"
                    "---\r\n\r\n"
                    "# 校园网指南\r\n\r\n新生先激活统一认证账号。\r\n"
                ).encode("utf-8")
            )
            legacy = root / "legacy.txt"
            legacy.write_bytes("旧版办事说明采用中文编码。".encode("gb18030"))
            write_sidecar(legacy, {"title": "旧版说明", "auth_required": "false"})
            write_minimal_docx(root / "notice.docx", ["图书馆通知", "开放安排见官网。"])
            (root / "notice.docx.ocr.txt").write_text(
                "## 图片转录\n\n图片中的补充开放安排。", encoding="utf-8"
            )

            documents, issues, fingerprint = load_documents(root)

            self.assertEqual(issues, [])
            self.assertEqual(len(documents), 3)
            self.assertEqual(len(fingerprint), 64)
            by_title = {document.title: document for document in documents}
            self.assertIn("校园网指南", by_title)
            self.assertFalse(by_title["校园网指南"].auth_required)
            self.assertEqual(by_title["校园网指南"].tags, ["网络", "新生"])
            self.assertFalse(by_title["旧版说明"].auth_required)
            self.assertIn("开放安排见官网", next(doc.text for doc in documents if doc.relative_path.endswith(".docx")))
            self.assertIn("图片中的补充开放安排", next(doc.text for doc in documents if doc.relative_path.endswith(".docx")))

    def test_bad_document_becomes_issue_without_stopping_ingestion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_markdown(root / "valid.md", title="有效资料", body="这是有效资料正文。")
            (root / "broken.docx").write_bytes(b"not-a-zip")

            documents, issues, _fingerprint = load_documents(root)

            self.assertEqual(len(documents), 1)
            self.assertEqual(len(issues), 1)
            self.assertEqual(issues[0].path, "broken.docx")

    def test_pdf_text_cache_is_used_as_part_of_one_document(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "student-handbook.pdf"
            cache = root / "student-handbook.pdf.txt"
            pdf.write_bytes(b"placeholder-pdf-bytes")
            cache.write_text("## 第 1 页\n\n学生手册缓存正文。", encoding="utf-8")
            write_sidecar(pdf, {"title": "学生手册", "authority": "official"})

            documents, issues, first_fingerprint = load_documents(root)

            self.assertEqual(issues, [])
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].title, "学生手册")
            self.assertIn("学生手册缓存正文", documents[0].text)
            self.assertEqual(documents[0].relative_path, "student-handbook.pdf")

            cache.write_text("## 第 1 页\n\n缓存正文已经更新。", encoding="utf-8")
            _documents, _issues, second_fingerprint = load_documents(root)
            self.assertNotEqual(first_fingerprint, second_fingerprint)


class ChunkingAndIndexTests(unittest.TestCase):
    def test_oversize_sentence_is_split_within_limit(self) -> None:
        document = KnowledgeDocument(
            source_id="source",
            relative_path="long.md",
            title="长文档",
            text="# 说明\n\n" + "甲" * 451,
            content_hash="hash",
        )

        chunks = chunk_documents([document], target_chars=100, overlap_chars=20)

        self.assertGreater(len(chunks), 4)
        self.assertTrue(all(0 < len(chunk.text) <= 100 for chunk in chunks))

    def test_index_reuses_unchanged_corpus_and_rebuilds_for_chunk_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            write_markdown(source / "network.md", title="校园网络", body="新生需要先激活统一认证账号。")
            config = make_config(root, source_dir=source)
            knowledge = KnowledgeBase(config, embeddings=FakeEmbeddings())

            first = knowledge.ensure_ready()
            second = KnowledgeBase(config, embeddings=FakeEmbeddings()).ensure_ready()
            changed = replace(config, rag_chunk_size=500)
            third = KnowledgeBase(changed, embeddings=FakeEmbeddings()).ensure_ready()

            self.assertFalse(first.skipped_rebuild)
            self.assertTrue(second.skipped_rebuild)
            self.assertFalse(third.skipped_rebuild)
            self.assertNotEqual(first.fingerprint, third.fingerprint)
            self.assertEqual(load_index(config.knowledge_index_file).fingerprint, third.fingerprint)

            changed_model = replace(changed, embedding_model="fake-embedding-v2")
            fourth = KnowledgeBase(
                changed_model, embeddings=FakeEmbeddings()
            ).ensure_ready()
            self.assertFalse(fourth.skipped_rebuild)
            self.assertNotEqual(third.fingerprint, fourth.fingerprint)
            self.assertEqual(
                load_index(config.knowledge_index_file).embedding_model,
                "fake-embedding-v2",
            )

    def test_invalid_index_is_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            write_markdown(source / "library.md", title="图书馆", body="开放时间以官网为准。")
            config = make_config(root, source_dir=source)
            config.knowledge_index_file.parent.mkdir(parents=True)
            config.knowledge_index_file.write_text("{}", encoding="utf-8")

            with self.assertRaises(InvalidIndexError):
                load_index(config.knowledge_index_file)

            report = KnowledgeBase(config, embeddings=FakeEmbeddings()).ensure_ready()
            self.assertFalse(report.skipped_rebuild)
            self.assertGreater(report.chunks, 0)

    def test_corrupt_vector_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            write_markdown(source / "network.md", title="校园网络", body="新生开通校园网。")
            config = make_config(root, source_dir=source)
            KnowledgeBase(config, embeddings=FakeEmbeddings()).ensure_ready()
            payload = json.loads(config.knowledge_index_file.read_text(encoding="utf-8"))
            payload["chunks"][0]["vector"] = "AAAA"
            config.knowledge_index_file.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )

            with self.assertRaisesRegex(InvalidIndexError, "向量字节长度"):
                load_index(config.knowledge_index_file)

    def test_failed_embedding_does_not_overwrite_existing_index(self) -> None:
        class FailingEmbeddings(FakeEmbeddings):
            def embed_documents(self, _texts: list[str]) -> list[list[float]]:
                raise RuntimeError("embedding failed")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            write_markdown(source / "network.md", title="校园网络", body="新生开通校园网。")
            config = make_config(root, source_dir=source)
            KnowledgeBase(config, embeddings=FakeEmbeddings()).ensure_ready()
            original = config.knowledge_index_file.read_bytes()

            with self.assertRaisesRegex(RuntimeError, "embedding failed"):
                KnowledgeBase(config, embeddings=FailingEmbeddings()).rebuild()

            self.assertEqual(config.knowledge_index_file.read_bytes(), original)

    def test_ollama_embeddings_uses_configured_model_url_and_timeout(self) -> None:
        created: dict[str, object] = {}

        class FakeOllamaEmbeddings:
            def __init__(self, **kwargs: object) -> None:
                created.update(kwargs)

        module = types.ModuleType("langchain_ollama")
        module.OllamaEmbeddings = FakeOllamaEmbeddings
        with tempfile.TemporaryDirectory() as temporary:
            config = make_config(Path(temporary))
            with patch.dict(sys.modules, {"langchain_ollama": module}):
                KnowledgeBase(config)._get_embeddings()

        self.assertEqual(created["model"], "fake-embedding-v1")
        self.assertEqual(created["base_url"], "http://127.0.0.1:11434")
        self.assertEqual(created["client_kwargs"], {"timeout": 2.0})


class RetrievalTests(unittest.TestCase):
    def test_known_question_matches_and_unrelated_question_abstains(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            write_markdown(
                source / "network.md",
                title="校园公共无线网",
                body="新生需要先激活统一认证账号，再连接校园公共无线网 scut。",
            )
            write_markdown(
                source / "library.md",
                title="图书馆开放时间",
                body="大学城校区图书馆开放时间为每天 8:00 至 22:00。",
            )
            knowledge = KnowledgeBase(
                make_config(root, source_dir=source), embeddings=FakeEmbeddings()
            )
            knowledge.ensure_ready()

            known = knowledge.search("新生怎么申请校园网络？")
            unknown = knowledge.search("哪里可以买到便宜的篮球鞋？")

            self.assertTrue(known)
            self.assertEqual(known[0].chunk.title, "校园公共无线网")
            self.assertEqual(unknown, [])

    def test_keyword_fallback_recalls_exact_policy_below_vector_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            write_markdown(
                source / "student-card.md",
                title="学生证管理办法",
                body=(
                    "学生遗失学生证或损坏学生证，应及时向学院办公室提交书面报告，"
                    "由学院教务员核实后集中办理。"
                ),
            )
            write_markdown(
                source / "library.md",
                title="图书馆开放时间",
                body="大学城校区图书馆每天 8:00 开放。",
            )
            knowledge = KnowledgeBase(
                make_config(root, source_dir=source, rag_min_score=0.99),
                embeddings=FakeEmbeddings(),
            )
            knowledge.ensure_ready()

            known = knowledge.search("学生证损坏怎么办")
            unknown = knowledge.search("哪里可以买到便宜的篮球鞋？")
            partial_concept_match = knowledge.search("图书馆校外访问怎么使用？")

            self.assertTrue(known)
            self.assertEqual(known[0].chunk.title, "学生证管理办法")
            self.assertGreater(known[0].bm25_score, 0.0)
            self.assertGreater(known[0].coverage, 0.0)
            self.assertIn("学生证", known[0].matched_terms)
            self.assertEqual(unknown, [])
            self.assertEqual(partial_concept_match, [])


if __name__ == "__main__":
    unittest.main()
