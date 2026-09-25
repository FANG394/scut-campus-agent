from __future__ import annotations

import hashlib
import threading
from collections.abc import Sequence
from typing import Protocol

from campus_agent.config import AppConfig
from campus_agent.models import IngestionReport, KnowledgeChunk, SearchHit
from campus_agent.rag.chunker import chunk_documents
from campus_agent.rag.index import (
    VECTOR_INDEX_VERSION,
    IndexData,
    InvalidIndexError,
    build_index_data,
    load_index,
    save_index,
)
from campus_agent.rag.loaders import corpus_fingerprint, load_documents
from campus_agent.rag.retriever import HybridRetriever


class EmbeddingsClient(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class KnowledgeBase:
    def __init__(
        self,
        config: AppConfig,
        embeddings: EmbeddingsClient | None = None,
    ) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._index: IndexData | None = None
        self._retriever: HybridRetriever | None = None
        self._embeddings = embeddings

    @property
    def embedding_model(self) -> str:
        return str(getattr(self.config, "embedding_model", "qwen3-embedding:4b")).strip()

    @property
    def embedding_batch_size(self) -> int:
        return max(1, int(getattr(self.config, "embedding_batch_size", 16)))

    def _get_embeddings(self) -> EmbeddingsClient:
        if self._embeddings is not None:
            return self._embeddings
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "向量检索需要 langchain-ollama；请先安装项目模型依赖。"
            ) from exc
        base_url = str(
            getattr(self.config, "ollama_base_url", "http://127.0.0.1:11434")
        ).strip()
        options: dict[str, object] = {"model": self.embedding_model}
        if base_url:
            options["base_url"] = base_url
        options["client_kwargs"] = {"timeout": self.config.llm_timeout_seconds}
        self._embeddings = OllamaEmbeddings(**options)
        return self._embeddings

    def ensure_ready(self, *, force: bool = False) -> IngestionReport:
        with self._lock:
            current_fingerprint = self._configured_fingerprint(
                corpus_fingerprint(self.config.knowledge_source_dir)
            )
            if not force and self.config.knowledge_index_file.is_file():
                try:
                    existing = load_index(self.config.knowledge_index_file)
                    if (
                        existing.fingerprint == current_fingerprint
                        and existing.embedding_model == self.embedding_model
                    ):
                        self._activate(existing)
                        return self._report(existing, skipped=True)
                except InvalidIndexError:
                    pass
            return self.rebuild()

    def rebuild(self) -> IngestionReport:
        with self._lock:
            self.config.knowledge_source_dir.mkdir(parents=True, exist_ok=True)
            documents, issues, fingerprint = load_documents(self.config.knowledge_source_dir)
            fingerprint = self._configured_fingerprint(fingerprint)
            chunks = chunk_documents(
                documents,
                target_chars=self.config.rag_chunk_size,
                overlap_chars=self.config.rag_chunk_overlap,
            )
            vectors = self._embed_chunks(chunks)
            index = build_index_data(
                chunks,
                vectors,
                fingerprint=fingerprint,
                embedding_model=self.embedding_model,
                issues=issues,
            )
            # save_index uses a same-directory temporary file and os.replace;
            # a failed embedding/build therefore never overwrites a good index.
            save_index(index, self.config.knowledge_index_file)
            self._activate(index)
            return self._report(index, skipped=False)

    def _embed_chunks(self, chunks: Sequence[KnowledgeChunk]) -> list[list[float]]:
        if not chunks:
            return []
        embeddings = self._get_embeddings()
        texts = [
            f"标题：{chunk.title}\n章节：{chunk.section}\n正文：{chunk.text}"
            for chunk in chunks
        ]
        vectors: list[list[float]] = []
        batch_size = self.embedding_batch_size
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            embedded = embeddings.embed_documents(batch)
            if len(embedded) != len(batch):
                raise RuntimeError(
                    "嵌入模型返回的向量数量不匹配："
                    f"请求 {len(batch)}，返回 {len(embedded)}。"
                )
            vectors.extend(embedded)
        return vectors

    def _configured_fingerprint(self, corpus_hash: str) -> str:
        signature = (
            f"{corpus_hash}|chunk_size={self.config.rag_chunk_size}"
            f"|chunk_overlap={self.config.rag_chunk_overlap}"
            f"|embedding_model={self.embedding_model}"
            f"|vector_index={VECTOR_INDEX_VERSION}"
        )
        return hashlib.sha256(signature.encode("utf-8")).hexdigest()

    def _activate(self, index: IndexData) -> None:
        if index.embedding_model != self.embedding_model:
            raise InvalidIndexError("索引嵌入模型与当前配置不匹配，需要重建。")
        self._index = index
        self._retriever = HybridRetriever(index, self._get_embeddings())

    def _report(self, index: IndexData, *, skipped: bool) -> IngestionReport:
        return IngestionReport(
            documents=index.document_count,
            chunks=len(index.chunks),
            fingerprint=index.fingerprint,
            built_at=index.built_at,
            index_file=str(self.config.knowledge_index_file),
            issues=list(index.issues),
            skipped_rebuild=skipped,
        )

    def search(self, query: str, *, top_k: int | None = None) -> list[SearchHit]:
        with self._lock:
            if self._retriever is None:
                self.ensure_ready()
            assert self._retriever is not None
            return self._retriever.search(
                query,
                top_k=top_k or self.config.rag_top_k,
                min_score=self.config.rag_min_score,
            )

    def status(self) -> dict[str, object]:
        with self._lock:
            if self._index is None:
                self.ensure_ready()
            assert self._index is not None
            return {
                "ready": True,
                "documents": self._index.document_count,
                "chunks": len(self._index.chunks),
                "built_at": self._index.built_at,
                "source_dir": str(self.config.knowledge_source_dir),
                "index_file": str(self.config.knowledge_index_file),
                "retriever": "hybrid-vector-bm25",
                "embedding_model": self._index.embedding_model,
                "embedding_dimension": self._index.embedding_dimension,
                "warnings": [issue.message for issue in self._index.issues],
            }

    @property
    def index(self) -> IndexData | None:
        return self._index
