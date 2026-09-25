from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str


@dataclass(slots=True)
class KnowledgeDocument:
    source_id: str
    relative_path: str
    title: str
    text: str
    content_hash: str
    source_url: str = ""
    updated_at: str = ""
    published_at: str = ""
    verified_at: str = ""
    source_owner: str = ""
    authority: str = "unknown"
    verification_status: str = "unverified"
    volatility: str = "unknown"
    audience: str = ""
    campus_scope: str = ""
    auth_required: bool = False
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class KnowledgeChunk:
    chunk_id: str
    source_id: str
    relative_path: str
    title: str
    section: str
    text: str
    content_hash: str
    source_url: str = ""
    updated_at: str = ""
    published_at: str = ""
    verified_at: str = ""
    source_owner: str = ""
    authority: str = "unknown"
    verification_status: str = "unverified"
    volatility: str = "unknown"
    audience: str = ""
    campus_scope: str = ""
    auth_required: bool = False
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SearchHit:
    chunk: KnowledgeChunk
    score: float
    bm25_score: float
    coverage: float
    matched_terms: list[str] = field(default_factory=list)

    def as_source(self, ordinal: int, excerpt_chars: int = 220) -> dict[str, Any]:
        excerpt = " ".join(self.chunk.text.split())
        if len(excerpt) > excerpt_chars:
            excerpt = excerpt[: excerpt_chars - 1].rstrip() + "…"
        return {
            "id": f"来源{ordinal}",
            "chunk_id": self.chunk.chunk_id,
            "title": self.chunk.title,
            "section": self.chunk.section,
            "path": self.chunk.relative_path,
            "source_url": self.chunk.source_url or None,
            "updated_at": self.chunk.updated_at or None,
            "published_at": self.chunk.published_at or None,
            "verified_at": self.chunk.verified_at or None,
            "source_owner": self.chunk.source_owner or None,
            "authority": self.chunk.authority,
            "verification_status": self.chunk.verification_status,
            "volatility": self.chunk.volatility,
            "audience": self.chunk.audience or None,
            "campus_scope": self.chunk.campus_scope or None,
            "auth_required": self.chunk.auth_required,
            "score": round(self.score, 4),
            "bm25_score": round(self.bm25_score, 4),
            "keyword_coverage": round(self.coverage, 4),
            "matched_terms": list(self.matched_terms),
            "excerpt": excerpt,
        }


@dataclass(slots=True)
class IngestionIssue:
    path: str
    message: str
    severity: str = "warning"


@dataclass(slots=True)
class IngestionReport:
    documents: int
    chunks: int
    fingerprint: str
    built_at: str
    index_file: str
    issues: list[IngestionIssue] = field(default_factory=list)
    skipped_rebuild: bool = False

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["warnings"] = [
            issue.message for issue in self.issues if issue.severity == "warning"
        ]
        return payload


@dataclass(slots=True)
class ToolCallTrace:
    call_id: str
    name: str
    display_name: str
    arguments: dict[str, Any]
    status: str
    data_mode: str
    item_count: int
    message: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ChatResult:
    answer: str
    session_id: str
    mode: str
    provider: str
    knowledge_used: bool
    sources: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    task_plan: dict[str, Any] = field(default_factory=dict)
    task_status: str = ""
    agent_runs: list[dict[str, Any]] = field(default_factory=list)
    review: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ChatStreamUpdate:
    kind: str
    text: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChatStreamPlan:
    session_id: str
    mode: str
    provider: str
    knowledge_used: bool
    updates: Iterator[ChatStreamUpdate]
