from __future__ import annotations

import base64
import binascii
import json
import math
import os
import sys
import tempfile
from array import array
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from campus_agent.models import IngestionIssue, KnowledgeChunk
from campus_agent.rag.chunker import CHUNKER_VERSION
from campus_agent.rag.loaders import PARSER_VERSION


INDEX_SCHEMA_VERSION = 3
VECTOR_INDEX_VERSION = "cosine-float32-v1"
VECTOR_FORMAT = "float32-le-base64"
_UNIT_VECTOR_TOLERANCE = 1e-3


class InvalidIndexError(RuntimeError):
    pass


@dataclass(slots=True)
class IndexedChunk:
    chunk: KnowledgeChunk
    vector: array


@dataclass(slots=True)
class IndexData:
    fingerprint: str
    built_at: str
    embedding_model: str
    embedding_dimension: int
    chunks: list[IndexedChunk]
    issues: list[IngestionIssue]

    @property
    def document_count(self) -> int:
        return len({item.chunk.source_id for item in self.chunks})


def normalise_vector(
    values: Iterable[float],
    *,
    expected_dimension: int | None = None,
) -> array:
    """Return a finite L2-normalised float32 vector."""

    try:
        numeric = [float(value) for value in values]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("向量包含无法解析的数值。") from exc
    if expected_dimension is not None and len(numeric) != expected_dimension:
        raise ValueError(
            f"向量维度不匹配：期望 {expected_dimension}，实际 {len(numeric)}。"
        )
    if not numeric:
        raise ValueError("向量不能为空。")
    if any(not math.isfinite(value) for value in numeric):
        raise ValueError("向量包含非有限数值。")
    norm = math.hypot(*numeric)
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("向量范数无效。")
    try:
        output = array("f", (value / norm for value in numeric))
    except (OverflowError, ValueError) as exc:
        raise ValueError("向量无法转换为 float32。") from exc
    if output.itemsize != 4:  # pragma: no cover - CPython uses IEEE-754 float32
        raise RuntimeError("当前 Python 平台不支持 4 字节 float32 向量。")
    if any(not math.isfinite(value) for value in output):
        raise ValueError("float32 向量包含非有限数值。")
    return output


def build_index_data(
    chunks: list[KnowledgeChunk],
    vectors: list[Iterable[float]],
    *,
    fingerprint: str,
    embedding_model: str,
    issues: list[IngestionIssue] | None = None,
) -> IndexData:
    if len(chunks) != len(vectors):
        raise ValueError(
            f"知识块与向量数量不一致：{len(chunks)} 个知识块，{len(vectors)} 个向量。"
        )
    model = embedding_model.strip()
    if not model:
        raise ValueError("嵌入模型名称不能为空。")

    indexed: list[IndexedChunk] = []
    dimension = 0
    seen_chunk_ids: set[str] = set()
    for chunk, raw_vector in zip(chunks, vectors, strict=True):
        if chunk.chunk_id in seen_chunk_ids:
            raise ValueError(f"知识块 ID 重复：{chunk.chunk_id}")
        vector = normalise_vector(raw_vector, expected_dimension=dimension or None)
        if not dimension:
            dimension = len(vector)
        indexed.append(IndexedChunk(chunk=chunk, vector=vector))
        seen_chunk_ids.add(chunk.chunk_id)

    return IndexData(
        fingerprint=fingerprint,
        built_at=datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        embedding_model=model,
        embedding_dimension=dimension,
        chunks=indexed,
        issues=list(issues or []),
    )


def _encode_vector(vector: array) -> str:
    if vector.typecode != "f" or vector.itemsize != 4:
        raise ValueError("索引向量必须使用 float32 存储。")
    little_endian = array("f", vector)
    if sys.byteorder != "little":  # pragma: no cover - supported Windows is little endian
        little_endian.byteswap()
    return base64.b64encode(little_endian.tobytes()).decode("ascii")


def _decode_vector(encoded: object, dimension: int) -> array:
    if not isinstance(encoded, str):
        raise ValueError("向量编码必须是字符串。")
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
        raise ValueError("向量不是有效的 Base64 数据。") from exc
    if len(raw) != dimension * 4:
        raise ValueError(
            f"向量字节长度不匹配：期望 {dimension * 4}，实际 {len(raw)}。"
        )
    vector = array("f")
    vector.frombytes(raw)
    if sys.byteorder != "little":  # pragma: no cover - supported Windows is little endian
        vector.byteswap()
    if len(vector) != dimension:
        raise ValueError("向量维度不匹配。")
    if any(not math.isfinite(value) for value in vector):
        raise ValueError("向量包含非有限数值。")
    norm = math.hypot(*vector)
    if not math.isfinite(norm) or abs(norm - 1.0) > _UNIT_VECTOR_TOLERANCE:
        raise ValueError("索引向量未正确归一化。")
    return vector


def _payload(index: IndexData) -> dict[str, Any]:
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "parser_version": PARSER_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "vector_index_version": VECTOR_INDEX_VERSION,
        "vector_format": VECTOR_FORMAT,
        "fingerprint": index.fingerprint,
        "built_at": index.built_at,
        "embedding_model": index.embedding_model,
        "embedding_dimension": index.embedding_dimension,
        "issues": [asdict(issue) for issue in index.issues],
        "chunks": [
            {
                "chunk": asdict(item.chunk),
                "vector": _encode_vector(item.vector),
            }
            for item in index.chunks
        ],
    }


def save_index(index: IndexData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(_payload(index), ensure_ascii=False, separators=(",", ":"))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def load_index(path: Path) -> IndexData:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidIndexError(f"索引无法读取：{exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise InvalidIndexError("索引版本不匹配，需要重建。")
    if payload.get("parser_version") != PARSER_VERSION:
        raise InvalidIndexError("文档解析器已更新，需要重建索引。")
    if payload.get("chunker_version") != CHUNKER_VERSION:
        raise InvalidIndexError("切分器已更新，需要重建索引。")
    if payload.get("vector_index_version") != VECTOR_INDEX_VERSION:
        raise InvalidIndexError("向量检索器已更新，需要重建索引。")
    if payload.get("vector_format") != VECTOR_FORMAT:
        raise InvalidIndexError("向量存储格式不匹配，需要重建索引。")

    try:
        fingerprint = payload["fingerprint"]
        built_at = payload["built_at"]
        embedding_model = payload["embedding_model"]
        embedding_dimension = payload["embedding_dimension"]
        raw_chunks = payload["chunks"]
        raw_issues = payload["issues"]
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("fingerprint 无效")
        if not isinstance(built_at, str) or not built_at:
            raise ValueError("built_at 无效")
        if not isinstance(embedding_model, str) or not embedding_model.strip():
            raise ValueError("embedding_model 无效")
        if (
            not isinstance(embedding_dimension, int)
            or isinstance(embedding_dimension, bool)
            or embedding_dimension < 0
        ):
            raise ValueError("embedding_dimension 无效")
        if not isinstance(raw_chunks, list) or not isinstance(raw_issues, list):
            raise ValueError("chunks 或 issues 不是列表")
        if raw_chunks and embedding_dimension <= 0:
            raise ValueError("非空索引的 embedding_dimension 必须大于 0")
        if not raw_chunks and embedding_dimension != 0:
            raise ValueError("空索引的 embedding_dimension 必须为 0")

        chunks: list[IndexedChunk] = []
        seen_chunk_ids: set[str] = set()
        for item in raw_chunks:
            if not isinstance(item, dict):
                raise ValueError("chunks 中存在非对象条目")
            chunk_payload = item["chunk"]
            if not isinstance(chunk_payload, dict):
                raise ValueError("chunk 元数据不是对象")
            chunk = KnowledgeChunk(**chunk_payload)
            if not chunk.chunk_id or chunk.chunk_id in seen_chunk_ids:
                raise ValueError(f"chunk_id 缺失或重复：{chunk.chunk_id}")
            chunks.append(
                IndexedChunk(
                    chunk=chunk,
                    vector=_decode_vector(item["vector"], embedding_dimension),
                )
            )
            seen_chunk_ids.add(chunk.chunk_id)
        issues: list[IngestionIssue] = []
        for item in raw_issues:
            if not isinstance(item, dict):
                raise ValueError("issues 中存在非对象条目")
            issues.append(IngestionIssue(**item))
        return IndexData(
            fingerprint=fingerprint,
            built_at=built_at,
            embedding_model=embedding_model.strip(),
            embedding_dimension=embedding_dimension,
            chunks=chunks,
            issues=issues,
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise InvalidIndexError(f"索引结构无效：{exc}") from exc
