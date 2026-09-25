from __future__ import annotations

import hashlib
import re

from campus_agent.models import KnowledgeChunk, KnowledgeDocument


CHUNKER_VERSION = "heading-paragraph-v4"
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;])")


def _sections(text: str, fallback_title: str) -> list[tuple[str, str]]:
    heading_stack: list[str] = []
    current_lines: list[str] = []
    current_section = fallback_title
    output: list[tuple[str, str]] = []

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if body:
            output.append((current_section, body))
        current_lines.clear()

    for line in text.splitlines():
        match = _HEADING.match(line)
        if not match:
            current_lines.append(line)
            continue
        flush()
        level = len(match.group(1))
        heading = match.group(2).strip()
        heading_stack = heading_stack[: level - 1]
        while len(heading_stack) < level - 1:
            heading_stack.append("")
        heading_stack.append(heading)
        current_section = " > ".join(part for part in heading_stack if part)
    flush()
    return output or [(fallback_title, text.strip())]


def _split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]
    if len(sentences) <= 1:
        return [text[start : start + max_chars] for start in range(0, len(text), max_chars)]
    output: list[str] = []
    buffer = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if buffer.strip():
                output.append(buffer.strip())
                buffer = ""
            output.extend(
                sentence[start : start + max_chars]
                for start in range(0, len(sentence), max_chars)
            )
            continue
        if buffer and len(buffer) + len(sentence) > max_chars:
            output.append(buffer.strip())
            buffer = sentence
        else:
            buffer += sentence
    if buffer.strip():
        output.append(buffer.strip())
    return output


def _atoms(section_text: str, max_chars: int) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", section_text) if part.strip()]
    output: list[str] = []
    for paragraph in paragraphs:
        output.extend(_split_long(paragraph, max_chars))
    return output


def _pack(atoms: list[str], target_chars: int, overlap_chars: int) -> list[str]:
    chunks: list[str] = []
    buffer = ""
    for atom in atoms:
        separator = "\n\n" if buffer else ""
        if buffer and len(buffer) + len(separator) + len(atom) > target_chars:
            chunks.append(buffer.strip())
            available_overlap = max(0, target_chars - len(atom) - 2)
            actual_overlap = min(overlap_chars, available_overlap)
            overlap = buffer[-actual_overlap:].lstrip() if actual_overlap else ""
            buffer = f"{overlap}\n\n{atom}".strip() if overlap else atom
        else:
            buffer = f"{buffer}{separator}{atom}"
    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks


def chunk_documents(
    documents: list[KnowledgeDocument],
    *,
    target_chars: int = 700,
    overlap_chars: int = 100,
) -> list[KnowledgeChunk]:
    if overlap_chars >= target_chars:
        overlap_chars = max(0, target_chars // 5)
    chunks: list[KnowledgeChunk] = []
    for document in documents:
        ordinal = 0
        for section, section_text in _sections(document.text, document.title):
            for text in _pack(_atoms(section_text, target_chars), target_chars, overlap_chars):
                if not text.strip():
                    continue
                raw_id = f"{document.source_id}|{section}|{ordinal}|{text}".encode("utf-8")
                chunk_id = hashlib.sha256(raw_id).hexdigest()[:24]
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=chunk_id,
                        source_id=document.source_id,
                        relative_path=document.relative_path,
                        title=document.title,
                        section=section,
                        text=text,
                        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        source_url=document.source_url,
                        updated_at=document.updated_at,
                        published_at=document.published_at,
                        verified_at=document.verified_at,
                        source_owner=document.source_owner,
                        authority=document.authority,
                        verification_status=document.verification_status,
                        volatility=document.volatility,
                        audience=document.audience,
                        campus_scope=document.campus_scope,
                        auth_required=document.auth_required,
                        tags=list(document.tags),
                    )
                )
                ordinal += 1
    return chunks
