from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import unicodedata
import zipfile
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from campus_agent.models import IngestionIssue, KnowledgeDocument


SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".html", ".htm", ".docx", ".pdf"}
PARSER_VERSION = "loaders-v5"
PDF_TEXT_CACHE_SUFFIX = ".pdf.txt"
DOCX_OCR_CACHE_SUFFIX = ".docx.ocr.txt"
DERIVED_TEXT_CACHE_SUFFIXES = (PDF_TEXT_CACHE_SUFFIX, DOCX_OCR_CACHE_SUFFIX)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "yes", "是"}:
        return True
    if lowered in {"false", "no", "否"}:
        return False
    return value


def _parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}, text
    metadata: dict[str, Any] = {}
    for raw_line in text[4:end].splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#") or ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        key = key.strip().lower().replace("-", "_")
        if key:
            metadata[key] = _parse_scalar(value)
    return metadata, text[end + 5 :].lstrip()


def _load_sidecar(path: Path) -> dict[str, Any]:
    candidates = [path.with_suffix(path.suffix + ".meta.json"), path.with_suffix(".meta.json")]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return {str(key).lower().replace("-", "_"): value for key, value in payload.items()}
    return {}


class _VisibleHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden_depth = 0
        self.heading_level: int | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg"}:
            self.hidden_depth += 1
        if self.hidden_depth:
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.heading_level = int(tag[1])
            self.parts.append("\n" + "#" * self.heading_level + " ")
        elif tag in {"p", "div", "section", "article", "br", "li", "tr"}:
            self.parts.append("\n")
        if tag == "li":
            self.parts.append("- ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg"} and self.hidden_depth:
            self.hidden_depth -= 1
            return
        if not self.hidden_depth and tag in {"p", "div", "section", "article", "li", "tr"}:
            self.parts.append("\n")
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.heading_level = None

    def handle_data(self, data: str) -> None:
        if not self.hidden_depth:
            self.parts.append(data)


def _load_html(path: Path) -> str:
    parser = _VisibleHTMLParser()
    parser.feed(_read_text(path))
    return "".join(parser.parts)


def _load_docx(path: Path) -> str:
    """Extract DOCX paragraphs with only the Python standard library."""

    with zipfile.ZipFile(path) as archive:
        xml_data = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml_data)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.iterfind(".//w:p", ns):
        text = "".join(node.text or "" for node in paragraph.iterfind(".//w:t", ns)).strip()
        if not text:
            continue
        style_node = paragraph.find("./w:pPr/w:pStyle", ns)
        style = ""
        if style_node is not None:
            style = style_node.attrib.get(f"{{{ns['w']}}}val", "")
        match = re.search(r"(?:heading|标题)\s*([1-6])", style, flags=re.IGNORECASE)
        if match:
            text = f"{'#' * int(match.group(1))} {text}"
        paragraphs.append(text)
    text = "\n\n".join(paragraphs)
    ocr_cache = path.with_suffix(path.suffix + ".ocr.txt")
    if ocr_cache.is_file():
        supplemental = _read_text(ocr_cache).strip()
        if supplemental:
            text = f"{text}\n\n{supplemental}" if text else supplemental
    return text


def _load_pdf(path: Path) -> str:
    text_cache = path.with_suffix(path.suffix + ".txt")
    if text_cache.is_file():
        cached_text = _read_text(text_cache).strip()
        if not cached_text:
            raise RuntimeError(f"PDF 文本缓存为空：{text_cache.name}")
        return cached_text

    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("读取 PDF 需要可选依赖 pypdf；请执行 `python -m pip install pypdf`。") from exc
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:  # pragma: no cover - pypdf exception classes vary
            raise RuntimeError("PDF 已加密，无法读取。") from exc
    pages: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if page_text:
            pages.append(f"## 第 {index} 页\n\n{page_text}")
    if not pages:
        raise RuntimeError("PDF 未提取到文本，可能是扫描件；阶段二暂不自动执行 OCR。")
    return "\n\n".join(pages)


def _extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        match = re.match(r"^#{1,3}\s+(.+?)\s*$", line)
        if match:
            return match.group(1).strip()
    return fallback


def _normalise_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[,，;；]", value) if part.strip()]
    return []


def _normalise_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "是"}
    return False


def _load_one(path: Path, source_dir: Path) -> KnowledgeDocument:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        raw_text = _read_text(path)
    elif suffix in {".html", ".htm"}:
        raw_text = _load_html(path)
    elif suffix == ".docx":
        raw_text = _load_docx(path)
    elif suffix == ".pdf":
        raw_text = _load_pdf(path)
    else:  # defensive; discovery already filters extensions
        raise RuntimeError(f"不支持的文件类型：{suffix}")

    front_matter, body = _parse_front_matter(raw_text)
    metadata = {**_load_sidecar(path), **front_matter}
    text = _clean_text(body)
    if not text:
        raise RuntimeError("文档没有可索引文本。")

    relative_path = path.relative_to(source_dir).as_posix()
    content_hash = _sha256_bytes(text.encode("utf-8"))
    title = str(metadata.get("title") or _extract_title(text, path.stem)).strip()
    stat = path.stat()
    fallback_updated = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).date().isoformat()
    authority = str(metadata.get("authority") or "unknown").strip().lower()
    verification_status = str(
        metadata.get("verification_status")
        or ("source-provided" if authority == "official" else "unverified")
    ).strip()
    source_id = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:20]
    return KnowledgeDocument(
        source_id=source_id,
        relative_path=relative_path,
        title=title,
        text=text,
        content_hash=content_hash,
        source_url=str(metadata.get("source_url") or metadata.get("canonical_url") or "").strip(),
        updated_at=str(metadata.get("updated_at") or fallback_updated).strip(),
        published_at=str(metadata.get("published_at") or "").strip(),
        verified_at=str(metadata.get("verified_at") or "").strip(),
        source_owner=str(metadata.get("source_owner") or "").strip(),
        authority=authority,
        verification_status=verification_status,
        volatility=str(metadata.get("volatility") or "unknown").strip().lower(),
        audience=str(metadata.get("audience") or "").strip(),
        campus_scope=str(metadata.get("campus_scope") or "").strip(),
        auth_required=_normalise_bool(metadata.get("auth_required", False)),
        tags=_normalise_tags(metadata.get("tags", [])),
    )


def discover_files(source_dir: Path) -> list[Path]:
    if not source_dir.is_dir():
        return []
    paths: list[Path] = []
    for path in source_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if path.name.lower().endswith(DERIVED_TEXT_CACHE_SUFFIXES):
            continue
        relative_parts = path.relative_to(source_dir).parts
        if any(part.startswith(".") for part in relative_parts):
            continue
        paths.append(path)
    return sorted(paths, key=lambda item: item.relative_to(source_dir).as_posix().lower())


def corpus_fingerprint(source_dir: Path) -> str:
    digest = hashlib.sha256()
    source_files = discover_files(source_dir)
    has_uncached_pdf = any(
        path.suffix.lower() == ".pdf"
        and not path.with_suffix(path.suffix + ".txt").is_file()
        for path in source_files
    )
    capability = (
        "pypdf=1" if has_uncached_pdf and importlib.util.find_spec("pypdf") else "pypdf=0"
    )
    digest.update(f"{PARSER_VERSION}|{capability}\n".encode("utf-8"))
    for path in source_files:
        relative = path.relative_to(source_dir).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        derived_caches: tuple[Path, ...] = ()
        if path.suffix.lower() == ".pdf":
            derived_caches = (path.with_suffix(path.suffix + ".txt"),)
        elif path.suffix.lower() == ".docx":
            derived_caches = (path.with_suffix(path.suffix + ".ocr.txt"),)
        for text_cache in derived_caches:
            if text_cache.is_file():
                digest.update(text_cache.name.encode("utf-8"))
                digest.update(text_cache.read_bytes())
        for sidecar in (path.with_suffix(path.suffix + ".meta.json"), path.with_suffix(".meta.json")):
            if sidecar.is_file():
                digest.update(sidecar.name.encode("utf-8"))
                digest.update(sidecar.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_documents(source_dir: Path) -> tuple[list[KnowledgeDocument], list[IngestionIssue], str]:
    documents: list[KnowledgeDocument] = []
    issues: list[IngestionIssue] = []
    fingerprint = corpus_fingerprint(source_dir)
    for path in discover_files(source_dir):
        try:
            documents.append(_load_one(path, source_dir))
        except Exception as exc:
            issues.append(
                IngestionIssue(
                    path=path.relative_to(source_dir).as_posix(),
                    message=f"{path.name}：{exc}",
                )
            )
    return documents, issues, fingerprint
