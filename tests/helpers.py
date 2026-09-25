from __future__ import annotations

import json
import hashlib
import math
import re
import zipfile
from pathlib import Path

from campus_agent.config import AppConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_config(
    root: Path,
    *,
    source_dir: Path | None = None,
    index_file: Path | None = None,
    provider: str = "local",
    api_style: str = "chat_completions",
    api_key: str = "",
    model: str = "",
    base_url: str = "",
    rag_min_score: float = 0.60,
) -> AppConfig:
    return AppConfig(
        project_root=root,
        host="127.0.0.1",
        port=0,
        provider=provider,
        api_style=api_style,
        api_key=api_key,
        model=model,
        base_url=base_url,
        ollama_base_url="http://127.0.0.1:11434",
        embedding_model="fake-embedding-v1",
        embedding_batch_size=8,
        llm_timeout_seconds=2.0,
        llm_max_output_tokens=300,
        llm_chat_token_field="max_completion_tokens",
        max_history_messages=12,
        knowledge_source_dir=source_dir or root / "knowledge",
        knowledge_index_file=index_file or root / "index" / "index.json",
        tool_data_dir=PROJECT_ROOT / "data" / "tools",
        rag_top_k=4,
        rag_min_score=rag_min_score,
        rag_max_context_chars=6000,
        rag_chunk_size=700,
        rag_chunk_overlap=100,
    )


class FakeEmbeddings:
    """Small deterministic semantic embedder used only by offline tests."""

    _CONCEPTS = (
        ("校园网", "无线网", "网络", "scut", "上网", "开网"),
        ("新生", "入学"),
        ("统一认证", "账号", "认证"),
        ("图书馆",),
        ("开放时间", "开放", "时间"),
        ("校外访问", "校外", "访问", "vpn"),
        ("公共自习室", "自习室", "自习教室"),
        ("占座", "占位", "座位"),
        ("校外人员", "外来人员"),
        ("体测", "体质测试", "体育测试"),
        ("校园卡", "学生证", "证件"),
        ("毕业", "毕业要求"),
        ("学生手册", "本科生手册"),
        ("考试", "补考", "缓考"),
        ("处分", "违纪"),
        ("宿舍", "住宿"),
        ("收费", "费用", "资费"),
        ("联系电话", "电话", "联系方式"),
    )
    _FALLBACK_DIMENSIONS = 257

    def __init__(self) -> None:
        self.document_calls = 0
        self.query_calls = 0

    @classmethod
    def _embed(cls, text: str) -> list[float]:
        compact = re.sub(r"\s+", "", text.lower())
        vector = [
            1.0 if any(keyword in compact for keyword in aliases) else 0.0
            for aliases in cls._CONCEPTS
        ]
        fallback = [0.0] * cls._FALLBACK_DIMENSIONS
        digest = hashlib.sha256(compact.encode("utf-8")).digest()
        fallback[int.from_bytes(digest[:2], "big") % cls._FALLBACK_DIMENSIONS] = 0.15
        vector.extend(fallback)
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return self._embed(text)


def write_markdown(
    path: Path,
    *,
    title: str,
    body: str,
    source_url: str = "https://www.scut.edu.cn/example",
    verified_at: str = "2026-08-27",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "---",
                f"title: {title}",
                f"source_url: {source_url}",
                "authority: official",
                "verification_status: verified",
                f"verified_at: {verified_at}",
                "auth_required: false",
                "---",
                "",
                f"# {title}",
                "",
                body,
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_minimal_docx(path: Path, paragraphs: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    escaped = [
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        for text in paragraphs
    ]
    paragraph_xml = "".join(f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>" for text in escaped)
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{paragraph_xml}</w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="xml" ContentType="application/xml"/>'
        "</Types>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("word/document.xml", document_xml)


def write_sidecar(path: Path, payload: dict[str, object]) -> None:
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
