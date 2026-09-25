from __future__ import annotations

import os
import ipaddress
import math
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


def _load_dotenv(path: Path) -> None:
    """Load a small, dependency-free subset of dotenv syntax.

    Existing process environment variables always win. The parser deliberately
    ignores shell expansion so secrets are never executed as code.
    """

    if not path.is_file():
        return
    process_keys = set(os.environ)
    parsed: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not key.replace("_", "").isalnum():
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        parsed[key] = value
    for key, value in parsed.items():
        if key not in process_keys:
            os.environ[key] = value


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return min(max(value, minimum), maximum)


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if not math.isfinite(value):
        return default
    return min(max(value, minimum), maximum)


def _resolve_under(root: Path, raw: str | None, fallback: str) -> Path:
    candidate = Path(raw) if raw else Path(fallback)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def _valid_llm_base_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        return False
    if parsed.scheme == "https":
        return True
    if parsed.scheme != "http":
        return False
    if parsed.hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True, slots=True)
class AppConfig:
    project_root: Path
    host: str
    port: int
    provider: str
    api_style: str
    api_key: str
    model: str
    base_url: str
    ollama_base_url: str
    embedding_model: str
    embedding_batch_size: int
    llm_timeout_seconds: float
    llm_max_output_tokens: int
    llm_chat_token_field: str
    max_history_messages: int
    knowledge_source_dir: Path
    knowledge_index_file: Path
    tool_data_dir: Path
    rag_top_k: int
    rag_min_score: float
    rag_max_context_chars: int
    rag_chunk_size: int
    rag_chunk_overlap: int

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "AppConfig":
        root = (project_root or Path(__file__).resolve().parent.parent).resolve()
        _load_dotenv(root / ".env")

        provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower() or "ollama"
        api_style = os.getenv(
            "LLM_API_STYLE",
            "responses" if provider == "openai" else (
                "ollama" if provider == "ollama" else "chat_completions"
            ),
        ).strip().lower()
        base_url = os.getenv("LLM_BASE_URL", "").strip().rstrip("/")
        if provider == "openai" and not base_url:
            base_url = "https://api.openai.com/v1"
        ollama_base_url = (
            os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip().rstrip("/")
            or "http://127.0.0.1:11434"
        )
        if provider == "ollama" and not base_url:
            base_url = ollama_base_url

        configured_model = os.getenv("LLM_MODEL", "").strip()
        if provider == "ollama" and not configured_model:
            configured_model = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:4b").strip() or "qwen3:4b"

        return cls(
            project_root=root,
            host=os.getenv("APP_HOST", "127.0.0.1").strip() or "127.0.0.1",
            port=_env_int("APP_PORT", 8000, minimum=1, maximum=65535),
            provider=provider,
            api_style=api_style,
            api_key=(os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip(),
            model=configured_model,
            base_url=base_url,
            ollama_base_url=ollama_base_url,
            embedding_model=(
                os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:4b").strip()
                or "qwen3-embedding:4b"
            ),
            embedding_batch_size=_env_int(
                "OLLAMA_EMBEDDING_BATCH_SIZE", 16, minimum=1, maximum=128
            ),
            llm_timeout_seconds=_env_float(
                "LLM_TIMEOUT_SECONDS", 120.0, minimum=1.0, maximum=300.0
            ),
            llm_max_output_tokens=_env_int(
                "LLM_MAX_OUTPUT_TOKENS", 2048, minimum=64, maximum=8192
            ),
            llm_chat_token_field=(
                os.getenv("LLM_CHAT_TOKEN_FIELD", "max_completion_tokens").strip()
                if os.getenv("LLM_CHAT_TOKEN_FIELD", "").strip()
                in {"max_completion_tokens", "max_tokens"}
                else "max_completion_tokens"
            ),
            max_history_messages=_env_int(
                "APP_MAX_HISTORY_MESSAGES", 12, minimum=2, maximum=40
            ),
            knowledge_source_dir=_resolve_under(
                root,
                os.getenv("KNOWLEDGE_SOURCE_DIR"),
                "data/knowledge/source",
            ),
            knowledge_index_file=_resolve_under(
                root,
                os.getenv("KNOWLEDGE_INDEX_FILE"),
                "data/knowledge/index/index.json",
            ),
            tool_data_dir=_resolve_under(
                root,
                os.getenv("TOOL_DATA_DIR"),
                "data/tools",
            ),
            rag_top_k=_env_int("RAG_TOP_K", 4, minimum=1, maximum=12),
            rag_min_score=_env_float(
                "RAG_MIN_SCORE", 0.60, minimum=0.05, maximum=0.95
            ),
            rag_max_context_chars=_env_int(
                "RAG_MAX_CONTEXT_CHARS", 6000, minimum=500, maximum=30000
            ),
            rag_chunk_size=_env_int(
                "RAG_CHUNK_SIZE", 700, minimum=200, maximum=3000
            ),
            rag_chunk_overlap=_env_int(
                "RAG_CHUNK_OVERLAP", 100, minimum=0, maximum=500
            ),
        )

    @property
    def remote_llm_requested(self) -> bool:
        return self.provider not in {"", "local", "demo", "offline", "ollama"}

    @property
    def ollama_requested(self) -> bool:
        return self.provider == "ollama"

    @property
    def ollama_configured(self) -> bool:
        return bool(
            self.ollama_requested
            and self.model
            and self.embedding_model
            and _valid_llm_base_url(self.ollama_base_url)
        )

    @property
    def remote_llm_configured(self) -> bool:
        return bool(
            self.remote_llm_requested
            and self.api_key
            and self.model
            and _valid_llm_base_url(self.base_url)
            and self.api_style in {"responses", "chat_completions"}
        )

    @property
    def llm_requested(self) -> bool:
        return self.ollama_requested or self.remote_llm_requested

    @property
    def llm_configured(self) -> bool:
        return self.ollama_configured or self.remote_llm_configured

    @property
    def provider_status(self) -> dict[str, object]:
        if self.ollama_requested:
            missing: list[str] = []
            if not self.model:
                missing.append("LLM_MODEL")
            if not self.embedding_model:
                missing.append("OLLAMA_EMBEDDING_MODEL")
            if not _valid_llm_base_url(self.ollama_base_url):
                missing.append("OLLAMA_BASE_URL（仅本机地址可使用 HTTP）")
            return {
                "name": "ollama",
                "configured": not missing,
                "model": self.model or None,
                "embedding_model": self.embedding_model or None,
                "api_style": "ollama",
                "message": (
                    "本地 Ollama 配置完整"
                    if not missing
                    else f"缺少或无效配置：{', '.join(missing)}"
                ),
            }
        if not self.remote_llm_requested:
            return {
                "name": "local",
                "configured": True,
                "model": "extractive-demo",
                "embedding_model": self.embedding_model,
                "api_style": "offline",
                "message": "当前为离线演示模式，不会调用外部大模型。",
            }
        missing: list[str] = []
        if not self.api_key:
            missing.append("LLM_API_KEY")
        if not self.model:
            missing.append("LLM_MODEL")
        if not _valid_llm_base_url(self.base_url):
            missing.append("LLM_BASE_URL（需 HTTPS；仅本机地址可使用 HTTP）")
        if self.api_style not in {"responses", "chat_completions"}:
            missing.append("LLM_API_STYLE")
        return {
            "name": self.provider,
            "configured": not missing,
            "model": self.model or None,
            "api_style": self.api_style,
            "message": "配置完整" if not missing else f"缺少或无效配置：{', '.join(missing)}",
        }
