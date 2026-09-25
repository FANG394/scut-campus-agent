from __future__ import annotations

import re
import unicodedata
from collections import Counter


TOKENIZER_VERSION = "cjk-ngram-v4"
_SEGMENT = re.compile(r"[\u3400-\u9fff]+|[a-z0-9]+(?:[-_.:/][a-z0-9]+)*", re.IGNORECASE)
_NOISE = {
    "的",
    "了",
    "呢",
    "吗",
    "啊",
    "呀",
    "请",
    "问",
    "请问",
    "一下",
    "如何",
    "怎么",
    "怎样",
    "什么",
    "可以",
    "能否",
    "我",
    "你",
    "帮我",
}

SYNONYM_GROUPS = (
    ("校园网", "校园网络", "学校网络", "宿舍网络", "公共无线网", "网络接入", "上网服务"),
    ("新生", "大一学生", "新入学学生"),
    ("开通", "申请", "办理", "激活"),
    ("图书馆", "图书馆舍", "馆舍"),
    ("开放时间", "开馆时间", "闭馆时间", "几点关闭", "几点关门", "几点闭馆", "闭馆", "关门"),
    ("统一认证", "统一身份认证", "学校账号"),
    ("校外访问", "校外使用", "在家访问"),
    ("位置", "地址", "在哪里", "在哪", "怎么走"),
    ("丢失", "遗失", "挂失", "补办"),
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> list[str]:
    output: list[str] = []
    for segment in _SEGMENT.findall(normalize(text)):
        if re.fullmatch(r"[\u3400-\u9fff]+", segment):
            length = len(segment)
            if 1 < length <= 12 and segment not in _NOISE:
                output.append(segment)
            for char in segment:
                if char not in _NOISE:
                    output.append(char)
            for size in (2, 3):
                if length < size:
                    continue
                output.extend(
                    gram
                    for gram in (segment[index : index + size] for index in range(length - size + 1))
                    if gram not in _NOISE
                )
        elif segment not in _NOISE:
            output.append(segment)
    return output


def token_counts(text: str) -> dict[str, int]:
    return dict(Counter(_tokens(text)))


def query_terms(text: str) -> tuple[dict[str, float], dict[str, float]]:
    primary_counts = Counter(_tokens(text))
    primary: dict[str, float] = {}
    for token in primary_counts:
        if token in _NOISE:
            continue
        if re.fullmatch(r"[\u3400-\u9fff]", token):
            weight = 0.22
        elif re.fullmatch(r"[\u3400-\u9fff]{2}", token):
            weight = 1.1
        elif re.fullmatch(r"[\u3400-\u9fff]{3}", token):
            weight = 1.45
        else:
            weight = 1.7
        primary[token] = weight

    expanded: dict[str, float] = {}
    normalised_query = normalize(text)
    for group in SYNONYM_GROUPS:
        if not any(alias in normalised_query for alias in group):
            continue
        for alias in group:
            if alias in normalised_query:
                continue
            for token in set(_tokens(alias)):
                expanded[token] = max(expanded.get(token, 0.0), primary.get(token, 0.0), 0.28)
    return primary, expanded


def semantic_group_coverage(query: str, document_text: str) -> tuple[float, int]:
    """Return alias-group coverage and the number of concepts expressed by the query."""

    compact_query = re.sub(r"\s+", "", normalize(query))
    compact_document = re.sub(r"\s+", "", normalize(document_text))
    triggered = [
        group for group in SYNONYM_GROUPS if any(alias in compact_query for alias in group)
    ]
    if not triggered:
        return 0.0, 0
    matched = sum(
        1 for group in triggered if any(alias in compact_document for alias in group)
    )
    return matched / len(triggered), len(triggered)


def core_phrase(text: str) -> str:
    value = normalize(text)
    value = re.sub(
        r"(?:请问|麻烦|帮我|我想知道|怎么办|如何处理|怎么处理|如何|怎么|怎样|可以|能否|一下)",
        "",
        value,
    )
    value = re.sub(r"[^\u3400-\u9fffa-z0-9]+", "", value)
    return value[:40]
