from __future__ import annotations

import re


_SENSITIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "密码或密钥",
        re.compile(
            r"(?:密码|口令|pass(?:word)?|api[_ -]?key|access[_ -]?token)\s*"
            r"(?:(?:是|为)\s*)?(?:[:：=]\s*)?[A-Za-z0-9!@#$%^&*_.\-]{4,}",
            re.IGNORECASE,
        ),
    ),
    (
        "验证码",
        re.compile(r"(?:验证码|校验码|otp)\s*(?:(?:是|为)\s*)?(?:[:：=]\s*)?\d{4,8}", re.I),
    ),
    ("身份证号", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
    (
        "学号或账号",
        re.compile(r"(?:学号|账号)\s*(?:(?:是|为)\s*)?(?:[:：=]\s*)?[A-Za-z0-9_-]{6,}"),
    ),
    ("手机号码", re.compile(r"(?:手机号|手机号码)\s*(?:[:：=]\s*)?1\d{10}")),
)


def detect_sensitive_disclosure(text: str) -> list[str]:
    return [label for label, pattern in _SENSITIVE_PATTERNS if pattern.search(text)]


def redact_sensitive_text(text: str) -> str:
    redacted = text
    for _label, pattern in _SENSITIVE_PATTERNS:
        redacted = pattern.sub("[已脱敏]", redacted)
    return redacted
