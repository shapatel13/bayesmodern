from __future__ import annotations

import re


PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "phone": re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"),
    "mrn_like": re.compile(r"\b(?:MRN|Medical Record Number)[:\s#-]*[A-Z0-9-]{5,}\b", re.IGNORECASE),
}


def detect_pii(text: str) -> list[str]:
    findings: list[str] = []
    for label, pattern in PII_PATTERNS.items():
        if pattern.search(text):
            findings.append(label)
    return findings


def redact_pii(text: str) -> str:
    redacted = text
    for label, pattern in PII_PATTERNS.items():
        redacted = pattern.sub(f"[REDACTED_{label.upper()}]", redacted)
    return redacted

