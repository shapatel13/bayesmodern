from __future__ import annotations

import re

from datasets.transforms.normalize import as_text
from priorix_tasks.common import BenchmarkTask


_ESI_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\besi\s*[-:=]?\s*1\b", re.IGNORECASE), "emergent"),
    (re.compile(r"\besi\s*[-:=]?\s*2\b", re.IGNORECASE), "urgent"),
    (re.compile(r"\besi\s*[-:=]?\s*3\b", re.IGNORECASE), "expedited"),
    (re.compile(r"\besi\s*[-:=]?\s*[45]\b", re.IGNORECASE), "routine"),
    (re.compile(r"\bresuscitation\b|\bemergent\b|\bimmediate\b", re.IGNORECASE), "emergent"),
    (re.compile(r"\burgent\b|\bhigh acuity\b", re.IGNORECASE), "urgent"),
    (re.compile(r"\bexpedited\b|\bsemi-urgent\b", re.IGNORECASE), "expedited"),
    (re.compile(r"\broutine\b|\bnon-urgent\b|\blower acuity\b", re.IGNORECASE), "routine"),
]


def infer_triage_label(value: object) -> str | None:
    text = as_text(value)
    if not text:
        return None
    for pattern, label in _ESI_PATTERNS:
        if pattern.search(text):
            return label
    return None


def normalize_mietic_row(row: dict[str, object], split: str) -> BenchmarkTask:
    instruction = as_text(row.get("instruction"))
    case_text = as_text(row.get("input") or row.get("case") or row.get("clinical_note") or row.get("text"))
    output_text = as_text(row.get("output") or row.get("label") or row.get("triage"))
    prompt = "\n\n".join(part for part in [instruction, case_text] if part).strip()
    return BenchmarkTask(
        task_id=f"mietic-{row.get('id', row.get('encounter_id', 'unknown'))}",
        source_dataset="mietic",
        split=split,
        task_type="triage",
        prompt=prompt,
        gold_answer=output_text or None,
        gold_triage=infer_triage_label(output_text),
        metadata={
            "instruction": instruction or None,
            "original_output": output_text or None,
            "triage_source": "MIETIC",
            "credential_gated": True,
        },
    )
