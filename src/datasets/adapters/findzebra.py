from __future__ import annotations

from datasets.transforms.normalize import as_text
from priorix_tasks.common import BenchmarkTask


def _coalesce_text(value: object) -> str:
    if isinstance(value, list):
        parts = [as_text(item) for item in value if as_text(item)]
        return "\n\n".join(parts[:2]).strip()
    return as_text(value)


def _first_present(row: dict[str, object], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        text = _coalesce_text(value)
        if text:
            return text
    return ""


def _first_label(value: object) -> str:
    if isinstance(value, list):
        for item in value:
            text = as_text(item)
            if text:
                return text
        return ""
    return as_text(value)


def normalize_findzebra_row(row: dict[str, object], split: str) -> BenchmarkTask:
    prompt = _first_present(row, "case_text", "question", "text", "abstract", "title")
    diagnosis = _first_label(
        row.get("diagnosis")
        or row.get("answer")
        or row.get("disease")
        or row.get("diagnoses")
        or row.get("conditions")
        or row.get("labels")
    )
    return BenchmarkTask(
        task_id=f"findzebra-{row.get('id', 'unknown')}",
        source_dataset="findzebra",
        split=split,
        task_type="diagnosis_open",
        prompt=prompt,
        gold_diagnosis=diagnosis or None,
        metadata={"rare_disease": True, "title": as_text(row.get("title"))},
    )
