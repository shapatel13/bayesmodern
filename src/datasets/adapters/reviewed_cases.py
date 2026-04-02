from __future__ import annotations

import json

from priorix_tasks.common import BenchmarkTask, TaskType


def _as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in text.split(",") if item.strip()]


def _resolve_task_type(value: object) -> TaskType:
    raw = str(value or "diagnosis_open").strip().lower()
    allowed: set[str] = {
        "diagnosis_mcq",
        "diagnosis_open",
        "next_best_test",
        "triage",
        "evidence_verification",
        "medication_safety",
        "generation_audit",
    }
    if raw not in allowed:
        return "diagnosis_open"
    return raw  # type: ignore[return-value]


def normalize_reviewed_case_row(row: dict[str, object], split: str) -> BenchmarkTask:
    task_type = _resolve_task_type(row.get("task_type"))
    task_id = str(row.get("id") or row.get("case_id") or row.get("task_id") or "reviewed-case").strip()
    prompt = str(row.get("note_text") or row.get("prompt") or row.get("case_text") or "").strip()
    choices = _as_list(row.get("choices"))
    acceptable_tests = _as_list(row.get("acceptable_tests"))
    metadata = {
        "review_status": str(row.get("review_status") or "draft").strip().lower(),
        "reviewer_id": str(row.get("reviewer_id") or "").strip() or None,
        "review_notes": str(row.get("review_notes") or "").strip() or None,
        "tags": _as_list(row.get("tags")),
        "source": "reviewed_case",
    }
    return BenchmarkTask(
        task_id=f"reviewed-{task_id}",
        source_dataset="reviewed_cases",
        split=split,
        task_type=task_type,
        prompt=prompt,
        choices=choices,
        gold_answer=str(row.get("gold_answer") or "").strip() or None,
        gold_diagnosis=str(row.get("gold_diagnosis") or "").strip() or None,
        acceptable_tests=acceptable_tests,
        gold_triage=str(row.get("gold_triage") or "").strip() or None,
        gold_risk_grade=int(row["gold_risk_grade"]) if row.get("gold_risk_grade") not in {None, ""} else None,
        metadata=metadata,
    )
