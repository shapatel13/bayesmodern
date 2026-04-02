from __future__ import annotations

from datasets.transforms.normalize import as_text
from priorix_tasks.common import BenchmarkTask


def normalize_demo_case_row(row: dict[str, object], split: str) -> BenchmarkTask:
    acceptable_tests = row.get("acceptable_tests")
    if isinstance(acceptable_tests, list):
        tests = [as_text(item) for item in acceptable_tests if as_text(item)]
    elif isinstance(acceptable_tests, str):
        tests = [item.strip() for item in acceptable_tests.split(",") if item.strip()]
    else:
        tests = []

    return BenchmarkTask(
        task_id=f"priorix-demo-{row.get('id', 'unknown')}",
        source_dataset="priorix_demo_cases",
        split=split,
        task_type=as_text(row.get("task_type") or "diagnosis_open"),  # type: ignore[arg-type]
        prompt=as_text(row.get("prompt") or row.get("case_text")),
        gold_answer=as_text(row.get("gold_answer")) or None,
        gold_diagnosis=as_text(row.get("gold_diagnosis")) or None,
        acceptable_tests=tests,
        gold_triage=as_text(row.get("gold_triage")) or None,
        metadata={
            "title": as_text(row.get("title")) or None,
            "track": as_text(row.get("track")) or "demo",
            "notes": as_text(row.get("notes")) or None,
        },
    )
