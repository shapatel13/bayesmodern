from __future__ import annotations

from datasets.transforms.normalize import as_text
from priorix_tasks.common import BenchmarkTask


def normalize_findzebra_row(row: dict[str, object], split: str) -> BenchmarkTask:
    prompt = as_text(row.get("case_text") or row.get("text") or row.get("question"))
    return BenchmarkTask(
        task_id=f"findzebra-{row.get('id', 'unknown')}",
        source_dataset="findzebra",
        split=split,
        task_type="diagnosis_open",
        prompt=prompt,
        gold_diagnosis=as_text(row.get("diagnosis") or row.get("answer")),
        metadata={"rare_disease": True},
    )
