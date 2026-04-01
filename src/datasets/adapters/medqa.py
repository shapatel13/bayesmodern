from __future__ import annotations

from datasets.transforms.normalize import as_text
from priorix_tasks.common import BenchmarkTask


def normalize_medqa_row(row: dict[str, object], split: str) -> BenchmarkTask:
    options = row.get("options")
    if isinstance(options, list):
        choices = [as_text(option) for option in options]
    elif isinstance(options, dict):
        choices = [as_text(value) for _, value in sorted(options.items())]
    else:
        choices = [as_text(row.get(key)) for key in ("opa", "opb", "opc", "opd", "ope") if row.get(key) is not None]
    answer = as_text(row.get("answer"))
    return BenchmarkTask(
        task_id=f"medqa-{row.get('id', 'unknown')}",
        source_dataset="medqa",
        split=split,
        task_type="diagnosis_mcq",
        prompt=as_text(row.get("question")),
        choices=choices,
        gold_answer=answer,
        metadata={"meta": {key: row[key] for key in row.keys() if key not in {'question', 'answer', 'options'}}},
    )
