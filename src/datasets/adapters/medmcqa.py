from __future__ import annotations

from datasets.task_builders.common import BenchmarkTask
from datasets.transforms.normalize import as_text


def normalize_medmcqa_row(row: dict[str, object], split: str) -> BenchmarkTask:
    choices = [as_text(row.get("opa")), as_text(row.get("opb")), as_text(row.get("opc")), as_text(row.get("opd"))]
    correct_index = int(row.get("cop", 0))
    gold_answer = choices[correct_index] if 0 <= correct_index < len(choices) else None
    return BenchmarkTask(
        task_id=f"medmcqa-{row.get('id', row.get('question_id', 'unknown'))}",
        source_dataset="medmcqa",
        split=split,
        task_type="diagnosis_mcq",
        prompt=as_text(row.get("question")),
        choices=choices,
        gold_answer=gold_answer,
        metadata={"subject": as_text(row.get("subject_name")), "topic": as_text(row.get("topic_name"))},
    )

