from __future__ import annotations

from datasets.transforms.normalize import as_text
from priorix_tasks.common import BenchmarkTask


def normalize_pubmedqa_row(row: dict[str, object], split: str) -> BenchmarkTask:
    contexts = row.get("contexts") or row.get("context")
    context_text = " ".join(as_text(item) for item in contexts) if isinstance(contexts, list) else as_text(contexts)
    prompt = f"{as_text(row.get('question'))}\n\nContext:\n{context_text}".strip()
    return BenchmarkTask(
        task_id=f"pubmedqa-{row.get('pubid', row.get('id', 'unknown'))}",
        source_dataset="pubmedqa",
        split=split,
        task_type="evidence_verification",
        prompt=prompt,
        choices=["yes", "no", "maybe"],
        gold_answer=as_text(row.get("final_decision") or row.get("answer")),
    )
