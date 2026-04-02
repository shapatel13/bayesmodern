from __future__ import annotations

from datasets.transforms.normalize import as_text
from priorix_tasks.common import BenchmarkTask


def normalize_medval_bench_row(row: dict[str, object], split: str) -> BenchmarkTask:
    row_number = as_text(row.get("#") or row.get("row_number") or "unknown")
    medval_id = as_text(row.get("id") or f"medval-{row_number}")
    medval_task = as_text(row.get("task") or "medval_bench")
    source_input = as_text(row.get("input"))
    candidate_output = as_text(row.get("output"))
    reference_output = as_text(row.get("reference_output"))
    physician_error_assessment = as_text(row.get("physician_error_assessment"))
    try:
        gold_risk_grade = int(as_text(row.get("physician_risk_grade") or "0"))
    except ValueError:
        gold_risk_grade = None

    prompt_sections = [
        f"MedVAL-Bench task: {medval_task}",
        "Source input:",
        source_input,
        "",
        "Candidate output to audit:",
        candidate_output,
    ]
    if reference_output:
        prompt_sections.extend(["", "Reference output:", reference_output])

    metadata = {
        "row_number": row_number,
        "medval_task": medval_task,
        "source_input": source_input,
        "candidate_output": candidate_output,
        "reference_output": reference_output,
        "physician_error_assessment": physician_error_assessment,
        "physician_risk_grade": gold_risk_grade,
        "physician_reference_available": bool(physician_error_assessment),
        "reference_available": bool(reference_output),
    }
    return BenchmarkTask(
        task_id=f"medval-{medval_task}-{medval_id}",
        source_dataset="medval_bench",
        split=split,
        task_type="generation_audit",
        prompt="\n".join(prompt_sections).strip(),
        gold_risk_grade=gold_risk_grade,
        metadata=metadata,
    )
