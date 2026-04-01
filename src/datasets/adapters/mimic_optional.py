from __future__ import annotations

import os

from datasets.task_builders.common import BenchmarkTask
from datasets.transforms.normalize import as_text


def normalize_mimic_like_row(row: dict[str, object], split: str) -> BenchmarkTask:
    if os.getenv("PRIORI_ENABLE_MIMIC", "false").lower() != "true":
        raise PermissionError("MIMIC-like adapters are disabled by default and require explicit opt-in.")
    return BenchmarkTask(
        task_id=f"mimic-{row.get('stay_id', row.get('id', 'unknown'))}",
        source_dataset="mimic_optional",
        split=split,
        task_type="triage",
        prompt=as_text(row.get("clinical_note") or row.get("prompt")),
        gold_triage=as_text(row.get("triage") or row.get("acuity")),
        metadata={"credential_gated": True},
    )

