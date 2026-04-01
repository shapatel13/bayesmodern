from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


TaskType = Literal["diagnosis_mcq", "diagnosis_open", "next_best_test", "triage", "evidence_verification"]


class BenchmarkTask(BaseModel):
    task_id: str
    source_dataset: str
    split: str
    task_type: TaskType
    prompt: str
    choices: list[str] = Field(default_factory=list)
    gold_answer: str | None = None
    gold_diagnosis: str | None = None
    acceptable_tests: list[str] = Field(default_factory=list)
    gold_triage: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

