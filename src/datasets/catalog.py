from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from datasets.adapters.findzebra import normalize_findzebra_row
from datasets.adapters.medmcqa import normalize_medmcqa_row
from datasets.adapters.medqa import normalize_medqa_row
from datasets.adapters.pubmedqa import normalize_pubmedqa_row
from priorix_tasks.common import BenchmarkTask


Normalizer = Callable[[dict[str, object], str], BenchmarkTask]


@dataclass(frozen=True)
class BenchmarkDatasetSpec:
    key: str
    hf_dataset: str | None
    adapter: Normalizer
    default_subset: str | None = None
    default_train_split: str = "train"
    default_val_split: str = "validation"
    default_eval_split: str = "test"
    description: str = ""
    task_family: str = ""
    requires_credentials: bool = False
    notes: str = ""


DATASET_SPECS: dict[str, BenchmarkDatasetSpec] = {
    "medmcqa": BenchmarkDatasetSpec(
        key="medmcqa",
        hf_dataset="openlifescienceai/medmcqa",
        adapter=normalize_medmcqa_row,
        default_train_split="train",
        default_val_split="validation",
        default_eval_split="test",
        description="Broad medical MCQ benchmark for subject-level reasoning.",
        task_family="diagnosis_mcq",
    ),
    "medqa": BenchmarkDatasetSpec(
        key="medqa",
        hf_dataset="augtoma/medqa_usmle",
        adapter=normalize_medqa_row,
        default_train_split="train",
        default_val_split="validation",
        default_eval_split="test",
        description="USMLE-style difficult medical QA benchmark.",
        task_family="diagnosis_mcq",
    ),
    "pubmedqa": BenchmarkDatasetSpec(
        key="pubmedqa",
        hf_dataset="qiaojin/PubMedQA",
        adapter=normalize_pubmedqa_row,
        default_subset="pqa_labeled",
        default_train_split="train",
        default_val_split="train",
        default_eval_split="train",
        description="Biomedical evidence verification benchmark.",
        task_family="evidence_verification",
        notes="The labeled subset is small and often evaluated as a single split.",
    ),
    "findzebra": BenchmarkDatasetSpec(
        key="findzebra",
        hf_dataset="findzebra/case-reports",
        adapter=normalize_findzebra_row,
        default_train_split="train",
        default_val_split="train",
        default_eval_split="train",
        description="Rare disease case-report benchmark for open differential reasoning.",
        task_family="diagnosis_open",
        notes="Case-report fields are semi-structured and normalized heuristically into vignette tasks.",
    ),
}


def get_dataset_spec(dataset_key: str) -> BenchmarkDatasetSpec:
    normalized = dataset_key.strip().lower()
    if normalized not in DATASET_SPECS:
        raise KeyError(f"Unsupported benchmark dataset: {dataset_key}")
    return DATASET_SPECS[normalized]


def list_dataset_specs() -> list[BenchmarkDatasetSpec]:
    return list(DATASET_SPECS.values())
