from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from datasets.adapters.findzebra import normalize_findzebra_row
from datasets.adapters.medmcqa import normalize_medmcqa_row
from datasets.adapters.medqa import normalize_medqa_row
from datasets.adapters.mietic import normalize_mietic_row
from datasets.adapters.n2c2_2018_track2 import normalize_n2c2_2018_track2_row
from datasets.adapters.pubmedqa import normalize_pubmedqa_row
from priorix_tasks.common import BenchmarkTask


Normalizer = Callable[[dict[str, object], str], BenchmarkTask]
DatasetAccessMode = Literal["hf", "local", "hybrid"]


@dataclass(frozen=True)
class BenchmarkDatasetSpec:
    key: str
    hf_dataset: str | None
    adapter: Normalizer
    access_mode: DatasetAccessMode = "hf"
    default_subset: str | None = None
    default_train_split: str = "train"
    default_val_split: str = "validation"
    default_eval_split: str = "test"
    description: str = ""
    task_family: str = ""
    requires_credentials: bool = False
    notes: str = ""
    local_path_env: str | None = None
    local_path_setting: str | None = None
    local_split_filenames: dict[str, tuple[str, ...]] = field(default_factory=dict)


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
    "mietic": BenchmarkDatasetSpec(
        key="mietic",
        hf_dataset=None,
        adapter=normalize_mietic_row,
        access_mode="local",
        default_train_split="train",
        default_val_split="validation",
        default_eval_split="test",
        description="Emergency Department triage instruction corpus for acuity prediction and rationale evaluation.",
        task_family="triage",
        requires_credentials=True,
        notes="Credential-gated PhysioNet dataset. Point PRIORI_MIETIC_PATH to a local CSV/JSON/JSONL export.",
        local_path_env="PRIORI_MIETIC_PATH",
        local_path_setting="mietic_path",
        local_split_filenames={
            "train": ("train_40000_perceived_triage.csv", "train.csv", "train.jsonl", "train.json"),
            "validation": (
                "mietic_validation_500.csv",
                "valid_1500_perceived_triage.csv",
                "validation.csv",
                "validation.jsonl",
                "validation.json",
            ),
            "test": ("test_1500_perceived_triage.csv", "test.csv", "test.jsonl", "test.json"),
        },
    ),
    "n2c2_2018_track2": BenchmarkDatasetSpec(
        key="n2c2_2018_track2",
        hf_dataset="bigbio/n2c2_2018_track2",
        adapter=normalize_n2c2_2018_track2_row,
        access_mode="hybrid",
        default_train_split="train",
        default_val_split="validation",
        default_eval_split="test",
        description="Adverse drug event and medication extraction benchmark from clinical notes.",
        task_family="medication_safety",
        requires_credentials=True,
        notes=(
            "Credential-gated benchmark. Prefer a local export via PRIORI_N2C2_2018_TRACK2_PATH; "
            "HF/BigBio access may require additional setup."
        ),
        local_path_env="PRIORI_N2C2_2018_TRACK2_PATH",
        local_path_setting="n2c2_2018_track2_path",
        local_split_filenames={
            "train": ("train.jsonl", "train.json", "train.csv"),
            "validation": ("validation.jsonl", "validation.json", "validation.csv"),
            "test": ("test.jsonl", "test.json", "test.csv"),
        },
    ),
}


def get_dataset_spec(dataset_key: str) -> BenchmarkDatasetSpec:
    normalized = dataset_key.strip().lower()
    if normalized not in DATASET_SPECS:
        raise KeyError(f"Unsupported benchmark dataset: {dataset_key}")
    return DATASET_SPECS[normalized]


def list_dataset_specs() -> list[BenchmarkDatasetSpec]:
    return list(DATASET_SPECS.values())
