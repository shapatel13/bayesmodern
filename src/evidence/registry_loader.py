from __future__ import annotations

import csv
import json
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import AnyUrl, BaseModel, ConfigDict, Field, field_validator, model_validator


class SourceType(StrEnum):
    META_ANALYSIS = "meta_analysis"
    SYSTEMATIC_REVIEW = "systematic_review"
    PRIMARY_STUDY = "primary_study"
    GUIDELINE = "guideline"
    EXPERT_ESTIMATE = "expert_estimate"
    LOCAL_OVERRIDE = "local_override"


class EvidenceStatus(StrEnum):
    SOURCED = "sourced"
    ESTIMATED = "estimated"
    SEED_ONLY = "seed_only"
    DEPRECATED = "deprecated"


class TestCategory(StrEnum):
    HISTORY = "history"
    PHYSICAL_EXAM = "physical_exam"
    LAB = "lab"
    IMAGING = "imaging"
    ULTRASOUND = "ultrasound"
    CLINICAL_SCORE = "clinical_score"
    ECG = "ecg"
    MICROBIOLOGY = "microbiology"
    PROCEDURE = "procedure"
    COMPOSITE_RULE = "composite_rule"


class Setting(StrEnum):
    ED = "ED"
    ICU = "ICU"
    OUTPATIENT = "outpatient"
    INPATIENT = "inpatient"
    URGENT_CARE = "urgent_care"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class AgeGroup(StrEnum):
    ADULT = "adult"
    PEDIATRIC = "pediatric"
    GERIATRIC = "geriatric"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ApplicabilityFlag(StrEnum):
    ICU = "ICU"
    ED = "ED"
    OUTPATIENT = "outpatient"
    INPATIENT = "inpatient"
    PREGNANT = "pregnant"
    PEDIATRICS = "pediatrics"
    ELDERLY = "elderly"
    IMMUNOCOMPROMISED = "immunocompromised"
    RENAL_RISK = "renal_risk"
    RADIATION_RISK = "radiation_risk"
    INVASIVE_TEST = "invasive_test"
    LOW_COST = "low_cost"
    HIGH_COST = "high_cost"
    BEDSIDE = "bedside"
    SEND_OUT = "send_out"


SOURCE_QUALITY_SCORE: dict[SourceType, int] = {
    SourceType.LOCAL_OVERRIDE: 6,
    SourceType.META_ANALYSIS: 5,
    SourceType.SYSTEMATIC_REVIEW: 4,
    SourceType.GUIDELINE: 3,
    SourceType.PRIMARY_STUDY: 2,
    SourceType.EXPERT_ESTIMATE: 1,
}

NUMERIC_EVIDENCE_FIELDS = (
    "lr_positive",
    "lr_negative",
    "sensitivity",
    "specificity",
    "ci_95_lower_lr_positive",
    "ci_95_upper_lr_positive",
    "ci_95_lower_lr_negative",
    "ci_95_upper_lr_negative",
)

DEFAULT_JSONL_PATH = Path(__file__).resolve().parent / "data" / "lr_registry.seed.jsonl"
DEFAULT_CSV_PATH = Path(__file__).resolve().parent / "data" / "lr_registry.seed.csv"
DEFAULT_CURATED_JSONL_PATH = Path(__file__).resolve().parent / "data" / "lr_registry.curated.jsonl"
DEFAULT_CURATED_CSV_PATH = Path(__file__).resolve().parent / "data" / "lr_registry.curated.csv"


class LRPopulation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age_group: AgeGroup
    pregnancy: bool | None = None
    immunocompromised: bool | None = None
    renal_impairment_relevant: bool | None = None
    icu_population: bool | None = None
    notes: str | None = None


class LREntry(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    id: str = Field(pattern=r"^[a-z0-9_\-\.]+$")
    condition: str = Field(min_length=1)
    condition_group: str | None = None
    test_or_finding: str = Field(min_length=1)
    test_category: TestCategory | None = None
    comparator_or_threshold: str | None = None
    target_outcome: str | None = None
    lr_positive: float | None = Field(default=None, gt=0)
    lr_negative: float | None = Field(default=None, gt=0)
    sensitivity: float | None = Field(default=None, ge=0, le=1)
    specificity: float | None = Field(default=None, ge=0, le=1)
    ci_95_lower_lr_positive: float | None = Field(default=None, gt=0)
    ci_95_upper_lr_positive: float | None = Field(default=None, gt=0)
    ci_95_lower_lr_negative: float | None = Field(default=None, gt=0)
    ci_95_upper_lr_negative: float | None = Field(default=None, gt=0)
    pretest_anchor_min: float | None = Field(default=None, ge=0, le=1)
    pretest_anchor_max: float | None = Field(default=None, ge=0, le=1)
    setting: list[Setting] = Field(min_length=1)
    population: LRPopulation
    applicability_flags: list[ApplicabilityFlag] = Field(default_factory=list)
    harms_or_constraints: list[str] = Field(default_factory=list)
    direct_cost_usd: float | None = Field(default=None, ge=0)
    downstream_cascade_cost_usd: float | None = Field(default=None, ge=0)
    turnaround_time_hours: float | None = Field(default=None, ge=0)
    source_type: SourceType
    source_citation: str | None = None
    source_url: AnyUrl | None = None
    pmid_or_doi: str | None = None
    publication_year: int | None = Field(default=None, ge=1900, le=2100)
    evidence_status: EvidenceStatus
    notes_for_model: str | None = None
    preference_weight_hint: float | None = Field(default=None, ge=0, le=1)
    status_updated_at: date

    @field_validator("condition", "condition_group", "test_or_finding", "notes_for_model", "source_citation")
    @classmethod
    def strip_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def validate_consistency(self) -> "LREntry":
        if self.pretest_anchor_min is not None and self.pretest_anchor_max is not None:
            if self.pretest_anchor_min > self.pretest_anchor_max:
                raise ValueError("pretest_anchor_min cannot exceed pretest_anchor_max.")

        lr_bounds = (
            (self.ci_95_lower_lr_positive, self.ci_95_upper_lr_positive, "lr_positive"),
            (self.ci_95_lower_lr_negative, self.ci_95_upper_lr_negative, "lr_negative"),
        )
        for low, high, label in lr_bounds:
            if low is not None and high is not None and low > high:
                raise ValueError(f"{label} lower CI cannot exceed upper CI.")

        has_numeric_evidence = any(getattr(self, field_name) is not None for field_name in NUMERIC_EVIDENCE_FIELDS)
        has_provenance = bool(self.source_citation or self.source_url or self.pmid_or_doi)

        if self.evidence_status == EvidenceStatus.SOURCED and not self.source_citation:
            raise ValueError("Sourced rows require source_citation.")
        if self.evidence_status == EvidenceStatus.SEED_ONLY and has_numeric_evidence:
            raise ValueError("seed_only rows cannot contain numeric LR/sensitivity/specificity evidence.")
        if has_numeric_evidence and not has_provenance:
            raise ValueError("Numeric LR/sensitivity/specificity evidence requires provenance.")
        if self.evidence_status == EvidenceStatus.ESTIMATED and self.source_type not in {
            SourceType.EXPERT_ESTIMATE,
            SourceType.LOCAL_OVERRIDE,
        }:
            raise ValueError("Estimated rows should use expert_estimate or local_override source_type.")
        return self

    @property
    def provenance_score(self) -> int:
        return SOURCE_QUALITY_SCORE[self.source_type]

    @property
    def confidence_label(self) -> str:
        if self.evidence_status == EvidenceStatus.ESTIMATED:
            return "lower_confidence_estimated"
        if self.evidence_status == EvidenceStatus.SEED_ONLY:
            return "seed_registry_only"
        if self.evidence_status == EvidenceStatus.DEPRECATED:
            return "deprecated"
        return "sourced"


def _coerce_optional_number(value: str | None, *, integer: bool = False) -> int | float | None:
    if value is None:
        return None
    stripped = value.strip()
    if stripped in {"", "null", "None"}:
        return None
    return int(stripped) if integer else float(stripped)


def _coerce_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _coerce_json_field(value: str | None, default: Any) -> Any:
    if value is None:
        return default
    stripped = value.strip()
    if stripped in {"", "null", "None"}:
        return default
    return json.loads(stripped)


def _entry_from_csv_row(row: dict[str, str]) -> dict[str, Any]:
    numeric_fields = {
        "lr_positive",
        "lr_negative",
        "sensitivity",
        "specificity",
        "ci_95_lower_lr_positive",
        "ci_95_upper_lr_positive",
        "ci_95_lower_lr_negative",
        "ci_95_upper_lr_negative",
        "pretest_anchor_min",
        "pretest_anchor_max",
        "direct_cost_usd",
        "downstream_cascade_cost_usd",
        "turnaround_time_hours",
        "preference_weight_hint",
    }
    integer_fields = {"publication_year"}

    payload: dict[str, Any] = {}
    for key, value in row.items():
        if key in {"setting", "applicability_flags", "harms_or_constraints"}:
            payload[key] = _coerce_json_field(value, [])
        elif key == "population":
            payload[key] = _coerce_json_field(value, {})
        elif key in numeric_fields:
            payload[key] = _coerce_optional_number(value)
        elif key in integer_fields:
            payload[key] = _coerce_optional_number(value, integer=True)
        else:
            payload[key] = _coerce_optional_string(value)
    return payload


def load_registry(path: str | Path) -> list[LREntry]:
    registry_path = Path(path)
    if registry_path.suffix.lower() == ".jsonl":
        return load_registry_jsonl(registry_path)
    if registry_path.suffix.lower() == ".csv":
        return load_registry_csv(registry_path)
    raise ValueError(f"Unsupported registry format: {registry_path.suffix}")


def load_registry_jsonl(path: str | Path) -> list[LREntry]:
    registry_path = Path(path)
    entries: list[LREntry] = []
    for raw_line in registry_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        entries.append(LREntry.model_validate(json.loads(line)))
    return entries


def load_registry_csv(path: str | Path) -> list[LREntry]:
    registry_path = Path(path)
    with registry_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [LREntry.model_validate(_entry_from_csv_row(row)) for row in reader]


def load_seed_registry(prefer: str = "jsonl") -> list[LREntry]:
    if prefer == "jsonl":
        return load_registry_jsonl(DEFAULT_JSONL_PATH)
    if prefer == "csv":
        return load_registry_csv(DEFAULT_CSV_PATH)
    raise ValueError("prefer must be either 'jsonl' or 'csv'.")


def load_curated_registry(prefer: str = "jsonl") -> list[LREntry]:
    if prefer == "jsonl":
        path = DEFAULT_CURATED_JSONL_PATH
        return load_registry_jsonl(path) if path.exists() else []
    if prefer == "csv":
        path = DEFAULT_CURATED_CSV_PATH
        return load_registry_csv(path) if path.exists() else []
    raise ValueError("prefer must be either 'jsonl' or 'csv'.")


def load_default_registry_bundle(prefer: str = "jsonl") -> list[LREntry]:
    entries = [*load_seed_registry(prefer=prefer), *load_curated_registry(prefer=prefer)]
    seen_ids: set[str] = set()
    duplicates: list[str] = []
    for entry in entries:
        if entry.id in seen_ids:
            duplicates.append(entry.id)
        seen_ids.add(entry.id)
    if duplicates:
        duplicate_list = ", ".join(sorted(set(duplicates)))
        raise ValueError(f"Duplicate LR registry ids in default bundle: {duplicate_list}")
    return entries
