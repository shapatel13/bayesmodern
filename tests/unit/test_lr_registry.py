from __future__ import annotations

from datetime import date

import pytest

from evidence.query import query_registry, resolve_registry_entry
from evidence.registry_loader import (
    DEFAULT_CSV_PATH,
    DEFAULT_CURATED_CSV_PATH,
    DEFAULT_CURATED_JSONL_PATH,
    DEFAULT_JSONL_PATH,
    EvidenceStatus,
    LREntry,
    SourceType,
    load_curated_registry,
    load_default_registry_bundle,
    load_registry,
)
from evidence.registry_validator import RegistryValidationError, validate_registry_entries


def _entry(
    *,
    entry_id: str,
    condition: str = "Pulmonary embolism",
    condition_group: str | None = "thromboembolic",
    test_or_finding: str = "D-dimer",
    setting: list[str] | None = None,
    source_type: str = "primary_study",
    evidence_status: str = "sourced",
    source_citation: str | None = "Example citation",
    publication_year: int | None = 2024,
    age_group: str = "adult",
    lr_positive: float | None = 2.0,
    lr_negative: float | None = 0.2,
) -> LREntry:
    return LREntry.model_validate(
        {
            "id": entry_id,
            "condition": condition,
            "condition_group": condition_group,
            "test_or_finding": test_or_finding,
            "test_category": "lab",
            "setting": setting or ["ED"],
            "population": {"age_group": age_group},
            "applicability_flags": ["ED"],
            "source_type": source_type,
            "source_citation": source_citation,
            "publication_year": publication_year,
            "evidence_status": evidence_status,
            "lr_positive": lr_positive,
            "lr_negative": lr_negative,
            "status_updated_at": date(2026, 4, 1),
            "notes_for_model": "Test note",
        }
    )


def test_registry_loader_supports_jsonl_and_csv_seed_files() -> None:
    jsonl_entries = load_registry(DEFAULT_JSONL_PATH)
    csv_entries = load_registry(DEFAULT_CSV_PATH)
    assert len(jsonl_entries) == len(csv_entries) >= 10
    assert jsonl_entries[0].id == csv_entries[0].id
    assert all(entry.evidence_status == EvidenceStatus.SEED_ONLY for entry in jsonl_entries)


def test_curated_registry_supports_jsonl_and_csv_and_contains_sourced_rows() -> None:
    jsonl_entries = load_registry(DEFAULT_CURATED_JSONL_PATH)
    csv_entries = load_registry(DEFAULT_CURATED_CSV_PATH)
    assert len(jsonl_entries) == len(csv_entries) >= 5
    assert {entry.id for entry in jsonl_entries} == {entry.id for entry in csv_entries}
    assert all(entry.evidence_status == EvidenceStatus.SOURCED for entry in jsonl_entries)


def test_default_registry_bundle_merges_seed_and_curated_rows() -> None:
    bundle = load_default_registry_bundle()
    curated = load_curated_registry()
    seed = load_registry(DEFAULT_JSONL_PATH)
    assert len(bundle) == len(seed) + len(curated)
    assert any(entry.evidence_status == EvidenceStatus.SOURCED for entry in bundle)


def test_query_registry_returns_best_sourced_row_first() -> None:
    entries = [
        _entry(entry_id="pe_ddimer_primary", source_type="primary_study"),
        _entry(entry_id="pe_ddimer_meta", source_type="meta_analysis", publication_year=2025),
        _entry(
            entry_id="pe_ddimer_seed",
            source_type="systematic_review",
            source_citation=None,
            evidence_status="seed_only",
            publication_year=None,
            lr_positive=None,
            lr_negative=None,
        ),
    ]
    ranked = query_registry(entries, condition="Pulmonary embolism", test_or_finding="D-dimer", setting="ED")
    assert ranked[0].id == "pe_ddimer_meta"


def test_query_registry_filters_by_age_group_and_source_quality() -> None:
    entries = [
        _entry(entry_id="adult_guideline", source_type="guideline"),
        _entry(entry_id="peds_meta", age_group="pediatric", source_type="meta_analysis"),
        _entry(entry_id="adult_primary", source_type="primary_study"),
    ]
    ranked = query_registry(
        entries,
        condition="Pulmonary embolism",
        test_or_finding="D-dimer",
        age_group="adult",
        minimum_source_quality=SourceType.GUIDELINE,
    )
    assert [entry.id for entry in ranked] == ["adult_guideline"]


def test_resolver_prefers_exact_then_broader_then_related_then_estimated_then_seed() -> None:
    entries = [
        _entry(entry_id="exact_ed", source_type="guideline", setting=["ED"]),
        _entry(entry_id="broader_mixed", source_type="meta_analysis", setting=["ED", "inpatient"]),
        _entry(
            entry_id="related_syndrome",
            condition="Deep vein thrombosis",
            condition_group="thromboembolic",
            source_type="meta_analysis",
            test_or_finding="D-dimer",
        ),
        _entry(
            entry_id="estimated_row",
            source_type="expert_estimate",
            evidence_status="estimated",
            source_citation="Expert panel estimate",
        ),
        _entry(
            entry_id="seed_row",
            source_type="guideline",
            source_citation=None,
            evidence_status="seed_only",
            publication_year=None,
            lr_positive=None,
            lr_negative=None,
        ),
    ]
    resolution = resolve_registry_entry(
        entries,
        condition="Pulmonary embolism",
        test_or_finding="D-dimer",
        setting="ED",
        condition_group="thromboembolic",
    )
    assert resolution.selected is not None
    assert resolution.selected.id == "exact_ed"
    assert resolution.selection_reason == "exact_condition_exact_setting_sourced"


def test_resolver_warns_when_only_seed_rows_exist() -> None:
    entries = load_registry(DEFAULT_JSONL_PATH)
    resolution = resolve_registry_entry(
        entries,
        condition="Pulmonary embolism",
        test_or_finding="D-dimer",
        setting="ED",
        condition_group="thromboembolic",
    )
    assert resolution.selected is not None
    assert resolution.selected.id == "pe_d_dimer_adult_ed"
    assert resolution.message == "No sourced LR data available yet for this condition/test; using seed registry only."
    assert resolution.confidence_label == "seed_registry_only"


def test_default_bundle_prefers_curated_sourced_row_over_seed_only() -> None:
    entries = load_default_registry_bundle()
    resolution = resolve_registry_entry(
        entries,
        condition="Pulmonary embolism",
        test_or_finding="D-dimer",
        setting="ED",
        condition_group="thromboembolic",
    )
    assert resolution.selected is not None
    assert resolution.selected.id == "pe_d_dimer_age_adjusted_meta_2021"
    assert resolution.selection_reason == "exact_condition_broader_setting_sourced"


def test_query_registry_allows_alias_style_test_names() -> None:
    entries = load_default_registry_bundle()
    ranked = query_registry(entries, condition="Appendicitis", test_or_finding="ultrasound", age_group="pediatric")
    assert ranked
    assert ranked[0].id == "appendicitis_us_peds_nonradiologist_meta_2025"


def test_estimated_rows_are_labeled_lower_confidence() -> None:
    entry = _entry(
        entry_id="estimated_row",
        source_type="expert_estimate",
        evidence_status="estimated",
        source_citation="Expert estimate with explicit note",
    )
    assert entry.confidence_label == "lower_confidence_estimated"


def test_numeric_lr_fields_require_provenance() -> None:
    with pytest.raises(ValueError):
        _entry(
            entry_id="missing_provenance",
            source_citation=None,
        )


def test_duplicate_ids_fail_validation() -> None:
    entries = [_entry(entry_id="dup_row"), _entry(entry_id="dup_row")]
    with pytest.raises(RegistryValidationError):
        validate_registry_entries(entries)
