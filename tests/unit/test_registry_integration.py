from __future__ import annotations

from core.models import ClinicalDecisionContext
from core.next_best_test import NextBestTestEngine
from evidence.cost_catalog import default_test_catalog
from evidence.disease_profiles import default_hypotheses
from evidence.literature_registry import is_known_provenance_ref
from evidence.registry_loader import LREntry


def _build_registry_entry(**overrides) -> LREntry:
    payload = {
        "id": "pe_d_dimer_curated",
        "condition": "Pulmonary embolism",
        "condition_group": "thromboembolic",
        "test_or_finding": "D-dimer",
        "test_category": "lab",
        "comparator_or_threshold": "age-adjusted threshold",
        "target_outcome": "rule out PE",
        "lr_positive": 3.4,
        "lr_negative": 0.12,
        "sensitivity": None,
        "specificity": None,
        "ci_95_lower_lr_positive": 2.8,
        "ci_95_upper_lr_positive": 4.1,
        "ci_95_lower_lr_negative": 0.08,
        "ci_95_upper_lr_negative": 0.18,
        "pretest_anchor_min": 0.05,
        "pretest_anchor_max": 0.4,
        "setting": ["ED"],
        "population": {
            "age_group": "adult",
            "pregnancy": False,
            "immunocompromised": None,
            "renal_impairment_relevant": False,
            "icu_population": False,
            "notes": "Curated test entry for registry integration tests.",
        },
        "applicability_flags": ["ED", "low_cost"],
        "harms_or_constraints": ["Assay-specific threshold interpretation remains necessary."],
        "direct_cost_usd": 55,
        "downstream_cascade_cost_usd": 250,
        "turnaround_time_hours": 1,
        "source_type": "meta_analysis",
        "source_citation": "Curated meta-analysis citation.",
        "source_url": "https://example.org/pe-d-dimer",
        "pmid_or_doi": "10.1000/pe.dimer",
        "publication_year": 2024,
        "evidence_status": "sourced",
        "notes_for_model": "Use within a structured low/intermediate-risk PE pathway.",
        "preference_weight_hint": 0.85,
        "status_updated_at": "2026-04-01",
    }
    payload.update(overrides)
    return LREntry.model_validate(payload)


def test_sourced_registry_row_overrides_starter_test_lr() -> None:
    curated_entry = _build_registry_entry()
    catalog = default_test_catalog(registry_entries=[curated_entry])
    d_dimer = next(test for test in catalog if test.slug == "d_dimer")

    assert d_dimer.source_type == "evidence_registry"
    assert d_dimer.diagnosis_lrs["pe"].positive_lr == 3.4
    assert d_dimer.diagnosis_lrs["pe"].negative_lr == 0.12
    assert d_dimer.direct_cost == 55
    assert d_dimer.downstream_cost == 250
    assert "registry:pe_d_dimer_curated" in d_dimer.provenance_refs


def test_seed_only_registry_row_keeps_starter_math_and_surfaces_notice() -> None:
    seed_entry = _build_registry_entry(
        id="pe_d_dimer_seed",
        lr_positive=None,
        lr_negative=None,
        ci_95_lower_lr_positive=None,
        ci_95_upper_lr_positive=None,
        ci_95_lower_lr_negative=None,
        ci_95_upper_lr_negative=None,
        source_type="systematic_review",
        source_citation=None,
        source_url=None,
        pmid_or_doi=None,
        publication_year=None,
        evidence_status="seed_only",
    )
    d_dimer = next(test for test in default_test_catalog(registry_entries=[seed_entry]) if test.slug == "d_dimer")
    recommendation = NextBestTestEngine().rank(
        {"pe": 0.25},
        [d_dimer],
        ClinicalDecisionContext(case_id="registry-seed-only"),
    )[0]

    assert d_dimer.source_type == "hard_coded"
    assert d_dimer.diagnosis_lrs["pe"].positive_lr == 2.2
    assert recommendation.rationale.startswith(
        "No sourced LR data available yet for this condition/test; using seed registry only."
    )
    assert "registry:pe_d_dimer_seed" in recommendation.provenance_badges


def test_sourced_registry_row_overrides_hypothesis_evidence() -> None:
    acs_entry = _build_registry_entry(
        id="acs_hs_trop_curated",
        condition="Acute coronary syndrome",
        condition_group="cardiovascular",
        test_or_finding="High-sensitivity troponin",
        target_outcome="support or reduce likelihood of MI/ACS",
        lr_positive=7.2,
        lr_negative=0.18,
        ci_95_lower_lr_positive=5.8,
        ci_95_upper_lr_positive=8.4,
        ci_95_lower_lr_negative=0.12,
        ci_95_upper_lr_negative=0.25,
        direct_cost_usd=45,
        downstream_cascade_cost_usd=320,
        source_url="https://example.org/acs-hs-trop",
        pmid_or_doi="10.1000/acs.trop",
    )
    hypotheses = default_hypotheses(registry_entries=[acs_entry])
    acs = next(hypothesis for hypothesis in hypotheses if hypothesis.slug == "acs")
    troponin = next(item for item in acs.supporting_findings if item.finding_key == "troponin_positive")

    assert troponin.source_type == "evidence_registry"
    assert troponin.lr.positive_lr == 7.2
    assert troponin.lr.negative_lr == 0.18
    assert "registry:acs_hs_trop_curated" in troponin.provenance_refs


def test_seed_registry_references_are_treated_as_known_provenance() -> None:
    assert is_known_provenance_ref("registry:pe_d_dimer_adult_ed") is True
