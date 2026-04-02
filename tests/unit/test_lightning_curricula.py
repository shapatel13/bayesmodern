from __future__ import annotations

from datasets.curricula import get_lightning_curriculum, list_lightning_curricula


def test_lightning_curricula_include_public_feedback_tracks() -> None:
    keys = {curriculum.key for curriculum in list_lightning_curricula()}
    assert {
        "broad_medical_feedback_lab",
        "diagnostic_reasoning_feedback_lab",
        "evidence_rare_feedback_lab",
        "clinical_safety_feedback_lab",
        "physician_audit_feedback_lab",
        "reviewed_cases_feedback_lab",
        "continuous_improvement_feedback_lab",
    } <= keys


def test_broad_feedback_curriculum_contains_core_hf_benchmarks() -> None:
    curriculum = get_lightning_curriculum("broad_medical_feedback_lab")
    dataset_keys = [component.dataset_key for component in curriculum.components]

    assert dataset_keys == ["medmcqa", "medqa", "pubmedqa", "findzebra"]
    assert curriculum.access_mode == "public_hf"


def test_continuous_improvement_curriculum_prefers_reviewed_cases_when_available() -> None:
    curriculum = get_lightning_curriculum("continuous_improvement_feedback_lab")
    dataset_keys = [component.dataset_key for component in curriculum.components]

    assert dataset_keys[-1] == "reviewed_cases"
    assert curriculum.access_mode == "hybrid"
