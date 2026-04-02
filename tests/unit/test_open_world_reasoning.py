from __future__ import annotations

from agent import orchestrator as orchestrator_module
from agent.orchestrator import PRIORIXOrchestrator
from core.models import ClinicalDecisionContext
from llm.extraction import extract_context_from_text
from llm.open_world_reasoning import (
    OpenWorldEvidenceClue,
    OpenWorldHypothesisProposal,
    OpenWorldReasoningPlan,
    OpenWorldTestProposal,
)
from utils.config import Settings


def test_open_world_reasoning_can_add_unseen_diagnosis_and_test(monkeypatch) -> None:
    plan = OpenWorldReasoningPlan(
        specialty="neurology_infectious",
        hypotheses=[
            OpenWorldHypothesisProposal(
                slug="bacterial_meningitis",
                name="Bacterial Meningitis",
                category="neurologic_infectious",
                dangerous=True,
                urgency_weight=1.0,
                prior_hint=0.18,
                rationale="Fever, neck stiffness, and photophobia support meningitis.",
                evidence_clues=[
                    OpenWorldEvidenceClue(label="Fever", finding_key="fever", strength="moderate"),
                    OpenWorldEvidenceClue(label="Neck stiffness", finding_key="neck_stiffness", strength="strong"),
                    OpenWorldEvidenceClue(label="Photophobia", finding_key="photophobia", strength="strong"),
                ],
            )
        ],
        suggested_tests=[
            OpenWorldTestProposal(
                slug="lumbar_puncture",
                name="Lumbar puncture",
                target_diagnoses=["bacterial_meningitis"],
                rationale="Highest-yield diagnostic test when meningitis is being considered.",
                rule_in_strength="strong",
                rule_out_strength="moderate",
                actionability=1.0,
                urgency_modifier=1.8,
                direct_cost_hint_usd=280,
                downstream_cost_hint_usd=120,
                invasiveness=0.35,
            )
        ],
    )
    monkeypatch.setattr(orchestrator_module, "generate_open_world_reasoning_plan", lambda *args, **kwargs: plan)
    orchestrator = PRIORIXOrchestrator(
        settings=Settings(_env_file=None, allow_live_llm=False, default_model_provider="offline")
    )

    report = orchestrator.analyze_context(
        ClinicalDecisionContext(
            case_id="open-world-1",
            symptoms_free_text="Fever, neck stiffness, and photophobia.",
        )
    )

    assert report.context.specialty == "neurology_infectious"
    assert any(finding.key == "neck_stiffness" for finding in report.context.findings)
    assert report.differential.ranked[0].slug == "bacterial_meningitis"
    assert report.next_best_tests[0].slug == "lumbar_puncture"
    assert report.reasoning_runtime.mode == "hybrid_open_world"
    assert report.reasoning_runtime.open_world_triggered is True
    assert report.reasoning_runtime.open_world_hypothesis_count == 1


def test_open_world_reasoning_can_enrich_existing_hypothesis_without_duplicate(monkeypatch) -> None:
    plan = OpenWorldReasoningPlan(
        specialty="cardiology",
        hypotheses=[
            OpenWorldHypothesisProposal(
                slug="acs",
                name="Acute Coronary Syndrome",
                category="cardiovascular",
                dangerous=True,
                urgency_weight=0.95,
                prior_hint=0.2,
                rationale="Classic ischemic symptom cluster.",
                evidence_clues=[
                    OpenWorldEvidenceClue(label="Pressure-like chest pain", finding_key="pressure_chest_pain", strength="strong"),
                    OpenWorldEvidenceClue(label="Diaphoresis", finding_key="diaphoresis", strength="moderate"),
                    OpenWorldEvidenceClue(label="Radiation to arm or jaw", finding_key="pain_radiation", strength="strong"),
                ],
            )
        ],
        suggested_tests=[],
    )
    monkeypatch.setattr(orchestrator_module, "generate_open_world_reasoning_plan", lambda *args, **kwargs: plan)
    orchestrator = PRIORIXOrchestrator(
        settings=Settings(_env_file=None, allow_live_llm=False, default_model_provider="offline")
    )

    report = orchestrator.analyze_context(
        ClinicalDecisionContext(
            case_id="open-world-2",
            symptoms_free_text="Crushing substernal chest pressure radiating to the jaw with diaphoresis.",
        )
    )
    slugs = [entry.slug for entry in report.differential.ranked]

    assert slugs.count("acs") == 1
    assert report.differential.ranked[0].slug == "acs"
    assert report.next_best_tests[0].slug == "ecg"
    assert report.reasoning_runtime.mode == "hybrid_open_world"


def test_open_world_reasoning_failure_falls_back_to_existing_engine(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("generation unavailable")

    monkeypatch.setattr(orchestrator_module, "generate_open_world_reasoning_plan", boom)
    orchestrator = PRIORIXOrchestrator(
        settings=Settings(_env_file=None, allow_live_llm=False, default_model_provider="offline")
    )

    report = orchestrator.analyze_context(
        ClinicalDecisionContext(
            case_id="open-world-3",
            findings=[],
            symptoms_free_text="Crushing chest pain with diaphoresis.",
        )
    )

    assert report.differential.ranked
    assert report.reasoning_runtime.mode == "curated_only"
    assert report.reasoning_runtime.open_world_triggered is False


def test_confident_curated_case_skips_open_world_generation(monkeypatch) -> None:
    calls = {"count": 0}
    extraction_settings = Settings(_env_file=None, allow_live_llm=False, default_model_provider="offline")

    def fake_generate(*args, **kwargs):
        calls["count"] += 1
        raise AssertionError("open-world generation should not be called for confident curated cases")

    monkeypatch.setattr(orchestrator_module, "generate_open_world_reasoning_plan", fake_generate)
    orchestrator = PRIORIXOrchestrator(
        settings=Settings(_env_file=None, allow_live_llm=True, default_model_provider="openai", openai_api_key="test-key")
    )
    context = extract_context_from_text(
        case_id="open-world-4",
        note_text="Pleuritic chest pain with tachycardia and hypoxemia, no fever.",
        settings=extraction_settings,
    )

    report = orchestrator.analyze_context(context)

    assert calls["count"] == 0
    assert report.differential.ranked[0].slug == "pe"
    assert report.reasoning_runtime.mode == "curated_only"
    assert report.reasoning_runtime.open_world_triggered is False


def test_open_world_disabled_by_settings_skips_generation(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_generate(*args, **kwargs):
        calls["count"] += 1
        raise AssertionError("open-world generation should not be called when disabled in settings")

    monkeypatch.setattr(orchestrator_module, "generate_open_world_reasoning_plan", fake_generate)
    settings = Settings(
        _env_file=None,
        allow_live_llm=False,
        default_model_provider="offline",
        open_world_reasoning_enabled=False,
    )
    context = extract_context_from_text(
        case_id="open-world-5",
        note_text="Fever, neck stiffness, and photophobia.",
        settings=settings,
    )
    orchestrator = PRIORIXOrchestrator(settings=settings)

    report = orchestrator.analyze_context(context)

    assert calls["count"] == 0
    assert report.reasoning_runtime.mode == "curated_only"
    assert report.reasoning_runtime.open_world_triggered is False
    assert report.reasoning_runtime.gate_reason == "Open-world expansion disabled by configuration."
