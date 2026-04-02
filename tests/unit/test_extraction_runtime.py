from core.models import ClinicalDecisionContext, ClinicalFinding
from llm.extraction import extract_context_from_text
from utils.config import Settings


def test_keyword_extraction_without_live_llm() -> None:
    context = extract_context_from_text(
        case_id="extract-1",
        note_text="Pleuritic chest pain with tachycardia and hypoxemia after lisinopril-associated angioedema.",
        settings=Settings(_env_file=None, allow_live_llm=False),
    )
    keys = {finding.key for finding in context.findings}
    assert {"pleuritic_chest_pain", "tachycardia", "hypoxemia"} <= keys
    assert "lisinopril" in context.medications
    assert "angioedema" in context.adverse_events


def test_keyword_extraction_recognizes_ischemic_chest_pain_language() -> None:
    context = extract_context_from_text(
        case_id="extract-ischemic-1",
        note_text="Crushing substernal chest pain with diaphoresis and pain radiating to the jaw.",
        settings=Settings(_env_file=None, allow_live_llm=False),
    )
    keys = {finding.key for finding in context.findings}
    assert {"pressure_chest_pain", "diaphoresis", "pain_radiation"} <= keys


def test_keyword_extraction_recognizes_anticoagulated_bleeding_language() -> None:
    context = extract_context_from_text(
        case_id="extract-bleed-1",
        note_text="Patient on warfarin with melena and symptomatic anemia after recent dose escalation.",
        settings=Settings(_env_file=None, allow_live_llm=False),
    )
    keys = {finding.key for finding in context.findings}
    assert {"anticoagulated", "active_gi_bleeding", "melena", "symptomatic_anemia"} <= keys
    assert "warfarin" in context.medications


def test_extraction_falls_back_when_openai_parse_raises(monkeypatch) -> None:
    from llm import extraction

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(extraction, "extract_context_with_openai", boom)
    context = extract_context_from_text(
        case_id="extract-2",
        note_text="Orthopnea with crackles and edema.",
        settings=Settings(
            _env_file=None,
            allow_live_llm=True,
            default_model_provider="openai",
            openai_api_key="test-key",
        ),
    )
    assert isinstance(context, ClinicalDecisionContext)
    assert any(finding.key == "orthopnea" for finding in context.findings)


def test_extraction_merges_openai_and_keyword_findings(monkeypatch) -> None:
    from llm import extraction

    def fake_parse(*args, **kwargs):
        return ClinicalDecisionContext(
            case_id="extract-3",
            specialty="pulmonary",
            findings=[ClinicalFinding(key="fever", label="Fever", present=True, source_type="llm_inferred")],
            medications=["warfarin"],
            adverse_events=["gi bleed"],
        )

    monkeypatch.setattr(extraction, "extract_context_with_openai", fake_parse)
    context = extract_context_from_text(
        case_id="extract-3",
        note_text="Pleuritic chest pain with tachycardia.",
        settings=Settings(
            _env_file=None,
            allow_live_llm=True,
            default_model_provider="openai",
            openai_api_key="test-key",
        ),
    )
    keys = {finding.key for finding in context.findings}
    assert {"fever", "pleuritic_chest_pain", "tachycardia"} <= keys
    assert context.specialty == "pulmonary"
    assert "warfarin" in context.medications
    assert "gi bleed" in context.adverse_events
