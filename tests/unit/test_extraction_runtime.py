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


def test_keyword_extraction_handles_simple_negation() -> None:
    context = extract_context_from_text(
        case_id="extract-neg-1",
        note_text="Pleuritic chest pain with tachycardia and hypoxemia, no fever.",
        settings=Settings(_env_file=None, allow_live_llm=False),
    )
    findings = {finding.key: finding.present for finding in context.findings}
    assert findings["pleuritic_chest_pain"] is True
    assert findings["tachycardia"] is True
    assert findings["hypoxemia"] is True
    assert findings["fever"] is False


def test_keyword_extraction_parses_numeric_vitals_and_labs() -> None:
    context = extract_context_from_text(
        case_id="extract-numeric-1",
        note_text=(
            "68-year-old pregnant patient with BP 82/50, HR 128, SpO2 86%, temp 38.6 C, "
            "hemoglobin 6.8, INR 4.2, creatinine 2.6, lactate 4.8, BNP 1400."
        ),
        settings=Settings(_env_file=None, allow_live_llm=False),
    )
    findings = {finding.key: finding for finding in context.findings}

    assert context.age_years == 68
    assert context.pregnant is True
    assert context.hemodynamic_instability is True
    assert context.critical_values_present is True
    assert context.renal_impairment is True
    assert {"tachycardia", "hypoxemia", "fever", "low_hemoglobin", "severe_anemia", "supratherapeutic_inr", "creatinine_elevated", "elevated_lactate", "bnp_elevated"} <= set(findings)
    assert findings["severe_anemia"].value == 6.8


def test_keyword_extraction_detects_completed_tests_from_note_text() -> None:
    context = extract_context_from_text(
        case_id="extract-tests-1",
        note_text=(
            "ECG showed anterior changes. Troponin 0.12. BNP 900. Chest x-ray with edema. "
            "INR 3.8. Type and screen sent."
        ),
        settings=Settings(_env_file=None, allow_live_llm=False),
    )

    assert {"ecg", "hs_troponin", "bnp", "cxr", "pt_inr", "type_screen"} <= set(context.completed_tests)


def test_keyword_extraction_turns_numeric_troponin_and_ecg_language_into_findings() -> None:
    context = extract_context_from_text(
        case_id="extract-acs-result-1",
        note_text="ECG showed ischemic changes and troponin 0.12 in a patient with chest pressure.",
        settings=Settings(_env_file=None, allow_live_llm=False),
    )
    findings = {finding.key for finding in context.findings}

    assert {"ecg_ischemia", "troponin_positive", "pressure_chest_pain"} <= findings


def test_keyword_extraction_does_not_confuse_blood_pressure_with_chest_pressure() -> None:
    context = extract_context_from_text(
        case_id="extract-pressure-guard-1",
        note_text="Blood pressure 88/54 with melena and dyspnea but no chest pain.",
        settings=Settings(_env_file=None, allow_live_llm=False),
    )
    findings = {finding.key for finding in context.findings}

    assert "hemodynamic_instability" in findings
    assert "pressure_chest_pain" not in findings


def test_keyword_extraction_parses_diuretic_and_low_ef_clues() -> None:
    context = extract_context_from_text(
        case_id="extract-hf-1",
        note_text="Patient on torsemide with EF 30% and reduced LV systolic function.",
        settings=Settings(_env_file=None, allow_live_llm=False),
    )
    findings = {finding.key for finding in context.findings}

    assert "torsemide" in context.medications
    assert "reduced_ef" in findings


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
