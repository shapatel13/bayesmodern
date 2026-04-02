from __future__ import annotations

from core.models import ClinicalDecisionContext, ClinicalFinding
from llm.openai_extraction import extract_context_with_openai
from utils.config import Settings
from utils.logging import get_logger


KEYWORD_FINDINGS: dict[str, tuple[str, str]] = {
    "pleuritic": ("pleuritic_chest_pain", "Pleuritic chest pain"),
    "tachycard": ("tachycardia", "Tachycardia"),
    "hypox": ("hypoxemia", "Hypoxemia"),
    "fever": ("fever", "Fever"),
    "crackle": ("crackles", "Crackles"),
    "orthopnea": ("orthopnea", "Orthopnea"),
    "edema": ("leg_edema", "Leg edema"),
    "pressure": ("pressure_chest_pain", "Pressure-like chest pain"),
    "troponin": ("troponin_positive", "Positive troponin"),
    "sputum": ("purulent_sputum", "Purulent sputum"),
}

KEYWORD_MEDICATIONS: dict[str, str] = {
    "warfarin": "warfarin",
    "heparin": "heparin",
    "lisinopril": "lisinopril",
    "vancomycin": "vancomycin",
    "gentamicin": "gentamicin",
    "metformin": "metformin",
    "insulin": "insulin",
    "amiodarone": "amiodarone",
    "ibuprofen": "ibuprofen",
    "naproxen": "naproxen",
}

KEYWORD_ADVERSE_EVENTS: dict[str, str] = {
    "angioedema": "angioedema",
    "gi bleed": "gi bleed",
    "gastrointestinal bleed": "gi bleed",
    "hematemesis": "gi bleed",
    "melena": "gi bleed",
    "aki": "acute kidney injury",
    "acute kidney injury": "acute kidney injury",
    "renal failure": "acute kidney injury",
    "hypoglycemia": "hypoglycemia",
    "hyperkalemia": "hyperkalemia",
    "rash": "rash",
    "thrombocytopenia": "thrombocytopenia",
    "bleeding": "bleeding",
}

logger = get_logger(__name__)


def _keyword_extract_context(case_id: str, note_text: str) -> ClinicalDecisionContext:
    lowered = note_text.lower()
    findings = [
        ClinicalFinding(key=key, label=label, present=True)
        for needle, (key, label) in KEYWORD_FINDINGS.items()
        if needle in lowered
    ]
    medications = sorted({canonical for needle, canonical in KEYWORD_MEDICATIONS.items() if needle in lowered})
    adverse_events = sorted({canonical for needle, canonical in KEYWORD_ADVERSE_EVENTS.items() if needle in lowered})
    return ClinicalDecisionContext(
        case_id=case_id,
        symptoms_free_text=note_text,
        findings=findings,
        medications=medications,
        adverse_events=adverse_events,
        hemodynamic_instability=any(token in lowered for token in ("shock", "hypotension", "unstable")),
        critical_values_present=any(token in lowered for token in ("lactate", "critical", "severe hypoxia")),
    )


def _merge_contexts(primary: ClinicalDecisionContext, fallback: ClinicalDecisionContext) -> ClinicalDecisionContext:
    merged_findings = {finding.key: finding for finding in fallback.findings}
    for finding in primary.findings:
        merged_findings[finding.key] = finding
    return ClinicalDecisionContext(
        case_id=primary.case_id,
        specialty=primary.specialty or fallback.specialty,
        findings=list(merged_findings.values()),
        completed_tests=fallback.completed_tests,
        comorbidities=fallback.comorbidities,
        medications=sorted({*fallback.medications, *primary.medications}),
        adverse_events=sorted({*fallback.adverse_events, *primary.adverse_events}),
        symptoms_free_text=primary.symptoms_free_text or fallback.symptoms_free_text,
        age_years=primary.age_years or fallback.age_years,
        pregnant=primary.pregnant or fallback.pregnant,
        renal_impairment=primary.renal_impairment or fallback.renal_impairment,
        hemodynamic_instability=primary.hemodynamic_instability or fallback.hemodynamic_instability,
        critical_values_present=primary.critical_values_present or fallback.critical_values_present,
        safety_mode=primary.safety_mode or fallback.safety_mode,
    )


def extract_context_from_text(
    case_id: str,
    note_text: str,
    settings: Settings | None = None,
    *,
    prompt_template: str | None = None,
) -> ClinicalDecisionContext:
    keyword_context = _keyword_extract_context(case_id=case_id, note_text=note_text)
    if settings is None:
        return keyword_context
    if not settings.allow_live_llm or settings.default_model_provider != "openai" or not settings.openai_api_key:
        return keyword_context
    try:
        openai_context = extract_context_with_openai(
            case_id=case_id,
            note_text=note_text,
            settings=settings,
            prompt_template=prompt_template,
        )
        return _merge_contexts(primary=openai_context, fallback=keyword_context)
    except Exception as exc:
        logger.warning("OpenAI parsing failed; falling back to keyword extraction: %s", exc)
        return keyword_context
