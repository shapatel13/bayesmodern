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
    "crushing chest pain": ("pressure_chest_pain", "Pressure-like chest pain"),
    "crushing pain": ("pressure_chest_pain", "Pressure-like chest pain"),
    "substernal": ("pressure_chest_pain", "Pressure-like chest pain"),
    "tightness": ("pressure_chest_pain", "Pressure-like chest pain"),
    "chest heaviness": ("pressure_chest_pain", "Pressure-like chest pain"),
    "troponin": ("troponin_positive", "Positive troponin"),
    "diaphor": ("diaphoresis", "Diaphoresis"),
    "sweating": ("diaphoresis", "Diaphoresis"),
    "radiating to arm": ("pain_radiation", "Radiation to arm or jaw"),
    "radiating to the arm": ("pain_radiation", "Radiation to arm or jaw"),
    "radiates to arm": ("pain_radiation", "Radiation to arm or jaw"),
    "radiating to jaw": ("pain_radiation", "Radiation to arm or jaw"),
    "radiating to the jaw": ("pain_radiation", "Radiation to arm or jaw"),
    "radiates to jaw": ("pain_radiation", "Radiation to arm or jaw"),
    "sputum": ("purulent_sputum", "Purulent sputum"),
    "melena": ("melena", "Melena"),
    "black tarry": ("melena", "Melena"),
    "tarry stool": ("melena", "Melena"),
    "symptomatic anemia": ("symptomatic_anemia", "Symptomatic anemia"),
    "gi bleed": ("active_gi_bleeding", "Active gastrointestinal bleeding"),
    "gastrointestinal bleed": ("active_gi_bleeding", "Active gastrointestinal bleeding"),
    "hematemesis": ("active_gi_bleeding", "Active gastrointestinal bleeding"),
}

KEYWORD_MEDICATIONS: dict[str, str] = {
    "warfarin": "warfarin",
    "heparin": "heparin",
    "apixaban": "apixaban",
    "rivaroxaban": "rivaroxaban",
    "dabigatran": "dabigatran",
    "enoxaparin": "enoxaparin",
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
    "melena": "melena",
    "symptomatic anemia": "symptomatic anemia",
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
ANTICOAGULANT_MEDICATIONS = {"warfarin", "heparin", "apixaban", "rivaroxaban", "dabigatran", "enoxaparin"}
NEGATION_PREFIXES = ("no ", "without ", "denies ", "denied ", "not ")
NEGATED_FINDING_ALIASES: dict[str, tuple[str, str]] = {
    "afebrile": ("fever", "Fever"),
}


def _append_if_missing(findings: list[ClinicalFinding], key: str, label: str, *, source_type: str = "user_supplied") -> None:
    if any(item.key == key for item in findings):
        return
    findings.append(ClinicalFinding(key=key, label=label, present=True, source_type=source_type))


def _replace_or_append_finding(
    findings_by_key: dict[str, ClinicalFinding],
    *,
    key: str,
    label: str,
    present: bool,
    source_type: str = "user_supplied",
) -> None:
    existing = findings_by_key.get(key)
    if existing is None or existing.present is False:
        findings_by_key[key] = ClinicalFinding(key=key, label=label, present=present, source_type=source_type)


def _keyword_signal_state(lowered: str, needle: str) -> bool | None:
    states: list[bool] = []
    search_start = 0
    while True:
        index = lowered.find(needle, search_start)
        if index == -1:
            break
        prefix = lowered[max(0, index - 24):index]
        negated = any(prefix.endswith(cue) for cue in NEGATION_PREFIXES)
        states.append(not negated)
        search_start = index + len(needle)
    if not states:
        return None
    return True if any(states) else False


def _keyword_extract_context(case_id: str, note_text: str) -> ClinicalDecisionContext:
    lowered = note_text.lower()
    findings_by_key: dict[str, ClinicalFinding] = {}
    for needle, (key, label) in KEYWORD_FINDINGS.items():
        state = _keyword_signal_state(lowered, needle)
        if state is None:
            continue
        _replace_or_append_finding(findings_by_key, key=key, label=label, present=state)
    for needle, (key, label) in NEGATED_FINDING_ALIASES.items():
        if needle in lowered:
            findings_by_key[key] = ClinicalFinding(key=key, label=label, present=False)
    findings = list(findings_by_key.values())
    medications = sorted({canonical for needle, canonical in KEYWORD_MEDICATIONS.items() if needle in lowered})
    adverse_events = sorted({canonical for needle, canonical in KEYWORD_ADVERSE_EVENTS.items() if needle in lowered})
    if set(medications) & ANTICOAGULANT_MEDICATIONS:
        _append_if_missing(findings, "anticoagulated", "On anticoagulation")
    if "gi bleed" in adverse_events:
        _append_if_missing(findings, "active_gi_bleeding", "Active gastrointestinal bleeding")
    if "melena" in adverse_events:
        _append_if_missing(findings, "active_gi_bleeding", "Active gastrointestinal bleeding")
        _append_if_missing(findings, "melena", "Melena")
    if "symptomatic anemia" in adverse_events:
        _append_if_missing(findings, "symptomatic_anemia", "Symptomatic anemia")
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
