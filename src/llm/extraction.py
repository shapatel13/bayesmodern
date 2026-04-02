from __future__ import annotations

import re

from core.models import ClinicalDecisionContext, ClinicalFinding
from llm.openai_extraction import extract_context_with_openai
from utils.config import Settings
from utils.logging import get_logger


KEYWORD_FINDINGS: dict[str, tuple[str, str]] = {
    "pleuritic": ("pleuritic_chest_pain", "Pleuritic chest pain"),
    "tachycard": ("tachycardia", "Tachycardia"),
    "hypox": ("hypoxemia", "Hypoxemia"),
    "fever": ("fever", "Fever"),
    "hypotension": ("hemodynamic_instability", "Hemodynamic instability"),
    "shock": ("hemodynamic_instability", "Hemodynamic instability"),
    "unstable": ("hemodynamic_instability", "Hemodynamic instability"),
    "crackle": ("crackles", "Crackles"),
    "orthopnea": ("orthopnea", "Orthopnea"),
    "edema": ("leg_edema", "Leg edema"),
    "jvp": ("elevated_jvp", "Elevated JVP"),
    "jugular venous distension": ("elevated_jvp", "Elevated JVP"),
    "plethoric ivc": ("elevated_jvp", "Elevated JVP"),
    "cool extremit": ("cool_extremities", "Cool extremities"),
    "warm extremit": ("warm_extremities", "Warm extremities"),
    "oliguria": ("oliguria", "Oliguria"),
    "reduced ef": ("reduced_ef", "Reduced ejection fraction"),
    "low ejection fraction": ("reduced_ef", "Reduced ejection fraction"),
    "rv strain": ("rv_strain", "RV strain"),
    "ischemic changes": ("ecg_ischemia", "Ischemic ECG changes"),
    "st elevation": ("ecg_ischemia", "Ischemic ECG changes"),
    "st depression": ("ecg_ischemia", "Ischemic ECG changes"),
    "t-wave inversion": ("ecg_ischemia", "Ischemic ECG changes"),
    "elevated troponin": ("troponin_positive", "Positive troponin"),
    "positive troponin": ("troponin_positive", "Positive troponin"),
    "elevated bnp": ("bnp_elevated", "Elevated BNP/NT-proBNP"),
    "high bnp": ("bnp_elevated", "Elevated BNP/NT-proBNP"),
    "elevated lactate": ("elevated_lactate", "Elevated lactate"),
    "supratherapeutic inr": ("supratherapeutic_inr", "Supratherapeutic INR"),
    "high inr": ("supratherapeutic_inr", "Supratherapeutic INR"),
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
COMPLETED_TEST_PATTERNS: dict[str, tuple[str, ...]] = {
    "ecg": ("ecg", "ekg", "electrocardiogram"),
    "hs_troponin": ("troponin", "hs troponin", "high-sensitivity troponin"),
    "d_dimer": ("d-dimer", "d dimer", "ddimer"),
    "cta_pe": ("ct pulmonary angiography", "ctpa", "cta chest", "cta pe", "ct angiography"),
    "cxr": ("cxr", "chest xray", "chest x-ray", "chest radiograph"),
    "bnp": ("bnp", "nt-probnp"),
    "cbc": ("cbc", "complete blood count", "hemoglobin", "hgb", "hb "),
    "pt_inr": ("inr", "pt/inr", "pt inr", "protime"),
    "type_screen": ("type and screen", "type & screen"),
    "upper_endoscopy": ("endoscopy", "egd", "upper endoscopy"),
    "bedside_echo": ("bedside echo", "echocardiogram", "echo showed", "tte"),
    "plr_lvot_vti": ("passive leg raise", "plr", "lvot vti"),
    "repeat_hemoglobin": ("repeat hemoglobin", "repeat hgb", "serial hemoglobin"),
}

AGE_PATTERN = re.compile(r"\b(\d{1,3})[- ]year[- ]old\b")
BP_PATTERN = re.compile(r"\b(?:bp|blood pressure)\s*(?:is|was|of|:)?\s*(\d{2,3})\s*/\s*(\d{2,3})\b")
HR_PATTERN = re.compile(r"\b(?:hr|heart rate|pulse)\s*(?:is|was|of|:)?\s*(\d{2,3})\b")
SPO2_PATTERN = re.compile(r"\b(?:spo2|sat(?:s)?|oxygen saturation)\s*(?:is|was|of|:)?\s*(\d{2,3})\s*%?")
TEMP_PATTERN = re.compile(r"\b(?:temp|temperature)\s*(?:is|was|of|:)?\s*(\d{2,3}(?:\.\d+)?)\s*([cf])?\b")
HGB_PATTERN = re.compile(r"\b(?:hgb|hemoglobin|hb)\s*(?:is|was|of|:)?\s*(\d{1,2}(?:\.\d+)?)\b")
INR_PATTERN = re.compile(r"\b(?:inr)\s*(?:is|was|of|:)?\s*(\d{1,2}(?:\.\d+)?)\b")
CREATININE_PATTERN = re.compile(r"\b(?:creatinine|cr)\s*(?:is|was|of|:)?\s*(\d{1,2}(?:\.\d+)?)\b")
LACTATE_PATTERN = re.compile(r"\b(?:lactate)\s*(?:is|was|of|:)?\s*(\d{1,2}(?:\.\d+)?)\b")
SODIUM_PATTERN = re.compile(r"\b(?:sodium|na)\s*(?:is|was|of|:)?\s*(\d{2,3}(?:\.\d+)?)\b")
BNP_PATTERN = re.compile(r"\b(?:bnp|nt-probnp)\s*(?:is|was|of|:)?\s*(\d{2,6}(?:\.\d+)?)\b")
TROPONIN_PATTERN = re.compile(r"\b(?:troponin|hs troponin|high-sensitivity troponin)\s*(?:is|was|of|:)?\s*(\d+(?:\.\d+)?)\b")


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


def _upsert_numeric_finding(
    findings_by_key: dict[str, ClinicalFinding],
    *,
    key: str,
    label: str,
    value: float,
    units: str | None,
    note: str,
    present: bool = True,
    confidence: float = 0.95,
    source_type: str = "hard_coded",
) -> None:
    findings_by_key[key] = ClinicalFinding(
        key=key,
        label=label,
        present=present,
        value=value,
        units=units,
        note=note,
        confidence=confidence,
        source_type=source_type,
    )


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


def _extract_completed_tests(lowered: str) -> list[str]:
    completed: list[str] = []
    for slug, needles in COMPLETED_TEST_PATTERNS.items():
        if any(needle in lowered for needle in needles):
            completed.append(slug)
    return sorted(set(completed))


def _extract_age(note_text: str) -> int | None:
    match = AGE_PATTERN.search(note_text.lower())
    if not match:
        return None
    age = int(match.group(1))
    return age if 0 <= age <= 120 else None


def _extract_numeric_findings(note_text: str) -> tuple[dict[str, ClinicalFinding], dict[str, float], dict[str, str]]:
    lowered = note_text.lower()
    findings: dict[str, ClinicalFinding] = {}
    numeric_values: dict[str, float] = {}
    units: dict[str, str] = {}

    if match := BP_PATTERN.search(lowered):
        systolic = float(match.group(1))
        diastolic = float(match.group(2))
        numeric_values["systolic_bp"] = systolic
        numeric_values["diastolic_bp"] = diastolic
        units["systolic_bp"] = "mmHg"
        units["diastolic_bp"] = "mmHg"
        if systolic < 90:
            _upsert_numeric_finding(
                findings,
                key="hemodynamic_instability",
                label="Hemodynamic instability",
                value=systolic,
                units="mmHg",
                note=f"Systolic blood pressure {systolic:.0f} mmHg",
            )

    if match := HR_PATTERN.search(lowered):
        heart_rate = float(match.group(1))
        numeric_values["heart_rate"] = heart_rate
        units["heart_rate"] = "bpm"
        if heart_rate >= 100:
            _upsert_numeric_finding(
                findings,
                key="tachycardia",
                label="Tachycardia",
                value=heart_rate,
                units="bpm",
                note=f"Heart rate {heart_rate:.0f} bpm",
            )

    if match := SPO2_PATTERN.search(lowered):
        spo2 = float(match.group(1))
        numeric_values["spo2"] = spo2
        units["spo2"] = "%"
        if spo2 < 92:
            _upsert_numeric_finding(
                findings,
                key="hypoxemia",
                label="Hypoxemia",
                value=spo2,
                units="%",
                note=f"Oxygen saturation {spo2:.0f}%",
            )
        if spo2 < 88:
            _upsert_numeric_finding(
                findings,
                key="critical_values_present",
                label="Critical oxygenation value",
                value=spo2,
                units="%",
                note=f"Severe hypoxemia with oxygen saturation {spo2:.0f}%",
            )

    if match := TEMP_PATTERN.search(lowered):
        temp_value = float(match.group(1))
        temp_unit = match.group(2) or ("f" if temp_value > 45 else "c")
        numeric_values["temperature"] = temp_value
        units["temperature"] = temp_unit.upper()
        fever = temp_value >= 100.4 if temp_unit == "f" else temp_value >= 38.0
        if fever:
            _upsert_numeric_finding(
                findings,
                key="fever",
                label="Fever",
                value=temp_value,
                units=temp_unit.upper(),
                note=f"Temperature {temp_value:.1f} {temp_unit.upper()}",
            )

    if match := HGB_PATTERN.search(lowered):
        hemoglobin = float(match.group(1))
        numeric_values["hemoglobin"] = hemoglobin
        units["hemoglobin"] = "g/dL"
        if hemoglobin < 10:
            _upsert_numeric_finding(
                findings,
                key="low_hemoglobin",
                label="Low hemoglobin",
                value=hemoglobin,
                units="g/dL",
                note=f"Hemoglobin {hemoglobin:.1f} g/dL",
            )
        if hemoglobin < 8:
            _upsert_numeric_finding(
                findings,
                key="severe_anemia",
                label="Severe anemia",
                value=hemoglobin,
                units="g/dL",
                note=f"Hemoglobin {hemoglobin:.1f} g/dL",
            )

    if match := INR_PATTERN.search(lowered):
        inr = float(match.group(1))
        numeric_values["inr"] = inr
        units["inr"] = ""
        if inr >= 3.0:
            _upsert_numeric_finding(
                findings,
                key="supratherapeutic_inr",
                label="Supratherapeutic INR",
                value=inr,
                units=None,
                note=f"INR {inr:.1f}",
            )

    if match := CREATININE_PATTERN.search(lowered):
        creatinine = float(match.group(1))
        numeric_values["creatinine"] = creatinine
        units["creatinine"] = "mg/dL"
        if creatinine >= 2.0:
            _upsert_numeric_finding(
                findings,
                key="creatinine_elevated",
                label="Elevated creatinine",
                value=creatinine,
                units="mg/dL",
                note=f"Creatinine {creatinine:.1f} mg/dL",
            )

    if match := LACTATE_PATTERN.search(lowered):
        lactate = float(match.group(1))
        numeric_values["lactate"] = lactate
        units["lactate"] = "mmol/L"
        if lactate >= 2.0:
            _upsert_numeric_finding(
                findings,
                key="elevated_lactate",
                label="Elevated lactate",
                value=lactate,
                units="mmol/L",
                note=f"Lactate {lactate:.1f} mmol/L",
            )
        if lactate >= 4.0:
            _upsert_numeric_finding(
                findings,
                key="critical_values_present",
                label="Critical lactate value",
                value=lactate,
                units="mmol/L",
                note=f"Lactate {lactate:.1f} mmol/L",
            )

    if match := SODIUM_PATTERN.search(lowered):
        sodium = float(match.group(1))
        numeric_values["sodium"] = sodium
        units["sodium"] = "mEq/L"
        if sodium < 130:
            _upsert_numeric_finding(
                findings,
                key="hyponatremia",
                label="Hyponatremia",
                value=sodium,
                units="mEq/L",
                note=f"Sodium {sodium:.0f} mEq/L",
            )

    if match := BNP_PATTERN.search(lowered):
        bnp = float(match.group(1))
        numeric_values["bnp"] = bnp
        units["bnp"] = "pg/mL"
        if bnp >= 500:
            _upsert_numeric_finding(
                findings,
                key="bnp_elevated",
                label="Elevated BNP/NT-proBNP",
                value=bnp,
                units="pg/mL",
                note=f"BNP/NT-proBNP {bnp:.0f}",
            )

    if match := TROPONIN_PATTERN.search(lowered):
        troponin = float(match.group(1))
        numeric_values["troponin"] = troponin
        units["troponin"] = "assay"
        if (0.04 <= troponin <= 20.0) or troponin >= 100.0:
            _upsert_numeric_finding(
                findings,
                key="troponin_positive",
                label="Positive troponin",
                value=troponin,
                units=None,
                note=f"Troponin {troponin:g}",
            )

    return findings, numeric_values, units


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
    numeric_findings, numeric_values, units = _extract_numeric_findings(note_text)
    findings_by_key.update(numeric_findings)
    findings = list(findings_by_key.values())
    medications = sorted({canonical for needle, canonical in KEYWORD_MEDICATIONS.items() if needle in lowered})
    adverse_events = sorted({canonical for needle, canonical in KEYWORD_ADVERSE_EVENTS.items() if needle in lowered})
    completed_tests = _extract_completed_tests(lowered)
    age_years = _extract_age(note_text)
    pregnant = any(token in lowered for token in ("pregnant", "pregnancy", "gestation"))
    renal_impairment = (
        any(token in lowered for token in ("ckd", "chronic kidney disease", "renal impairment", "dialysis"))
        or "creatinine" in numeric_values and numeric_values["creatinine"] >= 2.0
    )
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
        completed_tests=completed_tests,
        medications=medications,
        adverse_events=adverse_events,
        age_years=age_years,
        pregnant=pregnant,
        renal_impairment=renal_impairment,
        hemodynamic_instability=any(token in lowered for token in ("shock", "hypotension", "unstable")) or "hemodynamic_instability" in findings_by_key,
        critical_values_present=(
            any(token in lowered for token in ("critical", "severe hypoxia"))
            or "critical_values_present" in findings_by_key
            or ("lactate" in numeric_values and numeric_values["lactate"] >= 4.0)
        ),
    )


def _merge_contexts(primary: ClinicalDecisionContext, fallback: ClinicalDecisionContext) -> ClinicalDecisionContext:
    merged_findings = {finding.key: finding for finding in fallback.findings}
    for finding in primary.findings:
        merged_findings[finding.key] = finding
    return ClinicalDecisionContext(
        case_id=primary.case_id,
        specialty=primary.specialty or fallback.specialty,
        findings=list(merged_findings.values()),
        completed_tests=sorted({*fallback.completed_tests, *primary.completed_tests}),
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
