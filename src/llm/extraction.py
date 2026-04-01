from __future__ import annotations

from core.models import ClinicalDecisionContext, ClinicalFinding


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


def extract_context_from_text(case_id: str, note_text: str) -> ClinicalDecisionContext:
    lowered = note_text.lower()
    findings = [
        ClinicalFinding(key=key, label=label, present=True)
        for needle, (key, label) in KEYWORD_FINDINGS.items()
        if needle in lowered
    ]
    return ClinicalDecisionContext(
        case_id=case_id,
        symptoms_free_text=note_text,
        findings=findings,
        hemodynamic_instability=any(token in lowered for token in ("shock", "hypotension", "unstable")),
        critical_values_present=any(token in lowered for token in ("lactate", "critical", "severe hypoxia")),
    )

