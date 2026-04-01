from __future__ import annotations

from core.models import TriageAssessment


def assess_triage(
    dangerous_posterior_mass: float,
    hemodynamic_instability: bool,
    critical_values_present: bool,
) -> TriageAssessment:
    reasons: list[str] = []

    if hemodynamic_instability:
        reasons.append("Hemodynamic instability present")
    if critical_values_present:
        reasons.append("Critical values or red-flag findings present")
    if dangerous_posterior_mass >= 0.55:
        reasons.append("High aggregate posterior for dangerous diagnoses")

    if hemodynamic_instability or critical_values_present or dangerous_posterior_mass >= 0.85:
        return TriageAssessment(
            urgency="emergent",
            reasons=reasons or ["Immediate stabilization concern"],
            admit_threshold_crossed=True,
            icu_threshold_crossed=True,
        )
    if dangerous_posterior_mass >= 0.55:
        return TriageAssessment(
            urgency="urgent",
            reasons=reasons or ["Meaningful short-term deterioration risk"],
            admit_threshold_crossed=True,
            icu_threshold_crossed=False,
        )
    if dangerous_posterior_mass >= 0.3:
        return TriageAssessment(
            urgency="expedited",
            reasons=reasons or ["Moderate concern requiring prompt workup"],
            admit_threshold_crossed=False,
            icu_threshold_crossed=False,
        )
    return TriageAssessment(
        urgency="routine",
        reasons=reasons or ["No immediate instability signals detected"],
        admit_threshold_crossed=False,
        icu_threshold_crossed=False,
    )
