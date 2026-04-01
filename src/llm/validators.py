from __future__ import annotations

from llm.structured_output import ResearchReport


def validate_report(report: ResearchReport) -> list[str]:
    contradictions: list[str] = []
    posterior_sum = sum(entry.posterior for entry in report.differential.ranked)
    if abs(posterior_sum - 1.0) > 0.05:
        contradictions.append("Differential probabilities do not sum close to 1.0.")
    completed_tests = set(report.context.completed_tests)
    for recommendation in report.next_best_tests:
        if recommendation.slug in completed_tests and recommendation.disposition != "already_answered":
            contradictions.append(f"Recommended test `{recommendation.slug}` is already completed.")
    if report.triage.urgency == "emergent" and report.threshold_decision.action == "observe":
        contradictions.append("Emergent triage conflicts with an observe-only threshold decision.")
    return contradictions

