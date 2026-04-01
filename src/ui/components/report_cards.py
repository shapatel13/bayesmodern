from __future__ import annotations

from llm.structured_output import ResearchReport


def headline_cards(report: ResearchReport) -> list[tuple[str, str]]:
    top = report.differential.ranked[0] if report.differential.ranked else None
    top_test = report.next_best_tests[0] if report.next_best_tests else None
    return [
        ("Top Differential", f"{top.name} ({top.posterior:.1%})" if top else "Unavailable"),
        ("Triage", report.triage.urgency.title()),
        ("Threshold Action", report.threshold_decision.action.replace("_", " ").title()),
        ("Best Next Test", top_test.name if top_test else "No recommendation"),
    ]

