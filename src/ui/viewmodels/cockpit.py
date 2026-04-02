from __future__ import annotations

from llm.structured_output import ResearchReport


def differential_rows(report: ResearchReport) -> list[dict[str, object]]:
    return [
        {
            "Diagnosis": entry.name,
            "Posterior": round(entry.posterior, 3),
            "Interval": f"{entry.interval_low:.2f} - {entry.interval_high:.2f}",
            "Coverage": round(entry.symptom_coverage, 2),
            "Calibration": entry.calibration_state,
        }
        for entry in report.differential.ranked
    ]


def next_test_rows(report: ResearchReport) -> list[dict[str, object]]:
    return [
        {
            "Test": recommendation.name,
            "Disposition": recommendation.disposition,
            "Info Gain": round(recommendation.expected_information_gain, 3),
            "Mechanism Gain": round(recommendation.mechanistic_information_gain, 3),
            "Stewardship": round(recommendation.stewardship_score, 3),
            "Direct Cost": recommendation.direct_cost,
            "Downstream Cost": recommendation.downstream_cost,
            "Risk Penalty": round(recommendation.risk_penalty, 2),
            "LR+": round(recommendation.lr_plus, 2),
            "LR-": round(recommendation.lr_minus, 2),
        }
        for recommendation in report.next_best_tests
    ]


def mechanism_rows(report: ResearchReport) -> list[dict[str, object]]:
    return [
        {
            "Mechanism State": estimate.name,
            "Posterior": round(estimate.posterior, 3),
            "Interval": f"{estimate.interval_low:.2f} - {estimate.interval_high:.2f}",
            "Confidence": estimate.confidence_state,
            "Evidence For": len(estimate.evidence_for),
            "Evidence Against": len(estimate.evidence_against),
        }
        for estimate in report.mechanism_states.ranked
    ]


def provenance_rows(report: ResearchReport) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for entry in report.differential.ranked:
        rows.append(
            {
                "Artifact": entry.name,
                "Badges": ", ".join(entry.provenance_badges),
            }
        )
    for estimate in report.mechanism_states.ranked:
        rows.append(
            {
                "Artifact": estimate.name,
                "Badges": ", ".join(estimate.provenance_badges),
            }
        )
    for recommendation in report.next_best_tests:
        rows.append(
            {
                "Artifact": recommendation.name,
                "Badges": ", ".join(recommendation.provenance_badges),
            }
        )
    return rows
