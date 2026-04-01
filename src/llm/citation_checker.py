from __future__ import annotations

from evidence.literature_registry import is_known_provenance_ref
from llm.structured_output import ResearchReport


def find_missing_citations(report: ResearchReport) -> list[str]:
    missing: list[str] = []
    for entry in report.differential.ranked:
        for badge in entry.provenance_badges:
            if badge.startswith("source:"):
                continue
            if not is_known_provenance_ref(badge):
                missing.append(badge)
    for recommendation in report.next_best_tests:
        for badge in recommendation.provenance_badges:
            if badge.startswith("source:"):
                continue
            if not is_known_provenance_ref(badge):
                missing.append(badge)
    return sorted(set(missing))
