from __future__ import annotations

from evidence.literature_registry import LITERATURE_REGISTRY
from llm.structured_output import ResearchReport


def find_missing_citations(report: ResearchReport) -> list[str]:
    missing: list[str] = []
    for entry in report.differential.ranked:
        for badge in entry.provenance_badges:
            if badge.startswith("source:"):
                continue
            if badge not in LITERATURE_REGISTRY:
                missing.append(badge)
    for recommendation in report.next_best_tests:
        for badge in recommendation.provenance_badges:
            if badge.startswith("source:"):
                continue
            if badge not in LITERATURE_REGISTRY:
                missing.append(badge)
    return sorted(set(missing))

