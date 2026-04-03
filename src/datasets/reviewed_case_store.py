from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from utils.config import Settings, get_settings
from utils.dates import utc_now
from utils.ids import make_id


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REVIEWED_CASES_PATH = _REPO_ROOT / "artifacts" / "reviewed_cases" / "reviewed_cases.local.jsonl"
REVIEWED_CASE_TAGS: tuple[str, ...] = (
    "wrong_top_diagnosis",
    "bad_next_test",
    "bad_mechanism_inference",
    "urgency_error",
    "parser_miss",
    "unsupported_claim",
    "hallucination",
    "weak_calibration",
    "safety_risk",
    "cost_stewardship",
    "good_counterexample",
)


def _normalize_tags(tags: list[str] | None) -> list[str]:
    if not tags:
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_tag in tags:
        tag = str(raw_tag or "").strip().lower()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        normalized.append(tag)
    return normalized


def resolve_reviewed_cases_destination(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    raw_path = settings.reviewed_cases_path
    if not raw_path:
        return _DEFAULT_REVIEWED_CASES_PATH

    candidate = Path(raw_path)
    if candidate.suffix.lower() in {".jsonl", ".json", ".csv"}:
        return candidate
    return candidate / "reviewed_cases.local.jsonl"


def build_reviewed_case_row(
    *,
    note_text: str,
    gold_diagnosis: str | None,
    acceptable_tests: list[str],
    gold_triage: str | None,
    review_status: str,
    reviewer_id: str | None,
    review_notes: str | None,
    suggested_top_diagnosis: str | None = None,
    suggested_next_tests: list[str] | None = None,
    reviewed_mechanism_states: list[str] | None = None,
    reviewed_contributing_processes: list[str] | None = None,
    mechanism_feedback_summary: str | None = None,
    preferred_next_action: str | None = None,
    suggested_mechanism_states: list[str] | None = None,
    suggested_mechanism_summary: str | None = None,
    policy_version: str | None = None,
    prompt_version: str | None = None,
    tags: list[str] | None = None,
    task_type: str = "diagnosis_open",
) -> dict[str, Any]:
    return {
        "id": f"reviewed-{make_id('case')}",
        "task_type": task_type,
        "note_text": note_text.strip(),
        "gold_diagnosis": gold_diagnosis.strip() if gold_diagnosis else None,
        "acceptable_tests": [item for item in acceptable_tests if item],
        "gold_triage": gold_triage.strip() if gold_triage else None,
        "review_status": review_status.strip().lower(),
        "reviewer_id": reviewer_id.strip() if reviewer_id else None,
        "review_notes": review_notes.strip() if review_notes else None,
        "captured_at": utc_now().isoformat(),
        "captured_from": "research_console",
        "suggested_top_diagnosis": suggested_top_diagnosis,
        "suggested_next_tests": suggested_next_tests or [],
        "reviewed_mechanism_states": [item for item in (reviewed_mechanism_states or []) if item],
        "reviewed_contributing_processes": [item for item in (reviewed_contributing_processes or []) if item],
        "mechanism_feedback_summary": mechanism_feedback_summary.strip() if mechanism_feedback_summary else None,
        "preferred_next_action": preferred_next_action.strip() if preferred_next_action else None,
        "suggested_mechanism_states": [item for item in (suggested_mechanism_states or []) if item],
        "suggested_mechanism_summary": suggested_mechanism_summary.strip() if suggested_mechanism_summary else None,
        "policy_version": policy_version,
        "prompt_version": prompt_version,
        "tags": _normalize_tags(tags),
    }


def append_reviewed_case(row: dict[str, Any], settings: Settings | None = None) -> Path:
    destination = resolve_reviewed_cases_destination(settings)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True))
        handle.write("\n")
    return destination


def load_reviewed_cases(settings: Settings | None = None, *, limit: int | None = None) -> list[dict[str, Any]]:
    destination = resolve_reviewed_cases_destination(settings)
    if not destination.exists():
        return []

    rows: list[dict[str, Any]] = []
    with destination.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            rows.append(json.loads(payload))

    rows.reverse()
    if limit is not None:
        return rows[:limit]
    return rows


def summarize_reviewed_cases(settings: Settings | None = None) -> dict[str, Any]:
    rows = load_reviewed_cases(settings)
    approved = sum(1 for row in rows if str(row.get("review_status") or "").lower() == "approved")
    draft = sum(1 for row in rows if str(row.get("review_status") or "").lower() == "draft")
    diagnoses = [str(row.get("gold_diagnosis") or "unknown") for row in rows]
    tagged_cases = sum(1 for row in rows if row.get("tags"))
    tag_counts = Counter(
        str(tag).strip().lower()
        for row in rows
        for tag in (row.get("tags") or [])
        if str(tag).strip()
    )
    sorted_tag_counts = dict(sorted(tag_counts.items(), key=lambda item: (-item[1], item[0])))
    return {
        "total_cases": len(rows),
        "approved_cases": approved,
        "draft_cases": draft,
        "tagged_cases": tagged_cases,
        "tag_counts": sorted_tag_counts,
        "top_tag": next(iter(sorted_tag_counts), None),
        "recent_rows": rows[:5],
        "diagnoses": diagnoses[:10],
    }
