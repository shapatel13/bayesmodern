from __future__ import annotations

import json

from datasets import reviewed_case_store
from utils.config import Settings


def test_reviewed_case_store_appends_jsonl_row(tmp_path) -> None:
    settings = Settings(_env_file=None, reviewed_cases_path=str(tmp_path / "reviewed_cases.local.jsonl"))
    row = reviewed_case_store.build_reviewed_case_row(
        note_text="Crushing chest pain with hypotension.",
        gold_diagnosis="acs",
        acceptable_tests=["ecg", "hs_troponin"],
        gold_triage="emergent",
        review_status="approved",
        reviewer_id="tester",
        review_notes="Regression case.",
        suggested_top_diagnosis="pneumonia",
        suggested_next_tests=["cxr"],
        reviewed_mechanism_states=["thrombotic_ischemic_tendency"],
        reviewed_contributing_processes=["plaque rupture"],
        mechanism_feedback_summary="Mechanism should favor ischemia.",
        preferred_next_action="Immediate ECG and serial troponin.",
        suggested_mechanism_states=["low_effective_arterial_volume"],
        suggested_mechanism_summary="Mechanism layer remained broad.",
        policy_version="v1-deterministic",
        prompt_version="v1-offline",
        tags=["wrong_top_diagnosis", "parser_miss", "bad_mechanism_inference", "wrong_top_diagnosis"],
    )

    destination = reviewed_case_store.append_reviewed_case(row, settings)

    payload = json.loads(destination.read_text(encoding="utf-8").splitlines()[0])
    assert payload["gold_diagnosis"] == "acs"
    assert payload["acceptable_tests"] == ["ecg", "hs_troponin"]
    assert payload["suggested_top_diagnosis"] == "pneumonia"
    assert payload["reviewed_mechanism_states"] == ["thrombotic_ischemic_tendency"]
    assert payload["preferred_next_action"] == "Immediate ECG and serial troponin."
    assert payload["tags"] == ["wrong_top_diagnosis", "parser_miss", "bad_mechanism_inference"]


def test_reviewed_case_store_resolves_default_destination(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        reviewed_case_store,
        "_DEFAULT_REVIEWED_CASES_PATH",
        tmp_path / "artifacts" / "reviewed_cases" / "reviewed_cases.local.jsonl",
    )

    destination = reviewed_case_store.resolve_reviewed_cases_destination(Settings(_env_file=None))

    assert destination.name == "reviewed_cases.local.jsonl"


def test_reviewed_case_store_summarizes_rows(tmp_path) -> None:
    settings = Settings(_env_file=None, reviewed_cases_path=str(tmp_path / "reviewed_cases.local.jsonl"))
    first = reviewed_case_store.build_reviewed_case_row(
        note_text="Crushing chest pain with hypotension.",
        gold_diagnosis="acs",
        acceptable_tests=["ecg"],
        gold_triage="emergent",
        review_status="approved",
        reviewer_id="tester",
        review_notes="Approved regression case.",
        tags=["wrong_top_diagnosis", "parser_miss"],
    )
    second = reviewed_case_store.build_reviewed_case_row(
        note_text="Fever and crackles.",
        gold_diagnosis="pneumonia",
        acceptable_tests=["cxr"],
        gold_triage="expedited",
        review_status="draft",
        reviewer_id="tester",
        review_notes="Draft case.",
        tags=["good_counterexample"],
    )
    reviewed_case_store.append_reviewed_case(first, settings)
    reviewed_case_store.append_reviewed_case(second, settings)

    summary = reviewed_case_store.summarize_reviewed_cases(settings)

    assert summary["total_cases"] == 2
    assert summary["approved_cases"] == 1
    assert summary["draft_cases"] == 1
    assert summary["tagged_cases"] == 2
    assert summary["tag_counts"]["good_counterexample"] == 1
    assert summary["tag_counts"]["parser_miss"] == 1
    assert summary["tag_counts"]["wrong_top_diagnosis"] == 1
    assert summary["recent_rows"][0]["gold_diagnosis"] == "pneumonia"
