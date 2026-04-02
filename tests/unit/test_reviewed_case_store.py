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
        policy_version="v1-deterministic",
        prompt_version="v1-offline",
    )

    destination = reviewed_case_store.append_reviewed_case(row, settings)

    payload = json.loads(destination.read_text(encoding="utf-8").splitlines()[0])
    assert payload["gold_diagnosis"] == "acs"
    assert payload["acceptable_tests"] == ["ecg", "hs_troponin"]
    assert payload["suggested_top_diagnosis"] == "pneumonia"


def test_reviewed_case_store_resolves_default_destination(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        reviewed_case_store,
        "_DEFAULT_REVIEWED_CASES_PATH",
        tmp_path / "artifacts" / "reviewed_cases" / "reviewed_cases.local.jsonl",
    )

    destination = reviewed_case_store.resolve_reviewed_cases_destination(Settings(_env_file=None))

    assert destination.name == "reviewed_cases.local.jsonl"
