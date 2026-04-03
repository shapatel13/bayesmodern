from __future__ import annotations

import json

from datasets.adapters.findzebra import normalize_findzebra_row
from datasets.catalog import get_dataset_spec
from datasets.loader import load_benchmark_tasks, load_curriculum_train_validation_tasks, load_train_validation_tasks
from utils.config import Settings


def test_medmcqa_loader_normalizes_question_rows(monkeypatch) -> None:
    from datasets import loader

    def fake_load_dataset_rows(dataset_name: str, split: str, subset: str | None = None, limit: int | None = None):
        assert dataset_name == "openlifescienceai/medmcqa"
        assert split == "validation"
        return [
            {
                "id": 42,
                "question": "Most likely diagnosis?",
                "opa": "Diagnosis A",
                "opb": "Diagnosis B",
                "opc": "Diagnosis C",
                "opd": "Diagnosis D",
                "cop": 1,
                "subject_name": "Medicine",
                "topic_name": "Cardiology",
            }
        ]

    monkeypatch.setattr(loader, "load_dataset_rows", fake_load_dataset_rows)
    tasks = load_benchmark_tasks("medmcqa", split="validation", limit=1)

    assert len(tasks) == 1
    assert tasks[0].task_id == "medmcqa-42"
    assert tasks[0].gold_answer == "Diagnosis B"
    assert tasks[0].metadata["subject"] == "Medicine"


def test_train_validation_loader_uses_dataset_defaults(monkeypatch) -> None:
    from datasets import loader

    observed_splits: list[str] = []

    def fake_load_dataset_rows(dataset_name: str, split: str, subset: str | None = None, limit: int | None = None):
        observed_splits.append(split)
        return [{"id": split, "question": "Question", "options": {"A": "A"}, "answer": "A"}]

    monkeypatch.setattr(loader, "load_dataset_rows", fake_load_dataset_rows)
    pair = load_train_validation_tasks("medqa", train_limit=1, validation_limit=1)

    assert observed_splits == ["train", "test"]
    assert pair.spec == get_dataset_spec("medqa")
    assert pair.train[0].source_dataset == "medqa"
    assert pair.validation[0].source_dataset == "medqa"


def test_credential_gated_dataset_without_path_raises_helpful_error(monkeypatch) -> None:
    from datasets import loader

    monkeypatch.setattr(loader, "get_settings", lambda: Settings(_env_file=None))

    try:
        load_benchmark_tasks("mietic", split="test", limit=1)
    except ValueError as exc:
        assert "PRIORI_MIETIC_PATH" in str(exc)
    else:
        raise AssertionError("Expected local-only dataset access to raise a helpful error.")


def test_findzebra_case_report_normalizer_handles_list_fields() -> None:
    task = normalize_findzebra_row(
        {
            "id": "fabry-1",
            "title": "Fabry case report",
            "text": ["A young patient with renal dysfunction.", "Additional case detail."],
            "labels": ["Fabry disease", "lysosomal storage disease"],
        },
        "train",
    )

    assert "renal dysfunction" in task.prompt
    assert task.gold_diagnosis == "Fabry disease"
    assert task.metadata["rare_disease"] is True


def test_curriculum_loader_interleaves_public_hf_tasks(monkeypatch) -> None:
    from datasets import loader

    def fake_load_dataset_rows(dataset_name: str, split: str, subset: str | None = None, limit: int | None = None):
        if dataset_name == "openlifescienceai/medmcqa":
            return [
                {
                    "id": f"medmcqa-{split}-1",
                    "question": "Most likely diagnosis?",
                    "opa": "Diagnosis A",
                    "opb": "Diagnosis B",
                    "opc": "Diagnosis C",
                    "opd": "Diagnosis D",
                    "cop": 1,
                    "subject_name": "Medicine",
                    "topic_name": "General",
                },
                {
                    "id": f"medmcqa-{split}-2",
                    "question": "Next best step?",
                    "opa": "Option A",
                    "opb": "Option B",
                    "opc": "Option C",
                    "opd": "Option D",
                    "cop": 0,
                    "subject_name": "Medicine",
                    "topic_name": "General",
                },
            ]
        return [
            {
                "id": f"medqa-{split}-1",
                "question": "Clinical question?",
                "options": {"A": "A", "B": "B", "C": "C", "D": "D"},
                "answer": "A",
            },
            {
                "id": f"medqa-{split}-2",
                "question": "Clinical question 2?",
                "options": {"A": "A", "B": "B", "C": "C", "D": "D"},
                "answer": "B",
            },
        ]

    monkeypatch.setattr(loader, "load_dataset_rows", fake_load_dataset_rows)
    pair = load_curriculum_train_validation_tasks("diagnostic_reasoning_feedback_lab")

    assert pair.curriculum.key == "diagnostic_reasoning_feedback_lab"
    assert len(pair.components) == 2
    assert {component.dataset_key for component in pair.components} == {"medmcqa", "medqa"}
    assert pair.train[0].metadata["curriculum_key"] == "diagnostic_reasoning_feedback_lab"
    assert pair.train[0].metadata["reward_profile_hint"] == "diagnostic"
    assert {task.metadata["curriculum_component"] for task in pair.train[:4]} == {"medmcqa", "medqa"}


def test_curriculum_loader_respects_component_caps(monkeypatch) -> None:
    from datasets import loader

    def fake_load_dataset_rows(dataset_name: str, split: str, subset: str | None = None, limit: int | None = None):
        if dataset_name == "qiaojin/PubMedQA":
            rows = [{"pubid": "1", "question": "Claim?", "long_answer": "yes", "final_decision": "yes"}]
        else:
            rows = [{"id": "case-1", "title": "Rare case", "text": "Clinical text", "labels": ["Rare disease"]}]
        return rows if limit is None else rows[:limit]

    monkeypatch.setattr(loader, "load_dataset_rows", fake_load_dataset_rows)
    pair = load_curriculum_train_validation_tasks(
        "evidence_rare_feedback_lab",
        train_cap_per_component=1,
        validation_cap_per_component=0,
    )

    assert len(pair.train) == 2
    assert len(pair.validation) == 0


def test_reviewed_cases_loader_reads_local_jsonl(tmp_path, monkeypatch) -> None:
    from datasets import loader

    reviewed_path = tmp_path / "reviewed_cases.jsonl"
    rows = [
        {
            "id": f"case-{index}",
            "task_type": "diagnosis_open",
            "prompt": f"Clinical prompt {index}",
            "gold_diagnosis": "pe",
            "acceptable_tests": ["d_dimer"],
            "review_status": "approved",
        }
        for index in range(20)
    ]
    reviewed_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    monkeypatch.setattr(loader, "get_settings", lambda: Settings(_env_file=None, reviewed_cases_path=str(reviewed_path)))

    pair = load_train_validation_tasks("reviewed_cases", train_limit=5, validation_limit=5)

    assert pair.spec.key == "reviewed_cases"
    assert pair.train
    assert pair.validation
    assert all(task.source_dataset == "reviewed_cases" for task in pair.train + pair.validation)
    assert pair.train[0].metadata["review_status"] == "approved"


def test_reviewed_cases_loader_excludes_draft_rows(tmp_path, monkeypatch) -> None:
    from datasets import loader

    reviewed_path = tmp_path / "reviewed_cases.jsonl"
    rows = [
        {
            "id": "approved-1",
            "task_type": "diagnosis_open",
            "prompt": "Clinical prompt approved",
            "gold_diagnosis": "pe",
            "acceptable_tests": ["d_dimer"],
            "review_status": "approved",
        },
        {
            "id": "draft-1",
            "task_type": "diagnosis_open",
            "prompt": "Clinical prompt draft",
            "gold_diagnosis": "acs",
            "acceptable_tests": ["ecg"],
            "review_status": "draft",
        },
    ]
    reviewed_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    monkeypatch.setattr(loader, "get_settings", lambda: Settings(_env_file=None, reviewed_cases_path=str(reviewed_path)))

    pair = load_train_validation_tasks("reviewed_cases", train_limit=5, validation_limit=5)
    loaded_ids = {task.task_id for task in pair.train + pair.validation}

    assert "reviewed-approved-1" in loaded_ids
    assert "reviewed-draft-1" not in loaded_ids
