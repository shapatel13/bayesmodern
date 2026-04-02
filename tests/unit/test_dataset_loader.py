from __future__ import annotations

from datasets.adapters.findzebra import normalize_findzebra_row
from datasets.catalog import get_dataset_spec
from datasets.loader import load_benchmark_tasks, load_train_validation_tasks
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

    assert observed_splits == ["train", "validation"]
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
