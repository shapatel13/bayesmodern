from __future__ import annotations

from pathlib import Path

from agent.trace_schema import ExperimentTrace, RewardBreakdown, TraceStep
from datasets.adapters.mietic import infer_triage_label, normalize_mietic_row
from datasets.adapters.n2c2_2018_track2 import normalize_n2c2_2018_track2_row
from datasets.loader import load_benchmark_tasks
from eval.medication_safety_eval import medication_extraction_recall, medication_safety_summary
from eval.triage_eval import triage_accuracy, triage_confusion_counts
from llm.structured_output import ModelRoutingDecision, ResearchReport
from utils.config import Settings


def _sample_report(*, medications: list[str] | None = None, adverse_events: list[str] | None = None, urgency: str = "urgent") -> ResearchReport:
    return ResearchReport(
        context={
            "case_id": "task-1",
            "specialty": "general_internal_medicine",
            "findings": [],
            "completed_tests": [],
            "comorbidities": [],
            "medications": medications or [],
            "adverse_events": adverse_events or [],
            "symptoms_free_text": None,
            "age_years": None,
            "pregnant": False,
            "renal_impairment": False,
            "hemodynamic_instability": False,
            "critical_values_present": False,
            "safety_mode": "conservative",
        },
        differential={
            "ranked": [
                {
                    "slug": "pe",
                    "name": "Pulmonary Embolism",
                    "prior": 0.2,
                    "posterior": 0.7,
                    "interval_low": 0.6,
                    "interval_high": 0.8,
                    "evidence_for": [],
                    "evidence_against": [],
                    "symptom_coverage": 0.5,
                    "explaining_away": [],
                    "calibration_state": "moderately_uncertain",
                    "provenance_badges": ["source:hard-coded"],
                }
            ],
            "posterior_mass_top3": 0.7,
            "model_note": "demo",
        },
        next_best_tests=[],
        triage={
            "urgency": urgency,
            "reasons": ["demo"],
            "admit_threshold_crossed": urgency in {"urgent", "emergent"},
            "icu_threshold_crossed": urgency == "emergent",
        },
        threshold_decision={
            "action": "test",
            "clinician_language": "demo clinician threshold framing",
            "plain_language": "demo plain-language threshold framing",
        },
        contradictions=[],
        provenance_warnings=[],
        model_route=ModelRoutingDecision(
            parser_model="gpt-5.4-nano-2026-03-17",
            reasoning_model="gpt-5.4-nano-2026-03-17",
            verifier_model="gpt-5.4-nano-2026-03-17",
            mode="offline",
        ),
    )


def _trace(*, task_type: str, task_metadata: dict[str, object], gold_triage: str | None = None, medications: list[str] | None = None, adverse_events: list[str] | None = None, urgency: str = "urgent") -> ExperimentTrace:
    return ExperimentTrace(
        task_id="task-1",
        source_dataset="synthetic",
        task_type=task_type,
        task_metadata=task_metadata,
        gold_triage=gold_triage,
        prompt_version="v1-offline",
        policy_version="v1-deterministic",
        model_route="offline",
        steps=[TraceStep(name="extract", detail="demo")],
        report=_sample_report(medications=medications, adverse_events=adverse_events, urgency=urgency),
        reward=RewardBreakdown(
            diagnostic_correctness=1.0,
            topk_differential_quality=1.0,
            calibration_quality=1.0,
            next_test_quality=1.0,
            stewardship=1.0,
            safety=1.0,
            urgency=1.0,
            provenance=1.0,
            json_validity=1.0,
            consistency=1.0,
            total_reward=1.0,
        ),
    )


def test_infer_triage_label_maps_esi_levels() -> None:
    assert infer_triage_label("ESI 1 - immediate resuscitation") == "emergent"
    assert infer_triage_label("ESI 2 high acuity") == "urgent"
    assert infer_triage_label("ESI 3") == "expedited"
    assert infer_triage_label("ESI 5 low acuity") == "routine"


def test_mietic_normalizer_creates_triage_task() -> None:
    task = normalize_mietic_row(
        {
            "id": "visit-1",
            "instruction": "Assign the triage category.",
            "input": "Crushing chest pain, diaphoresis, hypotension.",
            "output": "ESI 1 immediate resuscitation",
        },
        "train",
    )

    assert task.task_type == "triage"
    assert task.gold_triage == "emergent"
    assert "Assign the triage category." in task.prompt


def test_n2c2_normalizer_collects_medications_and_adverse_events() -> None:
    task = normalize_n2c2_2018_track2_row(
        {
            "doc_id": "note-7",
            "text": "Started lisinopril. Later developed angioedema.",
            "entities": [
                {"type": "Drug", "text": "lisinopril"},
                {"type": "Adverse-Event", "text": "angioedema"},
            ],
            "relations": [{"type": "causes", "arg1": "lisinopril", "arg2": "angioedema"}],
        },
        "train",
    )

    assert task.task_type == "medication_safety"
    assert task.metadata["gold_medications"] == ["lisinopril"]
    assert task.metadata["gold_adverse_events"] == ["angioedema"]
    assert "causes" in task.metadata["gold_relations"][0]


def test_local_credentialed_loader_reads_mietic_csv(tmp_path: Path, monkeypatch) -> None:
    from datasets import loader

    csv_path = tmp_path / "validation.csv"
    csv_path.write_text(
        "id,instruction,input,output\nvisit-2,Assign triage,Fever and cough,ESI 3 urgent review\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(loader, "get_settings", lambda: Settings(_env_file=None, mietic_path=str(tmp_path)))

    tasks = load_benchmark_tasks("mietic", split="validation", limit=1)

    assert len(tasks) == 1
    assert tasks[0].source_dataset == "mietic"
    assert tasks[0].gold_triage == "expedited"


def test_local_hybrid_loader_reads_n2c2_jsonl(tmp_path: Path, monkeypatch) -> None:
    from datasets import loader

    dataset_dir = tmp_path / "n2c2"
    dataset_dir.mkdir()
    (dataset_dir / "test.jsonl").write_text(
        '{"id":"rx-1","note":"Warfarin then GI bleed.","medications":["warfarin"],"adverse_events":["gi bleed"]}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(loader, "get_settings", lambda: Settings(_env_file=None, n2c2_2018_track2_path=str(dataset_dir)))

    tasks = load_benchmark_tasks("n2c2_2018_track2", split="test", limit=1)

    assert len(tasks) == 1
    assert tasks[0].metadata["gold_medications"] == ["warfarin"]


def test_triage_eval_reports_accuracy_and_confusion() -> None:
    traces = [
        _trace(task_type="triage", task_metadata={}, gold_triage="urgent", urgency="urgent"),
        _trace(task_type="triage", task_metadata={}, gold_triage="emergent", urgency="urgent"),
    ]

    assert triage_accuracy(traces) == 0.5
    assert triage_confusion_counts(traces) == {"urgent->urgent": 1, "emergent->urgent": 1}


def test_medication_safety_eval_scores_overlap() -> None:
    traces = [
        _trace(
            task_type="medication_safety",
            task_metadata={"gold_medications": ["lisinopril"], "gold_adverse_events": ["angioedema"]},
            medications=["lisinopril"],
            adverse_events=["angioedema"],
        ),
        _trace(
            task_type="medication_safety",
            task_metadata={"gold_medications": ["warfarin"], "gold_adverse_events": ["gi bleed"]},
            medications=[],
            adverse_events=[],
        ),
    ]

    assert medication_extraction_recall(traces) == 0.5
    summary = medication_safety_summary(traces)
    assert summary["medication_recall"] == 0.5
    assert summary["adverse_event_recall"] == 0.5
