from __future__ import annotations

from pathlib import Path

from datasets.adapters.demo_cases import normalize_demo_case_row
from datasets.adapters.medval_bench import normalize_medval_bench_row
from agent.trace_schema import ExperimentTrace, RewardBreakdown, TraceStep
from datasets.adapters.mietic import infer_triage_label, normalize_mietic_row
from datasets.adapters.n2c2_2018_track2 import normalize_n2c2_2018_track2_row
from datasets.loader import load_benchmark_tasks
from eval.benchmark_runner import run_benchmark
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


def test_demo_case_normalizer_preserves_tests_and_diagnosis() -> None:
    task = normalize_demo_case_row(
        {
            "id": "demo-1",
            "task_type": "diagnosis_open",
            "prompt": "Pleuritic chest pain with hypoxemia after travel.",
            "gold_diagnosis": "pe",
            "acceptable_tests": ["d_dimer", "cta_pe"],
            "gold_triage": "urgent",
        },
        "train",
    )

    assert task.source_dataset == "priorix_demo_cases"
    assert task.gold_diagnosis == "pe"
    assert task.acceptable_tests == ["d_dimer", "cta_pe"]


def test_medval_bench_normalizer_creates_generation_audit_task() -> None:
    task = normalize_medval_bench_row(
        {
            "#": "17",
            "id": "286",
            "task": "medication2answer",
            "input": "what is fentanyl",
            "reference_output": "Fentanyl is a potent opioid analgesic.",
            "output": "Fentanyl is a mild over-the-counter pain reliever that is non-addictive.",
            "physician_error_assessment": "Unsafe hallucination.",
            "physician_risk_grade": "4",
        },
        "train",
    )

    assert task.task_type == "generation_audit"
    assert task.gold_risk_grade == 4
    assert task.metadata["medval_task"] == "medication2answer"
    assert "Candidate output to audit" in task.prompt


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


def test_single_file_medval_loader_partitions_local_csv(tmp_path: Path, monkeypatch) -> None:
    from datasets import loader

    csv_path = tmp_path / "medval_bench.csv"
    csv_path.write_text(
        "#,id,task,input,reference_output,output,physician_error_assessment,physician_risk_grade\n"
        "1,alpha,medication2answer,what is fentanyl,Fentanyl is a potent opioid,Fentanyl is OTC and non-addictive,Unsafe,4\n"
        "2,beta,medication2answer,what is heparin,Heparin is an anticoagulant,Heparin is harmless and OTC,Unsafe,4\n"
        "3,gamma,report2impression,report text,No acute findings,No acute findings,,1\n"
        "4,delta,report2impression,report text,No acute findings,Follow-up soon,Unsupported recommendation,3\n"
        "5,epsilon,query2question,23 surgeries and counting,How can I get rid of a birthmark permanently?,Are there cures in development?,Missing context,2\n"
        "6,zeta,query2question,three years of pain,What explains chronic pain?,What causes pain?,Missing duration,2\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(loader, "get_settings", lambda: Settings(_env_file=None, medval_bench_path=str(csv_path)))

    train_tasks = load_benchmark_tasks("medval_bench", split="train")
    validation_tasks = load_benchmark_tasks("medval_bench", split="validation")
    test_tasks = load_benchmark_tasks("medval_bench", split="test")
    limited_train = load_benchmark_tasks("medval_bench", split="train", limit=3)

    all_task_ids = {task.task_id for task in train_tasks + validation_tasks + test_tasks}
    assert len(all_task_ids) == 6
    assert all(task.task_type == "generation_audit" for task in train_tasks + validation_tasks + test_tasks)
    assert len({task.metadata["medval_task"] for task in limited_train}) >= 2


def test_repo_bundled_demo_datasets_load_without_env_configuration() -> None:
    reasoning_tasks = load_benchmark_tasks("priorix_demo_cases", split="test", limit=2)
    triage_tasks = load_benchmark_tasks("mietic_demo", split="test", limit=2)
    med_tasks = load_benchmark_tasks("n2c2_demo", split="test", limit=2)

    assert len(reasoning_tasks) == 2
    assert len(triage_tasks) == 2
    assert len(med_tasks) == 2
    assert reasoning_tasks[0].task_type == "diagnosis_open"
    assert triage_tasks[0].task_type == "triage"
    assert med_tasks[0].task_type == "medication_safety"


def test_generation_audit_benchmark_path_returns_audit_report() -> None:
    traces = run_benchmark(
        [
            normalize_medval_bench_row(
                {
                    "#": "17",
                    "id": "286",
                    "task": "medication2answer",
                    "input": "what is fentanyl",
                    "reference_output": "Fentanyl is a potent opioid analgesic.",
                    "output": "Fentanyl is a mild over-the-counter pain reliever that is non-addictive.",
                    "physician_error_assessment": "Unsafe hallucination.",
                    "physician_risk_grade": "4",
                },
                "test",
            )
        ]
    )

    assert len(traces) == 1
    assert traces[0].report.generation_audit is not None
    assert traces[0].report.generation_audit.predicted_risk_grade >= 3


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
