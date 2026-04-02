from __future__ import annotations

import json
from pathlib import Path

from agent.lightning_adapter import detect_lightning_runtime, export_lightning_bundle, render_prompt_template
from agent.trace_schema import ExperimentTrace, RewardBreakdown, TraceStep
from llm.structured_output import ModelRoutingDecision, ResearchReport
from priorix_tasks.common import BenchmarkTask
from utils.config import Settings


def _sample_task(task_id: str = "task-1") -> BenchmarkTask:
    return BenchmarkTask(
        task_id=task_id,
        source_dataset="synthetic",
        split="test",
        task_type="diagnosis_open",
        prompt="Pleuritic chest pain with tachycardia and hypoxemia.",
        gold_diagnosis="pe",
        acceptable_tests=["d_dimer"],
        gold_triage="urgent",
    )


def _sample_trace() -> ExperimentTrace:
    report = ResearchReport(
        context={
            "case_id": "task-1",
            "specialty": "general_internal_medicine",
            "findings": [],
            "completed_tests": [],
            "comorbidities": [],
            "medications": [],
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
        next_best_tests=[
            {
                "slug": "d_dimer",
                "name": "D-dimer",
                "score": 0.2,
                "expected_information_gain": 0.1,
                "expected_posterior_movement": 0.15,
                "stewardship_score": 0.2,
                "disposition": "worth_it_now",
                "discriminates_between": ["pe"],
                "rationale": "demo",
                "lr_plus": 2.0,
                "lr_minus": 0.2,
                "direct_cost": 40.0,
                "downstream_cost": 0.0,
                "risk_penalty": 0.0,
                "provenance_badges": ["source:hard-coded"],
            }
        ],
        triage={
            "urgency": "urgent",
            "reasons": ["demo"],
            "admit_threshold_crossed": True,
            "icu_threshold_crossed": False,
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
    return ExperimentTrace(
        task_id="task-1",
        source_dataset="synthetic",
        task_type="diagnosis_open",
        task_metadata={
            "curriculum_key": "broad_medical_feedback_lab",
            "curriculum_component": "medqa",
            "reward_profile_hint": "diagnostic",
            "source_hf_dataset": "augtoma/medqa_usmle",
            "source_task_family": "diagnosis_mcq",
        },
        gold_diagnosis="pe",
        acceptable_tests=["d_dimer"],
        gold_triage="urgent",
        prompt_version="v1-offline",
        policy_version="v1-deterministic",
        model_route="offline",
        steps=[TraceStep(name="extract", detail="demo")],
        report=report,
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


def test_detect_lightning_runtime_falls_back_to_export_mode_when_package_missing(monkeypatch) -> None:
    from agent import lightning_adapter

    monkeypatch.setattr(lightning_adapter, "_agentlightning_version", lambda: None)
    settings = Settings(_env_file=None, allow_live_llm=False)
    runtime = detect_lightning_runtime(settings)

    assert runtime.mode == "export_only"
    assert runtime.package_available is False


def test_render_prompt_template_formats_task_text() -> None:
    rendered = render_prompt_template("Case:\n{task}", "Example vignette")
    assert "Example vignette" in rendered


def test_export_lightning_bundle_writes_machine_readable_files(tmp_path: Path, monkeypatch) -> None:
    from agent import lightning_adapter

    monkeypatch.setattr(lightning_adapter, "_agentlightning_version", lambda: None)
    manifest = export_lightning_bundle(
        train_tasks=[_sample_task()],
        validation_tasks=[_sample_task("task-2")],
        traces=[_sample_trace()],
        report_markdown="# Demo",
        output_dir=tmp_path,
        settings=Settings(_env_file=None, allow_live_llm=False),
    )

    manifest_path = tmp_path / "lightning_bundle_manifest.json"
    transitions_path = tmp_path / "lightning_transitions.jsonl"
    train_path = tmp_path / "lightning_train_tasks.jsonl"

    assert manifest_path.exists()
    assert transitions_path.exists()
    assert train_path.exists()

    transition_lines = transitions_path.read_text(encoding="utf-8").splitlines()
    assert len(transition_lines) == 1
    transition_payload = json.loads(transition_lines[0])
    assert transition_payload["task_id"] == "task-1"
    assert transition_payload["state"]["curriculum_key"] == "broad_medical_feedback_lab"
    assert transition_payload["info"]["source_hf_dataset"] == "augtoma/medqa_usmle"

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["runtime"]["mode"] == "export_only"
    assert manifest_payload["baseline_policy_version"] == "v1-deterministic"
    assert manifest.train_tasks_path.endswith("lightning_train_tasks.jsonl")


def test_export_lightning_bundle_records_curriculum_metadata(tmp_path: Path, monkeypatch) -> None:
    from agent import lightning_adapter

    monkeypatch.setattr(lightning_adapter, "_agentlightning_version", lambda: None)
    manifest = export_lightning_bundle(
        train_tasks=[_sample_task()],
        validation_tasks=[_sample_task("task-2")],
        traces=[_sample_trace()],
        report_markdown="# Demo",
        output_dir=tmp_path,
        settings=Settings(_env_file=None, allow_live_llm=False),
        curriculum_key="broad_medical_feedback_lab",
        component_datasets=["medmcqa", "medqa", "pubmedqa", "findzebra"],
        reward_profiles=["diagnostic", "evidence_verification"],
    )

    assert manifest.curriculum_key == "broad_medical_feedback_lab"
    assert manifest.component_datasets == ["medmcqa", "medqa", "pubmedqa", "findzebra"]
    assert manifest.reward_profiles == ["diagnostic", "evidence_verification"]
