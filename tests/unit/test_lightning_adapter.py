from __future__ import annotations

import json
from pathlib import Path

from agent.lightning_adapter import (
    create_lightning_rollout_agent,
    detect_lightning_runtime,
    export_lightning_bundle,
    render_prompt_template,
    resolve_lightning_training_config,
)
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
        mechanism_states={
            "ranked": [
                {
                    "slug": "thrombotic_ischemic_tendency",
                    "name": "Thrombotic / ischemic tendency",
                    "category": "vascular",
                    "prior": 0.2,
                    "posterior": 0.73,
                    "interval_low": 0.61,
                    "interval_high": 0.82,
                    "evidence_for": [],
                    "evidence_against": [],
                    "confidence_state": "moderately_uncertain",
                    "provenance_badges": ["source:hard-coded"],
                }
            ],
            "active_states": ["Thrombotic / ischemic tendency"],
            "mixed_physiology": False,
            "summary": "Dominant mechanism signal: Thrombotic / ischemic tendency.",
            "model_note": "demo mechanism note",
        },
        next_best_tests=[
            {
                "slug": "d_dimer",
                "name": "D-dimer",
                "score": 0.2,
                "expected_information_gain": 0.1,
                "expected_posterior_movement": 0.15,
                "mechanistic_information_gain": 0.12,
                "stewardship_score": 0.2,
                "disposition": "worth_it_now",
                "discriminates_between": ["pe"],
                "target_states": ["thrombotic_ischemic_tendency"],
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
        reasoning_runtime={
            "mode": "hybrid_open_world",
            "open_world_considered": True,
            "open_world_triggered": True,
            "gate_reason": "Base curated posterior was broad.",
            "base_top_diagnosis": "pe",
            "base_top_posterior": 0.41,
            "final_top_diagnosis": "pe",
            "final_top_posterior": 0.7,
            "open_world_hypothesis_count": 2,
            "open_world_test_count": 1,
            "notes": ["Converted open-world candidates back into deterministic scoring."],
        },
        decision_quality={
            "needs_clinician_review": True,
            "reasons": ["Broad differential remained after scoring."],
            "structured_signal_count": 3,
            "top_differential_gap": 0.08,
            "low_signal_case": False,
            "broad_differential": True,
            "mixed_mechanism_uncertainty": False,
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


def test_detect_lightning_runtime_reports_missing_apo_dependency(monkeypatch) -> None:
    from agent import lightning_adapter

    monkeypatch.setattr(lightning_adapter, "_agentlightning_version", lambda: "0.3.0")
    monkeypatch.setattr(lightning_adapter, "_apo_dependency_issue", lambda: "missing dependency `poml`")
    settings = Settings(
        _env_file=None,
        allow_live_llm=True,
        default_model_provider="openai",
        openai_api_key="test-key",
    )
    runtime = detect_lightning_runtime(settings)

    assert runtime.mode == "export_only"
    assert runtime.package_available is True
    assert runtime.native_training_ready is False
    assert "poml" in runtime.reason


def test_render_prompt_template_formats_task_text() -> None:
    rendered = render_prompt_template("Case:\n{task}", "Example vignette")
    assert "Example vignette" in rendered


def test_resolve_lightning_training_config_uses_balanced_defaults_and_dummy_tracer() -> None:
    class FakeAGL:
        class DummyTracer:
            pass

    config = resolve_lightning_training_config(
        agl=FakeAGL(),
        settings=Settings(
            _env_file=None,
            lightning_training_profile="balanced",
            lightning_disable_agentops=True,
            lightning_rollout_batch_timeout_sec=900,
        ),
        train_task_count=10,
        validation_task_count=6,
        requested_n_runners=4,
    )

    assert config.n_runners == 4
    assert config.apo_kwargs["beam_width"] == 2
    assert config.apo_kwargs["branch_factor"] == 2
    assert config.apo_kwargs["beam_rounds"] == 1
    assert config.apo_kwargs["run_initial_validation"] is False
    assert config.apo_kwargs["val_batch_size"] == 6
    assert config.apo_kwargs["gradient_batch_size"] == 4
    assert config.tracer.__class__.__name__ == "DummyTracer"


def test_resolve_lightning_training_config_fast_profile_clamps_parallelism() -> None:
    class FakeAGL:
        class DummyTracer:
            pass

    config = resolve_lightning_training_config(
        agl=FakeAGL(),
        settings=Settings(
            _env_file=None,
            lightning_training_profile="fast",
            lightning_disable_agentops=False,
            lightning_rollout_batch_timeout_sec=300,
        ),
        train_task_count=1,
        validation_task_count=20,
        requested_n_runners=8,
    )

    assert config.n_runners == 1
    assert config.tracer is None
    assert config.apo_kwargs["beam_width"] == 1
    assert config.apo_kwargs["branch_factor"] == 1
    assert config.apo_kwargs["beam_rounds"] == 1
    assert config.apo_kwargs["gradient_batch_size"] == 1
    assert config.apo_kwargs["val_batch_size"] == 4
    assert config.apo_kwargs["rollout_batch_timeout"] == 300


def test_lightning_rollout_agent_accepts_dict_task_payload(monkeypatch) -> None:
    from agent import lightning_adapter

    emitted_objects: list[dict[str, object]] = []
    emitted_rewards: list[float] = []

    class FakeAGL:
        @staticmethod
        def rollout(func):
            return func

        @staticmethod
        def emit_object(payload):
            emitted_objects.append(payload)

        @staticmethod
        def emit_reward(value):
            emitted_rewards.append(value)

    class FakeOrchestrator:
        def __init__(self, settings=None, policy_version: str = "v1-deterministic") -> None:
            self.policy_version = policy_version

        def analyze_text_case(self, case_id: str, note_text: str, *, policy_version: str | None = None, prompt_template: str | None = None):
            assert case_id == "task-1"
            assert "Pleuritic chest pain" in note_text
            assert prompt_template == "Case:\n{task}"
            return _sample_trace().report

    class FakeRewardModel:
        def score(self, task, report):
            assert isinstance(task, BenchmarkTask)

            class Reward:
                total_reward = 0.75
                failure_categories = ["demo"]

            return Reward()

    monkeypatch.setattr(lightning_adapter, "_import_agentlightning", lambda: FakeAGL())
    monkeypatch.setattr(lightning_adapter, "PRIORIXOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(lightning_adapter, "CompositeRewardModel", FakeRewardModel)

    rollout = create_lightning_rollout_agent(settings=Settings(_env_file=None))
    reward = rollout(_sample_task().model_dump(), prompt_template="Case:\n{task}")

    assert reward == 0.75
    assert emitted_objects[0]["task_id"] == "task-1"
    assert emitted_objects[1]["rendered_prompt_preview"].startswith("Case:")
    assert emitted_rewards == [0.75]


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
    assert transition_payload["state"]["reasoning_mode"] == "hybrid_open_world"
    assert transition_payload["state"]["needs_clinician_review"] is True
    assert transition_payload["info"]["source_hf_dataset"] == "augtoma/medqa_usmle"
    assert transition_payload["action"]["top_mechanism"] == "thrombotic_ischemic_tendency"
    assert transition_payload["action"]["recommended_test_target_states"] == ["thrombotic_ischemic_tendency"]
    assert transition_payload["info"]["mechanism_mixed_physiology"] is False
    assert transition_payload["info"]["decision_quality_reasons"] == ["Broad differential remained after scoring."]

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["runtime"]["mode"] == "export_only"
    assert manifest_payload["baseline_policy_version"] == "v1-deterministic"
    assert manifest_payload["prompt_version"] == "v1-offline"
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
