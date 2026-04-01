from __future__ import annotations

from fastapi.testclient import TestClient

from agent.lightning_adapter import LightningRuntimeStatus
from agent.trace_schema import ExperimentTrace, RewardBreakdown, TraceStep
from apps.api.main import app
from eval.benchmark_runner import BenchmarkMetricsSummary
from eval.experiment_registry import ExperimentSummary
from llm.structured_output import ModelRoutingDecision, ResearchReport


def _sample_trace() -> ExperimentTrace:
    return ExperimentTrace(
        task_id="bench-pe-1",
        source_dataset="medmcqa",
        task_type="diagnosis_open",
        gold_diagnosis="pe",
        acceptable_tests=["d_dimer"],
        gold_triage="urgent",
        prompt_version="v1-offline",
        policy_version="v1-deterministic",
        model_route="offline",
        steps=[TraceStep(name="extract", detail="demo")],
        report=ResearchReport(
            context={
                "case_id": "bench-pe-1",
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
                    "score": 0.5,
                    "expected_information_gain": 0.2,
                    "expected_posterior_movement": 0.15,
                    "stewardship_score": 0.7,
                    "disposition": "worth_it_now",
                    "discriminates_between": ["pe", "pneumonia"],
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
                "clinician_language": "demo clinician framing",
                "plain_language": "demo plain-language framing",
            },
            contradictions=[],
            provenance_warnings=[],
            model_route=ModelRoutingDecision(
                parser_model="gpt-5.4-nano-2026-03-17",
                reasoning_model="gpt-5.4-nano-2026-03-17",
                verifier_model="gpt-5.4-nano-2026-03-17",
                mode="offline",
            ),
        ),
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


def _sample_experiment(experiment_id: str) -> ExperimentSummary:
    return ExperimentSummary(
        experiment_id=experiment_id,
        created_at="2026-04-01T12:00:00+00:00",
        dataset_key="medmcqa",
        dataset_hf_id="openlifescienceai/medmcqa",
        task_family="diagnosis_mcq",
        subset=None,
        train_split="train",
        validation_split="validation",
        train_cases=8,
        validation_cases=4,
        prompt_version="v1-offline",
        policy_version="v1-deterministic",
        artifact_dir=f"artifacts/evals/experiments/{experiment_id}",
        benchmark_summary=BenchmarkMetricsSummary(
            cases=8,
            mean_reward=0.8 if experiment_id == "exp_new" else 0.6,
            hard_veto_count=0,
            top1_differential_recall=0.7 if experiment_id == "exp_new" else 0.5,
            top3_differential_recall=0.9,
            next_best_test_hit_rate=0.8,
            unsafe_recommendation_rate=0.0 if experiment_id == "exp_new" else 0.1,
            unsupported_claim_rate=0.0,
            contradiction_rate=0.0,
            urgency_accuracy=0.9,
            brier_score=0.1,
            expected_calibration_error=0.05,
            log_loss=0.2,
            failure_categories={},
            top_diagnoses=["pe"],
        ),
        lightning_runtime=LightningRuntimeStatus(
            mode="export_only",
            package_available=True,
            package_version="0.2.1",
            platform_supported=False,
            native_training_ready=False,
            reason="Windows export-only test fixture.",
        ),
    )


def test_research_dataset_catalog_and_status_endpoints() -> None:
    client = TestClient(app)

    status_response = client.get("/api/research/status")
    datasets_response = client.get("/api/research/datasets")

    assert status_response.status_code == 200
    assert "lightning_runtime" in status_response.json()
    assert datasets_response.status_code == 200
    assert any(dataset["key"] == "medmcqa" for dataset in datasets_response.json()["datasets"])


def test_research_benchmark_and_experiment_endpoints(monkeypatch) -> None:
    from apps.api.routers import research

    sample_trace = _sample_trace()

    monkeypatch.setattr(research, "run_dataset_benchmark", lambda *args, **kwargs: [sample_trace])
    monkeypatch.setattr(
        research,
        "run_dataset_offline_experiment",
        lambda *args, **kwargs: (_sample_experiment("exp_new"), [sample_trace], "# Demo Report"),
    )
    monkeypatch.setattr(research, "list_experiment_summaries", lambda *args, **kwargs: [_sample_experiment("exp_old"), _sample_experiment("exp_new")])

    client = TestClient(app)

    benchmark_response = client.post("/api/research/benchmark/dataset", json={"dataset_key": "medmcqa", "limit": 1})
    rollout_response = client.post("/api/research/rollout/dataset", json={"dataset_key": "medmcqa", "train_limit": 1, "validation_limit": 1})
    list_response = client.get("/api/research/experiments")
    compare_response = client.post(
        "/api/research/experiments/compare",
        json={"baseline_experiment_id": "exp_old", "candidate_experiment_id": "exp_new"},
    )

    assert benchmark_response.status_code == 200
    assert benchmark_response.json()["summary"]["top1_differential_recall"] == 1.0
    assert rollout_response.status_code == 200
    assert rollout_response.json()["experiment"]["experiment_id"] == "exp_new"
    assert list_response.status_code == 200
    assert len(list_response.json()["experiments"]) == 2
    assert compare_response.status_code == 200
    assert compare_response.json()["comparison"]["delta_mean_reward"] > 0
