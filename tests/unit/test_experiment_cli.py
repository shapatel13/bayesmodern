from __future__ import annotations

import json
from pathlib import Path

from agent.lightning_adapter import LightningRuntimeStatus
from eval.benchmark_runner import BenchmarkMetricsSummary
from eval.experiment_cli import main
from eval.experiment_registry import ExperimentSummary


def _sample_summary(experiment_id: str) -> ExperimentSummary:
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
        artifact_dir=str(Path("artifacts/evals/experiments") / experiment_id),
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


def test_experiment_cli_status_prints_runtime(capsys) -> None:
    exit_code = main(["status"])

    captured = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert "lightning_runtime" in captured
    assert "medmcqa" in captured["datasets"]


def test_experiment_cli_compare_prints_deltas(monkeypatch, capsys) -> None:
    from eval import experiment_cli

    monkeypatch.setattr(
        experiment_cli,
        "list_experiment_summaries",
        lambda *args, **kwargs: [_sample_summary("exp_old"), _sample_summary("exp_new")],
    )

    exit_code = main(["compare-experiments", "exp_old", "exp_new"])

    captured = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert captured["delta_mean_reward"] > 0
