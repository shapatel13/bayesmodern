from __future__ import annotations

from pathlib import Path

from agent.lightning_adapter import LightningRuntimeStatus
from eval.benchmark_runner import BenchmarkMetricsSummary
from eval.experiment_registry import (
    compare_experiment_summaries,
    list_experiment_summaries,
    write_experiment_summary,
)
from eval.experiment_registry import ExperimentSummary


def _summary(experiment_id: str, artifact_dir: Path, *, mean_reward: float, top1: float, unsafe: float) -> ExperimentSummary:
    return ExperimentSummary(
        experiment_id=experiment_id,
        created_at=f"2026-04-01T00:00:0{experiment_id[-1]}+00:00",
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
        artifact_dir=str(artifact_dir),
        benchmark_summary=BenchmarkMetricsSummary(
            cases=8,
            mean_reward=mean_reward,
            hard_veto_count=0,
            top1_differential_recall=top1,
            top3_differential_recall=0.9,
            next_best_test_hit_rate=0.7,
            unsafe_recommendation_rate=unsafe,
            unsupported_claim_rate=0.0,
            contradiction_rate=0.0,
            urgency_accuracy=0.8,
            brier_score=0.12,
            expected_calibration_error=0.08,
            log_loss=0.2,
            failure_categories={"wrong_primary_diagnosis": 1},
            top_diagnoses=["pe", "acs"],
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


def test_list_experiment_summaries_returns_newest_first(tmp_path: Path) -> None:
    older_dir = tmp_path / "older"
    newer_dir = tmp_path / "newer"
    write_experiment_summary(_summary("exp_1", older_dir, mean_reward=0.6, top1=0.4, unsafe=0.2), older_dir)
    write_experiment_summary(_summary("exp_2", newer_dir, mean_reward=0.8, top1=0.7, unsafe=0.1), newer_dir)

    summaries = list_experiment_summaries(tmp_path)

    assert [summary.experiment_id for summary in summaries] == ["exp_2", "exp_1"]


def test_compare_experiment_summaries_marks_promotions_and_regressions(tmp_path: Path) -> None:
    baseline = _summary("exp_1", tmp_path / "one", mean_reward=0.6, top1=0.4, unsafe=0.2)
    candidate = _summary("exp_2", tmp_path / "two", mean_reward=0.8, top1=0.7, unsafe=0.1)

    comparison = compare_experiment_summaries(baseline, candidate)

    assert comparison.delta_mean_reward > 0
    assert comparison.delta_top1_differential_recall > 0
    assert comparison.delta_unsafe_recommendation_rate < 0
    assert "mean_reward" in comparison.promoted_dimensions
    assert "unsafe_recommendation_rate" in comparison.promoted_dimensions
    assert comparison.regressed_dimensions == []
    assert comparison.promotion_gate is not None
    assert comparison.promotion_gate.verdict == "promote"
