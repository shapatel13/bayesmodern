from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from agent.lightning_adapter import LightningRuntimeStatus
from eval.benchmark_runner import BenchmarkMetricsSummary
from utils.dates import utc_now
from utils.ids import make_id
from utils.jsonx import dumps_pretty


EXPERIMENT_SUMMARY_FILENAME = "experiment_summary.json"


class ExperimentSummary(BaseModel):
    experiment_id: str
    created_at: str
    dataset_key: str
    dataset_hf_id: str | None = None
    task_family: str
    subset: str | None = None
    train_split: str | None = None
    validation_split: str | None = None
    train_cases: int
    validation_cases: int
    prompt_version: str
    policy_version: str
    artifact_dir: str
    benchmark_summary: BenchmarkMetricsSummary
    lightning_runtime: LightningRuntimeStatus


class ExperimentComparison(BaseModel):
    baseline_experiment_id: str
    candidate_experiment_id: str
    delta_mean_reward: float
    delta_top1_differential_recall: float
    delta_top3_differential_recall: float
    delta_next_best_test_hit_rate: float
    delta_unsafe_recommendation_rate: float
    delta_brier_score: float
    delta_expected_calibration_error: float
    delta_log_loss: float
    promoted_dimensions: list[str]
    regressed_dimensions: list[str]


def create_experiment_dir(base_dir: Path, dataset_key: str) -> Path:
    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    normalized_dataset = "".join(char if char.isalnum() else "_" for char in dataset_key.lower()).strip("_")
    directory = base_dir / f"{stamp}_{normalized_dataset}_{make_id('exp')}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_experiment_summary(summary: ExperimentSummary, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / EXPERIMENT_SUMMARY_FILENAME
    destination.write_text(dumps_pretty(summary.model_dump()), encoding="utf-8")
    return destination


def read_experiment_summary(path: Path) -> ExperimentSummary:
    target = path / EXPERIMENT_SUMMARY_FILENAME if path.is_dir() else path
    return ExperimentSummary.model_validate_json(target.read_text(encoding="utf-8"))


def list_experiment_summaries(base_dir: Path) -> list[ExperimentSummary]:
    if not base_dir.exists():
        return []
    summaries = [read_experiment_summary(path) for path in base_dir.glob(f"*/{EXPERIMENT_SUMMARY_FILENAME}")]
    return sorted(summaries, key=lambda summary: summary.created_at, reverse=True)


def _metric_delta(
    baseline: float,
    candidate: float,
    *,
    preferred_direction: Literal["higher", "lower"],
) -> tuple[float, bool, bool]:
    delta = candidate - baseline
    if preferred_direction == "lower":
        promoted = candidate < baseline
        regressed = candidate > baseline
    else:
        promoted = candidate > baseline
        regressed = candidate < baseline
    return delta, promoted, regressed


def compare_experiment_summaries(
    baseline: ExperimentSummary,
    candidate: ExperimentSummary,
) -> ExperimentComparison:
    promoted_dimensions: list[str] = []
    regressed_dimensions: list[str] = []

    delta_mean_reward, promoted, regressed = _metric_delta(
        baseline.benchmark_summary.mean_reward,
        candidate.benchmark_summary.mean_reward,
        preferred_direction="higher",
    )
    if promoted:
        promoted_dimensions.append("mean_reward")
    if regressed:
        regressed_dimensions.append("mean_reward")

    delta_top1, promoted, regressed = _metric_delta(
        baseline.benchmark_summary.top1_differential_recall,
        candidate.benchmark_summary.top1_differential_recall,
        preferred_direction="higher",
    )
    if promoted:
        promoted_dimensions.append("top1_differential_recall")
    if regressed:
        regressed_dimensions.append("top1_differential_recall")

    delta_top3, promoted, regressed = _metric_delta(
        baseline.benchmark_summary.top3_differential_recall,
        candidate.benchmark_summary.top3_differential_recall,
        preferred_direction="higher",
    )
    if promoted:
        promoted_dimensions.append("top3_differential_recall")
    if regressed:
        regressed_dimensions.append("top3_differential_recall")

    delta_next_test, promoted, regressed = _metric_delta(
        baseline.benchmark_summary.next_best_test_hit_rate,
        candidate.benchmark_summary.next_best_test_hit_rate,
        preferred_direction="higher",
    )
    if promoted:
        promoted_dimensions.append("next_best_test_hit_rate")
    if regressed:
        regressed_dimensions.append("next_best_test_hit_rate")

    delta_unsafe, promoted, regressed = _metric_delta(
        baseline.benchmark_summary.unsafe_recommendation_rate,
        candidate.benchmark_summary.unsafe_recommendation_rate,
        preferred_direction="lower",
    )
    if promoted:
        promoted_dimensions.append("unsafe_recommendation_rate")
    if regressed:
        regressed_dimensions.append("unsafe_recommendation_rate")

    delta_brier, promoted, regressed = _metric_delta(
        baseline.benchmark_summary.brier_score,
        candidate.benchmark_summary.brier_score,
        preferred_direction="lower",
    )
    if promoted:
        promoted_dimensions.append("brier_score")
    if regressed:
        regressed_dimensions.append("brier_score")

    delta_ece, promoted, regressed = _metric_delta(
        baseline.benchmark_summary.expected_calibration_error,
        candidate.benchmark_summary.expected_calibration_error,
        preferred_direction="lower",
    )
    if promoted:
        promoted_dimensions.append("expected_calibration_error")
    if regressed:
        regressed_dimensions.append("expected_calibration_error")

    delta_log_loss, promoted, regressed = _metric_delta(
        baseline.benchmark_summary.log_loss,
        candidate.benchmark_summary.log_loss,
        preferred_direction="lower",
    )
    if promoted:
        promoted_dimensions.append("log_loss")
    if regressed:
        regressed_dimensions.append("log_loss")

    return ExperimentComparison(
        baseline_experiment_id=baseline.experiment_id,
        candidate_experiment_id=candidate.experiment_id,
        delta_mean_reward=delta_mean_reward,
        delta_top1_differential_recall=delta_top1,
        delta_top3_differential_recall=delta_top3,
        delta_next_best_test_hit_rate=delta_next_test,
        delta_unsafe_recommendation_rate=delta_unsafe,
        delta_brier_score=delta_brier,
        delta_expected_calibration_error=delta_ece,
        delta_log_loss=delta_log_loss,
        promoted_dimensions=promoted_dimensions,
        regressed_dimensions=regressed_dimensions,
    )
