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
    delta_medication_recall: float = 0.0
    delta_adverse_event_recall: float = 0.0
    promoted_dimensions: list[str]
    regressed_dimensions: list[str]
    promotion_gate: "PromotionGateDecision | None" = None


class PromotionGateDecision(BaseModel):
    verdict: Literal["promote", "hold", "reject"]
    blockers: list[str]
    rationale: list[str]
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

    delta_medication_recall, promoted, regressed = _metric_delta(
        baseline.benchmark_summary.medication_recall,
        candidate.benchmark_summary.medication_recall,
        preferred_direction="higher",
    )
    if promoted:
        promoted_dimensions.append("medication_recall")
    if regressed:
        regressed_dimensions.append("medication_recall")

    delta_adverse_event_recall, promoted, regressed = _metric_delta(
        baseline.benchmark_summary.adverse_event_recall,
        candidate.benchmark_summary.adverse_event_recall,
        preferred_direction="higher",
    )
    if promoted:
        promoted_dimensions.append("adverse_event_recall")
    if regressed:
        regressed_dimensions.append("adverse_event_recall")

    comparison = ExperimentComparison(
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
        delta_medication_recall=delta_medication_recall,
        delta_adverse_event_recall=delta_adverse_event_recall,
        promoted_dimensions=promoted_dimensions,
        regressed_dimensions=regressed_dimensions,
    )
    comparison.promotion_gate = evaluate_promotion_gate(baseline, candidate, comparison)
    return comparison


def evaluate_promotion_gate(
    baseline: ExperimentSummary,
    candidate: ExperimentSummary,
    comparison: ExperimentComparison,
) -> PromotionGateDecision:
    blockers: list[str] = []
    rationale: list[str] = []

    if candidate.benchmark_summary.unsafe_recommendation_rate > baseline.benchmark_summary.unsafe_recommendation_rate:
        blockers.append("unsafe recommendation rate regressed")
    if candidate.benchmark_summary.contradiction_rate > baseline.benchmark_summary.contradiction_rate:
        blockers.append("contradiction rate regressed")
    if candidate.benchmark_summary.unsupported_claim_rate > baseline.benchmark_summary.unsupported_claim_rate:
        blockers.append("unsupported claim rate regressed")
    if candidate.benchmark_summary.hard_veto_count > baseline.benchmark_summary.hard_veto_count:
        blockers.append("hard safety veto count regressed")

    if candidate.task_family == "triage" and candidate.benchmark_summary.urgency_accuracy < baseline.benchmark_summary.urgency_accuracy:
        blockers.append("triage accuracy regressed")
    if candidate.task_family == "medication_safety":
        if candidate.benchmark_summary.medication_recall < baseline.benchmark_summary.medication_recall:
            blockers.append("medication extraction recall regressed")
        if candidate.benchmark_summary.adverse_event_recall < baseline.benchmark_summary.adverse_event_recall:
            blockers.append("adverse-event recall regressed")

    if comparison.delta_mean_reward > 0:
        rationale.append("mean reward improved")
    if comparison.delta_top1_differential_recall > 0:
        rationale.append("top-1 differential recall improved")
    if comparison.delta_next_best_test_hit_rate > 0:
        rationale.append("next-best-test hit rate improved")
    if comparison.delta_medication_recall > 0:
        rationale.append("medication extraction recall improved")
    if comparison.delta_adverse_event_recall > 0:
        rationale.append("adverse-event recall improved")
    if comparison.delta_unsafe_recommendation_rate < 0:
        rationale.append("unsafe recommendation rate decreased")

    if blockers:
        verdict: Literal["promote", "hold", "reject"] = "reject"
    elif rationale:
        verdict = "promote"
    else:
        verdict = "hold"

    return PromotionGateDecision(
        verdict=verdict,
        blockers=blockers,
        rationale=rationale,
        promoted_dimensions=comparison.promoted_dimensions,
        regressed_dimensions=comparison.regressed_dimensions,
    )
