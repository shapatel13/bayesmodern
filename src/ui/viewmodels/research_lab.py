from __future__ import annotations

from datasets.catalog import BenchmarkDatasetSpec
from eval.benchmark_runner import BenchmarkMetricsSummary
from eval.experiment_registry import ExperimentComparison, ExperimentSummary


def dataset_catalog_rows(specs: list[BenchmarkDatasetSpec]) -> list[dict[str, object]]:
    return [
        {
            "Dataset": spec.key,
            "HF ID": spec.hf_dataset or "credentialed/local",
            "Task Family": spec.task_family,
            "Eval Split": spec.default_eval_split,
            "Subset": spec.default_subset or "-",
            "Notes": spec.notes or spec.description,
        }
        for spec in specs
    ]


def experiment_rows(summaries: list[ExperimentSummary]) -> list[dict[str, object]]:
    return [
        {
            "Experiment": summary.experiment_id,
            "Created": summary.created_at,
            "Dataset": summary.dataset_key,
            "Prompt": summary.prompt_version,
            "Policy": summary.policy_version,
            "Train Cases": summary.train_cases,
            "Val Cases": summary.validation_cases,
            "Mean Reward": round(summary.benchmark_summary.mean_reward, 3),
            "Top-1 Recall": round(summary.benchmark_summary.top1_differential_recall, 3),
            "Unsafe Rate": round(summary.benchmark_summary.unsafe_recommendation_rate, 3),
            "Lightning": summary.lightning_runtime.mode,
        }
        for summary in summaries
    ]


def benchmark_summary_cards(summary: BenchmarkMetricsSummary) -> list[tuple[str, str]]:
    return [
        ("Cases", str(summary.cases)),
        ("Mean Reward", f"{summary.mean_reward:.3f}"),
        ("Top-1 Recall", f"{summary.top1_differential_recall:.1%}"),
        ("Top-3 Recall", f"{summary.top3_differential_recall:.1%}"),
        ("Unsafe Rate", f"{summary.unsafe_recommendation_rate:.1%}"),
        ("ECE", f"{summary.expected_calibration_error:.3f}"),
    ]


def comparison_rows(comparison: ExperimentComparison) -> list[dict[str, object]]:
    return [
        {"Metric": "Mean Reward", "Delta": round(comparison.delta_mean_reward, 3), "Preferred Direction": "Higher"},
        {
            "Metric": "Top-1 Differential Recall",
            "Delta": round(comparison.delta_top1_differential_recall, 3),
            "Preferred Direction": "Higher",
        },
        {
            "Metric": "Top-3 Differential Recall",
            "Delta": round(comparison.delta_top3_differential_recall, 3),
            "Preferred Direction": "Higher",
        },
        {
            "Metric": "Next-Best-Test Hit Rate",
            "Delta": round(comparison.delta_next_best_test_hit_rate, 3),
            "Preferred Direction": "Higher",
        },
        {
            "Metric": "Unsafe Recommendation Rate",
            "Delta": round(comparison.delta_unsafe_recommendation_rate, 3),
            "Preferred Direction": "Lower",
        },
        {"Metric": "Brier Score", "Delta": round(comparison.delta_brier_score, 3), "Preferred Direction": "Lower"},
        {
            "Metric": "Expected Calibration Error",
            "Delta": round(comparison.delta_expected_calibration_error, 3),
            "Preferred Direction": "Lower",
        },
        {"Metric": "Log Loss", "Delta": round(comparison.delta_log_loss, 3), "Preferred Direction": "Lower"},
    ]
