from __future__ import annotations

from datasets.catalog import BenchmarkDatasetSpec
from datasets.curricula import LightningCurriculumSpec
from eval.benchmark_runner import BenchmarkMetricsSummary
from eval.experiment_registry import ExperimentComparison, ExperimentSummary
from eval.presets import ResearchPreset


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
            "Source": summary.dataset_key,
            "Kind": summary.source_kind,
            "Task Family": summary.task_family,
            "Prompt": summary.prompt_version,
            "Policy": summary.policy_version,
            "Train Cases": summary.train_cases,
            "Val Cases": summary.validation_cases,
            "Mean Reward": round(summary.benchmark_summary.mean_reward, 3),
            "Top-1 Recall": round(summary.benchmark_summary.top1_differential_recall, 3),
            "Urgency Acc": round(summary.benchmark_summary.urgency_accuracy, 3),
            "Med Recall": round(summary.benchmark_summary.medication_recall, 3),
            "ADE Recall": round(summary.benchmark_summary.adverse_event_recall, 3),
            "Audit Acc": round(summary.benchmark_summary.generation_risk_accuracy, 3),
            "Unsafe Rate": round(summary.benchmark_summary.unsafe_recommendation_rate, 3),
            "Lightning": summary.lightning_runtime.mode,
        }
        for summary in summaries
    ]


def benchmark_summary_cards(summary: BenchmarkMetricsSummary) -> list[tuple[str, str]]:
    cards = [
        ("Cases", str(summary.cases)),
        ("Mean Reward", f"{summary.mean_reward:.3f}"),
    ]
    if summary.medication_recall > 0 or summary.adverse_event_recall > 0:
        cards.extend(
            [
                ("Med Recall", f"{summary.medication_recall:.1%}"),
                ("ADE Recall", f"{summary.adverse_event_recall:.1%}"),
            ]
        )
    elif summary.generation_risk_accuracy > 0 or summary.generation_high_risk_recall > 0:
        cards.extend(
            [
                ("Audit Acc", f"{summary.generation_risk_accuracy:.1%}"),
                ("High-Risk Recall", f"{summary.generation_high_risk_recall:.1%}"),
            ]
        )
    elif summary.urgency_accuracy > 0:
        cards.extend(
            [
                ("Urgency Acc", f"{summary.urgency_accuracy:.1%}"),
                ("Top-1 Recall", f"{summary.top1_differential_recall:.1%}"),
            ]
        )
    else:
        cards.extend(
            [
                ("Top-1 Recall", f"{summary.top1_differential_recall:.1%}"),
                ("Top-3 Recall", f"{summary.top3_differential_recall:.1%}"),
            ]
        )
    cards.extend(
        [
            ("Unsafe Rate", f"{summary.unsafe_recommendation_rate:.1%}"),
            ("ECE", f"{summary.expected_calibration_error:.3f}"),
        ]
    )
    return cards


def comparison_rows(comparison: ExperimentComparison) -> list[dict[str, object]]:
    rows = [
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
    if comparison.delta_medication_recall or comparison.delta_adverse_event_recall:
        rows.extend(
            [
                {
                    "Metric": "Medication Recall",
                    "Delta": round(comparison.delta_medication_recall, 3),
                    "Preferred Direction": "Higher",
                },
                {
                    "Metric": "Adverse-Event Recall",
                    "Delta": round(comparison.delta_adverse_event_recall, 3),
                    "Preferred Direction": "Higher",
                },
            ]
        )
    return rows


def preset_rows(presets: list[ResearchPreset]) -> list[dict[str, object]]:
    return [
        {
            "Preset": preset.label,
            "Key": preset.key,
            "Dataset": preset.dataset_key,
            "Mode": preset.clinical_mode.replace("_", " "),
            "Task Family": preset.task_family,
            "Train": preset.train_limit,
            "Validation": preset.validation_limit,
            "Credentialed": "Yes" if preset.requires_credentials else "No",
            "Notes": preset.notes or preset.description,
        }
        for preset in presets
    ]


def curriculum_rows(curricula: list[LightningCurriculumSpec]) -> list[dict[str, object]]:
    return [
        {
            "Curriculum": curriculum.label,
            "Key": curriculum.key,
            "Access": curriculum.access_mode.replace("_", " "),
            "Datasets": ", ".join(component.dataset_key for component in curriculum.components),
            "Objective": curriculum.objective,
            "Focus Areas": ", ".join(curriculum.focus_areas),
            "Notes": curriculum.notes or curriculum.description,
        }
        for curriculum in curricula
    ]
