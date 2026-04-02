from __future__ import annotations

import json
from pathlib import Path

from agent.lightning_adapter import LightningRuntimeStatus
from agent.policy_optimizer import optimize_dataset_policy
from eval.benchmark_runner import BenchmarkMetricsSummary
from eval.experiment_registry import ExperimentSummary


def _runtime() -> LightningRuntimeStatus:
    return LightningRuntimeStatus(
        mode="export_only",
        package_available=False,
        package_version=None,
        platform_supported=False,
        native_training_ready=False,
        reason="Test export-only runtime.",
    )


def _summary(policy_version: str, artifact_dir: Path, *, mean_reward: float, unsafe_rate: float) -> ExperimentSummary:
    return ExperimentSummary(
        experiment_id=f"exp_{policy_version}",
        created_at="2026-04-02T12:00:00+00:00",
        dataset_key="medmcqa",
        dataset_hf_id="openlifescienceai/medmcqa",
        task_family="diagnosis_mcq",
        subset=None,
        train_split="train",
        validation_split="validation",
        train_cases=2,
        validation_cases=1,
        prompt_version="v1-offline",
        policy_version=policy_version,
        artifact_dir=str(artifact_dir),
        benchmark_summary=BenchmarkMetricsSummary(
            cases=2,
            mean_reward=mean_reward,
            hard_veto_count=0,
            top1_differential_recall=0.5,
            top3_differential_recall=1.0,
            next_best_test_hit_rate=0.5,
            unsafe_recommendation_rate=unsafe_rate,
            unsupported_claim_rate=0.0,
            contradiction_rate=0.0,
            urgency_accuracy=1.0,
            brier_score=0.1,
            expected_calibration_error=0.05,
            log_loss=0.2,
            failure_categories={},
            top_diagnoses=["pe"],
        ),
        lightning_runtime=_runtime(),
    )


def test_optimize_dataset_policy_promotes_best_safe_candidate(monkeypatch, tmp_path: Path) -> None:
    from agent import policy_optimizer
    from priorix_tasks.common import BenchmarkTask

    dataset_pair = type(
        "DatasetPair",
        (),
        {
            "train": [
                BenchmarkTask(
                    task_id="case-1",
                    source_dataset="medmcqa",
                    split="train",
                    task_type="diagnosis_mcq",
                    prompt="Chest pain and tachycardia.",
                    gold_diagnosis="pe",
                    acceptable_tests=["d_dimer"],
                    gold_triage="urgent",
                )
            ],
            "validation": [],
        },
    )()

    monkeypatch.setattr(policy_optimizer, "load_train_validation_tasks", lambda *args, **kwargs: dataset_pair)
    monkeypatch.setattr(policy_optimizer, "detect_lightning_runtime", lambda *args, **kwargs: _runtime())
    monkeypatch.setattr(
        policy_optimizer,
        "get_dataset_spec",
        lambda key: type("Spec", (), {"hf_dataset": "openlifescienceai/medmcqa", "task_family": "diagnosis_mcq", "default_subset": None})(),
    )

    def fake_run_offline_rollout(tasks, artifact_dir, **kwargs):
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return [], f"# {kwargs['policy_version']}"

    monkeypatch.setattr(policy_optimizer, "run_offline_rollout", fake_run_offline_rollout)

    def fake_summary_builder(*, spec, dataset_key, tasks, validation_tasks, traces, artifact_dir, prompt_version, policy_version, settings):
        rewards = {
            "v1-deterministic": (0.50, 0.00),
            "v1-balanced-bayesian": (0.72, 0.00),
            "v1-conservative-safety": (0.69, 0.00),
            "v1-stewardship": (0.62, 0.05),
            "v1-sensitive-triage": (0.60, 0.10),
        }
        mean_reward, unsafe_rate = rewards[policy_version]
        return _summary(policy_version, artifact_dir, mean_reward=mean_reward, unsafe_rate=unsafe_rate)

    monkeypatch.setattr(policy_optimizer, "_build_dataset_summary", fake_summary_builder)

    optimization = optimize_dataset_policy(
        "medmcqa",
        tmp_path,
        baseline_policy_version="v1-deterministic",
    )

    assert optimization.selected_policy_version == "v1-balanced-bayesian"
    assert optimization.lightning_runtime_mode == "export_only"
    summary_path = Path(optimization.artifact_dir) / "policy_optimization_summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["selected_policy_version"] == "v1-balanced-bayesian"
