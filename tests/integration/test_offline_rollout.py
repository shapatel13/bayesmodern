from pathlib import Path

from agent.lightning_adapter import traces_to_lightning_transitions
from agent.offline_rollout import run_offline_rollout
from agent.orchestrator import PRIORIXOrchestrator
from agent.reward_model import CompositeRewardModel
from priorix_tasks.common import BenchmarkTask


def _sample_task() -> BenchmarkTask:
    return BenchmarkTask(
        task_id="bench-pe-1",
        source_dataset="synthetic",
        split="test",
        task_type="diagnosis_open",
        prompt="Pleuritic chest pain with tachycardia and hypoxemia, no fever.",
        gold_diagnosis="pe",
        acceptable_tests=["d_dimer", "cta_pe"],
        gold_triage="urgent",
    )


def test_orchestrator_and_reward_model_generate_consistent_trace() -> None:
    task = _sample_task()
    report = PRIORIXOrchestrator().analyze_text_case(task.task_id, task.prompt)
    reward = CompositeRewardModel().score(task, report)
    assert reward.total_reward > 0
    assert report.differential.ranked[0].slug == "pe"


def test_offline_rollout_writes_report_and_transitions(tmp_path: Path) -> None:
    traces, report = run_offline_rollout([_sample_task()], tmp_path)
    assert traces
    assert "PRIORI-X Benchmark Report" in report
    assert (tmp_path / "benchmark_report.md").exists()
    assert (tmp_path / "lightning_bundle_manifest.json").exists()
    assert (tmp_path / "lightning_train_tasks.jsonl").exists()
    assert (tmp_path / "lightning_transitions.jsonl").exists()
    transitions = traces_to_lightning_transitions(traces)
    assert transitions[0].task_id == "bench-pe-1"
