from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from agent.lightning_adapter import detect_lightning_runtime
from agent.offline_rollout import run_dataset_offline_experiment
from apps.api.schemas.research import (
    DatasetBenchmarkRequest,
    DatasetBenchmarkResponse,
    DatasetCatalogItem,
    DatasetCatalogResponse,
    DatasetRolloutRequest,
    DatasetRolloutResponse,
    ExperimentComparisonRequest,
    ExperimentComparisonResponse,
    ExperimentListResponse,
    ResearchStatusResponse,
)
from datasets.catalog import BenchmarkDatasetSpec, get_dataset_spec, list_dataset_specs
from eval.benchmark_runner import build_markdown_report, run_dataset_benchmark, summarize_benchmark
from eval.experiment_registry import compare_experiment_summaries, list_experiment_summaries
from utils.config import get_settings


ARTIFACTS_ROOT = Path("artifacts/evals/experiments")

router = APIRouter(tags=["research"])


def _dataset_item(spec: BenchmarkDatasetSpec) -> DatasetCatalogItem:
    return DatasetCatalogItem(
        key=spec.key,
        hf_dataset=spec.hf_dataset,
        default_subset=spec.default_subset,
        default_train_split=spec.default_train_split,
        default_val_split=spec.default_val_split,
        default_eval_split=spec.default_eval_split,
        description=spec.description,
        task_family=spec.task_family,
        requires_credentials=spec.requires_credentials,
        notes=spec.notes,
    )


def _resolve_experiment_or_404(experiment_id: str):
    for summary in list_experiment_summaries(ARTIFACTS_ROOT):
        if summary.experiment_id == experiment_id:
            return summary
    raise HTTPException(status_code=404, detail=f"Experiment `{experiment_id}` was not found.")


@router.get("/research/status", response_model=ResearchStatusResponse)
def research_status() -> ResearchStatusResponse:
    settings = get_settings()
    return ResearchStatusResponse(
        experiment_namespace=settings.experiment_namespace,
        artifacts_root=str(ARTIFACTS_ROOT),
        dataset_count=len(list_dataset_specs()),
        lightning_runtime=detect_lightning_runtime(settings),
    )


@router.get("/research/datasets", response_model=DatasetCatalogResponse)
def research_datasets() -> DatasetCatalogResponse:
    return DatasetCatalogResponse(datasets=[_dataset_item(spec) for spec in list_dataset_specs()])


@router.post("/research/benchmark/dataset", response_model=DatasetBenchmarkResponse)
def benchmark_dataset(request: DatasetBenchmarkRequest) -> DatasetBenchmarkResponse:
    try:
        spec = get_dataset_spec(request.dataset_key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        traces = run_dataset_benchmark(
            request.dataset_key,
            split=request.split,
            subset=request.subset,
            limit=request.limit,
            prompt_version=request.prompt_version,
            policy_version=request.policy_version,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Benchmark run failed: {exc}") from exc

    return DatasetBenchmarkResponse(
        dataset=_dataset_item(spec),
        summary=summarize_benchmark(traces),
        report=build_markdown_report(traces),
        task_ids=[trace.task_id for trace in traces],
        lightning_runtime=detect_lightning_runtime(),
    )


@router.post("/research/rollout/dataset", response_model=DatasetRolloutResponse)
def rollout_dataset(request: DatasetRolloutRequest) -> DatasetRolloutResponse:
    try:
        experiment_summary, traces, report = run_dataset_offline_experiment(
            request.dataset_key,
            ARTIFACTS_ROOT,
            subset=request.subset,
            train_limit=request.train_limit,
            validation_limit=request.validation_limit,
            train_split=request.train_split,
            validation_split=request.validation_split,
            prompt_version=request.prompt_version,
            policy_version=request.policy_version,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Offline rollout failed: {exc}") from exc

    return DatasetRolloutResponse(
        experiment=experiment_summary,
        report=report,
        task_ids=[trace.task_id for trace in traces],
    )


@router.get("/research/experiments", response_model=ExperimentListResponse)
def list_experiments() -> ExperimentListResponse:
    return ExperimentListResponse(experiments=list_experiment_summaries(ARTIFACTS_ROOT))


@router.post("/research/experiments/compare", response_model=ExperimentComparisonResponse)
def compare_experiments(request: ExperimentComparisonRequest) -> ExperimentComparisonResponse:
    baseline = _resolve_experiment_or_404(request.baseline_experiment_id)
    candidate = _resolve_experiment_or_404(request.candidate_experiment_id)
    return ExperimentComparisonResponse(comparison=compare_experiment_summaries(baseline, candidate))
