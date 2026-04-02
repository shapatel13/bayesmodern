from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from agent.lightning_adapter import detect_lightning_runtime
from agent.lightning_train import auto_improve_prompt, train_curriculum_prompt, train_dataset_prompt
from agent.prompt_registry import get_active_prompt_record, list_prompt_records, promote_prompt_version
from agent.offline_rollout import run_curriculum_offline_experiment, run_dataset_offline_experiment
from agent.policy_optimizer import optimize_curriculum_policy, optimize_dataset_policy
from apps.api.schemas.research import (
    CurriculumCatalogResponse,
    CurriculumComponentItem,
    CurriculumPromptTrainingRequest,
    CurriculumPolicyOptimizationRequest,
    CurriculumRolloutRequest,
    DatasetBenchmarkRequest,
    DatasetBenchmarkResponse,
    DatasetCatalogItem,
    DatasetCatalogResponse,
    DatasetPromptTrainingRequest,
    DatasetPolicyOptimizationRequest,
    DatasetRolloutRequest,
    DatasetRolloutResponse,
    ExperimentComparisonRequest,
    ExperimentComparisonResponse,
    ExperimentListResponse,
    PromptCatalogResponse,
    PromptPromotionRequest,
    PromptTrainingResponse,
    PresetCatalogResponse,
    PresetRolloutRequest,
    PolicyCatalogResponse,
    PolicyOptimizationResponse,
    ResearchPresetItem,
    ResearchStatusResponse,
    ReasoningPolicyItem,
    LightningCurriculumItem,
)
from core.policy import ReasoningPolicy, list_reasoning_policies
from datasets.catalog import BenchmarkDatasetSpec, get_dataset_spec, list_dataset_specs
from datasets.curricula import LightningCurriculumSpec, get_lightning_curriculum, list_lightning_curricula
from eval.benchmark_runner import build_markdown_report, run_dataset_benchmark, summarize_benchmark
from eval.experiment_registry import compare_experiment_summaries, list_experiment_summaries
from eval.presets import ResearchPreset, get_research_preset, list_research_presets
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


def _preset_item(preset: ResearchPreset) -> ResearchPresetItem:
    return ResearchPresetItem(
        key=preset.key,
        label=preset.label,
        dataset_key=preset.dataset_key,
        description=preset.description,
        clinical_mode=preset.clinical_mode,
        task_family=preset.task_family,
        train_limit=preset.train_limit,
        validation_limit=preset.validation_limit,
        subset=preset.subset,
        prompt_version=preset.prompt_version,
        policy_version=preset.policy_version,
        requires_credentials=preset.requires_credentials,
        notes=preset.notes,
    )


def _curriculum_item(curriculum: LightningCurriculumSpec) -> LightningCurriculumItem:
    return LightningCurriculumItem(
        key=curriculum.key,
        label=curriculum.label,
        description=curriculum.description,
        objective=curriculum.objective,
        access_mode=curriculum.access_mode,
        notes=curriculum.notes,
        focus_areas=list(curriculum.focus_areas),
        components=[
            CurriculumComponentItem(
                dataset_key=component.dataset_key,
                train_limit=component.train_limit,
                validation_limit=component.validation_limit,
                subset=component.subset,
                train_split=component.train_split,
                validation_split=component.validation_split,
                weight=component.weight,
                optional=component.optional,
                notes=component.notes,
            )
            for component in curriculum.components
        ],
    )


def _policy_item(policy: ReasoningPolicy) -> ReasoningPolicyItem:
    return ReasoningPolicyItem(
        version=policy.version,
        label=policy.label,
        description=policy.description,
        differential=policy.differential,
        next_test=policy.next_test,
        thresholds=policy.thresholds,
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
        curriculum_count=len(list_lightning_curricula()),
        active_prompt_version=get_active_prompt_record().version,
        lightning_runtime=detect_lightning_runtime(settings),
    )


@router.get("/research/datasets", response_model=DatasetCatalogResponse)
def research_datasets() -> DatasetCatalogResponse:
    return DatasetCatalogResponse(datasets=[_dataset_item(spec) for spec in list_dataset_specs()])


@router.get("/research/curricula", response_model=CurriculumCatalogResponse)
def research_curricula() -> CurriculumCatalogResponse:
    return CurriculumCatalogResponse(curricula=[_curriculum_item(curriculum) for curriculum in list_lightning_curricula()])


@router.get("/research/policies", response_model=PolicyCatalogResponse)
def research_policies() -> PolicyCatalogResponse:
    return PolicyCatalogResponse(policies=[_policy_item(policy) for policy in list_reasoning_policies()])


@router.get("/research/prompts", response_model=PromptCatalogResponse)
def research_prompts() -> PromptCatalogResponse:
    return PromptCatalogResponse(
        active_prompt=get_active_prompt_record(),
        prompts=list_prompt_records(),
    )


@router.get("/research/presets", response_model=PresetCatalogResponse)
def research_presets() -> PresetCatalogResponse:
    return PresetCatalogResponse(presets=[_preset_item(preset) for preset in list_research_presets()])


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


@router.post("/research/optimize/dataset-policy", response_model=PolicyOptimizationResponse)
def optimize_dataset_policy_route(request: DatasetPolicyOptimizationRequest) -> PolicyOptimizationResponse:
    try:
        optimization = optimize_dataset_policy(
            request.dataset_key,
            ARTIFACTS_ROOT,
            subset=request.subset,
            train_limit=request.train_limit,
            validation_limit=request.validation_limit,
            train_split=request.train_split,
            validation_split=request.validation_split,
            settings=get_settings(),
            prompt_version=request.prompt_version,
            baseline_policy_version=request.baseline_policy_version,
            candidate_policy_versions=request.candidate_policy_versions or None,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Dataset policy optimization failed: {exc}") from exc
    return PolicyOptimizationResponse(optimization=optimization)


@router.post("/research/optimize/curriculum-policy", response_model=PolicyOptimizationResponse)
def optimize_curriculum_policy_route(request: CurriculumPolicyOptimizationRequest) -> PolicyOptimizationResponse:
    try:
        optimization = optimize_curriculum_policy(
            request.curriculum_key,
            ARTIFACTS_ROOT,
            settings=get_settings(),
            prompt_version=request.prompt_version,
            baseline_policy_version=request.baseline_policy_version,
            candidate_policy_versions=request.candidate_policy_versions or None,
            train_cap_per_component=request.train_cap_per_component,
            validation_cap_per_component=request.validation_cap_per_component,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Curriculum policy optimization failed: {exc}") from exc
    return PolicyOptimizationResponse(optimization=optimization)


@router.post("/research/prompts/promote", response_model=PromptCatalogResponse)
def promote_prompt_route(request: PromptPromotionRequest) -> PromptCatalogResponse:
    try:
        registry = promote_prompt_version(
            request.prompt_version,
            notes=["Promoted manually through the research API."],
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PromptCatalogResponse(
        active_prompt=get_active_prompt_record(),
        prompts=registry.prompts,
    )


@router.post("/research/train/dataset-prompt", response_model=PromptTrainingResponse)
def train_dataset_prompt_route(request: DatasetPromptTrainingRequest) -> PromptTrainingResponse:
    try:
        training = train_dataset_prompt(
            request.dataset_key,
            ARTIFACTS_ROOT,
            subset=request.subset,
            train_limit=request.train_limit,
            validation_limit=request.validation_limit,
            train_split=request.train_split,
            validation_split=request.validation_split,
            settings=get_settings(),
            prompt_version=request.prompt_version,
            policy_version=request.policy_version,
            n_runners=request.n_runners,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Dataset prompt training failed: {exc}") from exc
    return PromptTrainingResponse(training=training)


@router.post("/research/train/curriculum-prompt", response_model=PromptTrainingResponse)
def train_curriculum_prompt_route(request: CurriculumPromptTrainingRequest) -> PromptTrainingResponse:
    try:
        training = train_curriculum_prompt(
            request.curriculum_key,
            ARTIFACTS_ROOT,
            settings=get_settings(),
            prompt_version=request.prompt_version,
            policy_version=request.policy_version,
            train_cap_per_component=request.train_cap_per_component,
            validation_cap_per_component=request.validation_cap_per_component,
            n_runners=request.n_runners,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Curriculum prompt training failed: {exc}") from exc
    return PromptTrainingResponse(training=training)


@router.post("/research/train/auto-improve", response_model=PromptTrainingResponse)
def auto_improve_prompt_route() -> PromptTrainingResponse:
    try:
        training = auto_improve_prompt(ARTIFACTS_ROOT, settings=get_settings())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Auto-improvement failed: {exc}") from exc
    return PromptTrainingResponse(training=training)


@router.post("/research/rollout/preset", response_model=DatasetRolloutResponse)
def rollout_preset(request: PresetRolloutRequest) -> DatasetRolloutResponse:
    try:
        preset = get_research_preset(request.preset_key)
        experiment_summary, traces, report = run_dataset_offline_experiment(
            preset.dataset_key,
            ARTIFACTS_ROOT,
            subset=preset.subset,
            train_limit=preset.train_limit,
            validation_limit=preset.validation_limit,
            settings=get_settings(),
            prompt_version=preset.prompt_version,
            policy_version=preset.policy_version,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Preset rollout failed: {exc}") from exc

    return DatasetRolloutResponse(
        experiment=experiment_summary,
        report=report,
        task_ids=[trace.task_id for trace in traces],
    )


@router.post("/research/rollout/curriculum", response_model=DatasetRolloutResponse)
def rollout_curriculum(request: CurriculumRolloutRequest) -> DatasetRolloutResponse:
    try:
        _curriculum = get_lightning_curriculum(request.curriculum_key)
        experiment_summary, traces, report = run_curriculum_offline_experiment(
            request.curriculum_key,
            ARTIFACTS_ROOT,
            settings=get_settings(),
            prompt_version=request.prompt_version,
            policy_version=request.policy_version,
            train_cap_per_component=request.train_cap_per_component,
            validation_cap_per_component=request.validation_cap_per_component,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Curriculum rollout failed: {exc}") from exc

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
