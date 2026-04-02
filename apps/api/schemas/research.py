from __future__ import annotations

from pydantic import BaseModel, Field

from agent.lightning_adapter import LightningRuntimeStatus
from datasets.curricula import CurriculumAccessMode
from eval.benchmark_runner import BenchmarkMetricsSummary
from eval.experiment_registry import ExperimentComparison, ExperimentSummary, PromotionGateDecision
from eval.presets import ResearchPreset


class DatasetCatalogItem(BaseModel):
    key: str
    hf_dataset: str | None = None
    default_subset: str | None = None
    default_train_split: str
    default_val_split: str
    default_eval_split: str
    description: str
    task_family: str
    requires_credentials: bool
    notes: str = ""


class ResearchStatusResponse(BaseModel):
    experiment_namespace: str
    artifacts_root: str
    dataset_count: int
    curriculum_count: int
    lightning_runtime: LightningRuntimeStatus


class DatasetCatalogResponse(BaseModel):
    datasets: list[DatasetCatalogItem]


class CurriculumComponentItem(BaseModel):
    dataset_key: str
    train_limit: int
    validation_limit: int
    subset: str | None = None
    train_split: str | None = None
    validation_split: str | None = None
    weight: int
    optional: bool
    notes: str = ""


class LightningCurriculumItem(BaseModel):
    key: str
    label: str
    description: str
    objective: str
    access_mode: CurriculumAccessMode
    notes: str = ""
    focus_areas: list[str]
    components: list[CurriculumComponentItem]


class CurriculumCatalogResponse(BaseModel):
    curricula: list[LightningCurriculumItem]


class ResearchPresetItem(BaseModel):
    key: str
    label: str
    dataset_key: str
    description: str
    clinical_mode: str
    task_family: str
    train_limit: int
    validation_limit: int
    subset: str | None = None
    prompt_version: str
    policy_version: str
    requires_credentials: bool
    notes: str = ""


class PresetCatalogResponse(BaseModel):
    presets: list[ResearchPresetItem]


class DatasetBenchmarkRequest(BaseModel):
    dataset_key: str = Field(description="Dataset registry key, for example `medmcqa` or `pubmedqa`.")
    split: str | None = Field(default=None, description="Override the default evaluation split.")
    subset: str | None = Field(default=None, description="Optional HF subset/config name.")
    limit: int = Field(default=5, ge=1, le=100, description="Maximum benchmark tasks to evaluate.")
    prompt_version: str = Field(default="v1-offline")
    policy_version: str = Field(default="v1-deterministic")


class DatasetBenchmarkResponse(BaseModel):
    dataset: DatasetCatalogItem
    summary: BenchmarkMetricsSummary
    report: str
    task_ids: list[str]
    lightning_runtime: LightningRuntimeStatus


class DatasetRolloutRequest(BaseModel):
    dataset_key: str
    subset: str | None = None
    train_limit: int = Field(default=8, ge=1, le=250)
    validation_limit: int = Field(default=4, ge=0, le=250)
    train_split: str | None = None
    validation_split: str | None = None
    prompt_version: str = Field(default="v1-offline")
    policy_version: str = Field(default="v1-deterministic")


class DatasetRolloutResponse(BaseModel):
    experiment: ExperimentSummary
    report: str
    task_ids: list[str]


class PresetRolloutRequest(BaseModel):
    preset_key: str


class CurriculumRolloutRequest(BaseModel):
    curriculum_key: str
    train_cap_per_component: int = Field(default=8, ge=1, le=250)
    validation_cap_per_component: int = Field(default=4, ge=0, le=250)
    prompt_version: str = Field(default="v1-offline")
    policy_version: str = Field(default="v1-deterministic")


class ExperimentListResponse(BaseModel):
    experiments: list[ExperimentSummary]


class ExperimentComparisonRequest(BaseModel):
    baseline_experiment_id: str
    candidate_experiment_id: str


class ExperimentComparisonResponse(BaseModel):
    comparison: ExperimentComparison


class PromotionGateResponse(BaseModel):
    decision: PromotionGateDecision
