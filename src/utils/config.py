from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


SafetyMode = Literal["conservative", "standard", "research"]


class Settings(BaseSettings):
    app_name: str = Field(default="PRIORI-X", alias="PRIORI_APP_NAME")
    environment: str = Field(default="development", alias="PRIORI_ENV")
    log_level: str = Field(default="INFO", alias="PRIORI_LOG_LEVEL")
    safety_mode: SafetyMode = Field(default="conservative", alias="PRIORI_SAFETY_MODE")
    allow_live_llm: bool = Field(default=False, alias="PRIORI_ALLOW_LIVE_LLM")
    default_model_provider: str = Field(default="offline", alias="PRIORI_DEFAULT_MODEL_PROVIDER")
    default_reasoner: str = Field(default="hybrid-bayesian", alias="PRIORI_DEFAULT_REASONER")
    openai_parser_model: str = Field(
        default="gpt-5.4-nano-2026-03-17", alias="PRIORI_OPENAI_PARSER_MODEL"
    )
    openai_reasoning_model: str = Field(
        default="gpt-5.4-nano-2026-03-17", alias="PRIORI_OPENAI_REASONING_MODEL"
    )
    openai_verifier_model: str = Field(
        default="gpt-5.4-nano-2026-03-17", alias="PRIORI_OPENAI_VERIFIER_MODEL"
    )
    openai_case_review_model: str = Field(
        default="gpt-5.4", alias="PRIORI_OPENAI_CASE_REVIEW_MODEL"
    )
    redact_traces: bool = Field(default=True, alias="PRIORI_REDACT_TRACES")
    trace_retention_days: int = Field(default=30, alias="PRIORI_TRACE_RETENTION_DAYS")
    seed: int = Field(default=17, alias="PRIORI_SEED")
    experiment_namespace: str = Field(default="local-dev", alias="PRIORI_EXPERIMENT_NAMESPACE")
    mechanism_dag_enabled: bool = Field(default=True, alias="PRIORI_MECHANISM_DAG_ENABLED")
    open_world_reasoning_enabled: bool = Field(default=True, alias="PRIORI_OPEN_WORLD_REASONING_ENABLED")
    open_world_expand_uncertain_only: bool = Field(default=True, alias="PRIORI_OPEN_WORLD_EXPAND_UNCERTAIN_ONLY")
    open_world_max_hypotheses: int = Field(default=6, alias="PRIORI_OPEN_WORLD_MAX_HYPOTHESES")
    open_world_max_tests: int = Field(default=6, alias="PRIORI_OPEN_WORLD_MAX_TESTS")
    mietic_path: str | None = Field(default=None, alias="PRIORI_MIETIC_PATH")
    n2c2_2018_track2_path: str | None = Field(default=None, alias="PRIORI_N2C2_2018_TRACK2_PATH")
    medval_bench_path: str | None = Field(default=None, alias="PRIORI_MEDVAL_BENCH_PATH")
    reviewed_cases_path: str | None = Field(default=None, alias="PRIORI_REVIEWED_CASES_PATH")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    @property
    def live_llm_provider_ready(self) -> bool:
        return any([self.openai_api_key, self.google_api_key])


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
