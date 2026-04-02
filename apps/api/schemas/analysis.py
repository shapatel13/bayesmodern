from __future__ import annotations

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    case_id: str = Field(description="Stable case identifier for audit and reproducibility.")
    note_text: str = Field(description="Clinical vignette or note text to analyze.")
    policy_version: str = Field(default="v1-deterministic", description="Reasoning policy version to use.")
