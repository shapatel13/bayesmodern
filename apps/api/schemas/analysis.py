from __future__ import annotations

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    case_id: str = Field(description="Stable case identifier for audit and reproducibility.")
    note_text: str = Field(description="Clinical vignette or note text to analyze.")

