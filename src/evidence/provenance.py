from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


SourceKind = Literal["primary_literature", "clinical_rule", "cost_model", "risk_heuristic", "registry_note"]


class SourceRecord(BaseModel):
    source_id: str
    title: str
    kind: SourceKind
    citation: str
    notes: str


class ProvenanceBadge(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1, default=0.8)


def badges_for_refs(source_type: str, refs: list[str]) -> list[ProvenanceBadge]:
    badges = [ProvenanceBadge(label=f"source:{source_type.replace('_', '-')}")]
    badges.extend(ProvenanceBadge(label=ref) for ref in refs)
    return badges

