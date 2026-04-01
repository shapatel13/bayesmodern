from __future__ import annotations

from pydantic import BaseModel

from core.models import DiagnosisHypothesis
from evidence.lr_catalog import LR_CATALOG


class DiseaseProfile(BaseModel):
    slug: str
    name: str
    category: str
    prior: float
    dangerous: bool
    description: str

    def to_hypothesis(self) -> DiagnosisHypothesis:
        evidence_items = LR_CATALOG.get(self.slug, [])
        supporting = [item for item in evidence_items if item.lr.positive_lr >= 1.0]
        contradicting = [item for item in evidence_items if item.lr.positive_lr < 1.0]
        return DiagnosisHypothesis(
            slug=self.slug,
            name=self.name,
            category=self.category,
            prior=self.prior,
            supporting_findings=supporting,
            contradicting_findings=contradicting,
        )


DISEASE_PROFILES: dict[str, DiseaseProfile] = {
    "pe": DiseaseProfile(
        slug="pe",
        name="Pulmonary Embolism",
        category="vascular",
        prior=0.18,
        dangerous=True,
        description="Acute thromboembolic disease with moderate-to-high short-term harm if missed.",
    ),
    "pneumonia": DiseaseProfile(
        slug="pneumonia",
        name="Community-Acquired Pneumonia",
        category="infectious",
        prior=0.24,
        dangerous=True,
        description="Infectious parenchymal lung disease with urgency tied to oxygenation and sepsis context.",
    ),
    "heart_failure": DiseaseProfile(
        slug="heart_failure",
        name="Acute Decompensated Heart Failure",
        category="cardiovascular",
        prior=0.22,
        dangerous=True,
        description="Cardiogenic congestion syndrome with meaningful resource and escalation implications.",
    ),
    "acs": DiseaseProfile(
        slug="acs",
        name="Acute Coronary Syndrome",
        category="cardiovascular",
        prior=0.16,
        dangerous=True,
        description="Ischemic chest pain syndrome requiring prompt triage and serial reassessment.",
    ),
}


def default_hypotheses() -> list[DiagnosisHypothesis]:
    return [profile.to_hypothesis() for profile in DISEASE_PROFILES.values()]

