from __future__ import annotations

from pydantic import BaseModel

from core.models import ClinicalDecisionContext
from core.models import DiagnosisHypothesis
from evidence.integration import enrich_hypothesis_evidence
from evidence.lr_catalog import LR_CATALOG
from evidence.registry_loader import LREntry


class DiseaseProfile(BaseModel):
    slug: str
    name: str
    category: str
    prior: float
    dangerous: bool
    urgency_weight: float
    description: str

    def to_hypothesis(
        self,
        *,
        context: ClinicalDecisionContext | None = None,
        registry_entries: list[LREntry] | None = None,
    ) -> DiagnosisHypothesis:
        evidence_items = enrich_hypothesis_evidence(
            self.slug,
            LR_CATALOG.get(self.slug, []),
            context=context,
            registry_entries=registry_entries,
        )
        supporting = [item for item in evidence_items if item.lr.positive_lr >= 1.0]
        contradicting = [item for item in evidence_items if item.lr.positive_lr < 1.0]
        return DiagnosisHypothesis(
            slug=self.slug,
            name=self.name,
            category=self.category,
            prior=self.prior,
            dangerous=self.dangerous,
            urgency_weight=self.urgency_weight,
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
        urgency_weight=0.95,
        description="Acute thromboembolic disease with moderate-to-high short-term harm if missed.",
    ),
    "pneumonia": DiseaseProfile(
        slug="pneumonia",
        name="Community-Acquired Pneumonia",
        category="infectious",
        prior=0.24,
        dangerous=True,
        urgency_weight=0.55,
        description="Infectious parenchymal lung disease with urgency tied to oxygenation and sepsis context.",
    ),
    "heart_failure": DiseaseProfile(
        slug="heart_failure",
        name="Acute Decompensated Heart Failure",
        category="cardiovascular",
        prior=0.22,
        dangerous=True,
        urgency_weight=0.65,
        description="Cardiogenic congestion syndrome with meaningful resource and escalation implications.",
    ),
    "acs": DiseaseProfile(
        slug="acs",
        name="Acute Coronary Syndrome",
        category="cardiovascular",
        prior=0.16,
        dangerous=True,
        urgency_weight=0.9,
        description="Ischemic chest pain syndrome requiring prompt triage and serial reassessment.",
    ),
    "upper_gi_bleed": DiseaseProfile(
        slug="upper_gi_bleed",
        name="Upper Gastrointestinal Bleeding / Anticoagulant-Associated Hemorrhage",
        category="gastrointestinal",
        prior=0.09,
        dangerous=True,
        urgency_weight=0.95,
        description="Hemorrhagic gastrointestinal syndrome with anticoagulation-sensitive escalation and transfusion implications.",
    ),
}


def default_hypotheses(
    *,
    context: ClinicalDecisionContext | None = None,
    registry_entries: list[LREntry] | None = None,
) -> list[DiagnosisHypothesis]:
    return [
        profile.to_hypothesis(context=context, registry_entries=registry_entries)
        for profile in DISEASE_PROFILES.values()
    ]
