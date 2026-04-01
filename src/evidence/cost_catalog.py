from __future__ import annotations

from core.models import CandidateTest, ClinicalDecisionContext, LikelihoodRatioRange
from evidence.integration import enrich_candidate_test
from evidence.registry_loader import LREntry


TEST_CATALOG: dict[str, CandidateTest] = {
    "d_dimer": CandidateTest(
        slug="d_dimer",
        name="D-dimer",
        target_diagnoses=["pe"],
        diagnosis_lrs={"pe": LikelihoodRatioRange(positive_lr=2.2, negative_lr=0.2)},
        direct_cost=40,
        downstream_cost=0,
        actionability=0.7,
        bedside=True,
        provenance_refs=["study:d_dimer_pe", "cost:starter_us_hospital"],
    ),
    "cta_pe": CandidateTest(
        slug="cta_pe",
        name="CT Pulmonary Angiography",
        target_diagnoses=["pe"],
        diagnosis_lrs={"pe": LikelihoodRatioRange(positive_lr=8.0, negative_lr=0.1)},
        direct_cost=850,
        downstream_cost=300,
        radiation=0.6,
        nephrotoxicity=0.35,
        actionability=0.95,
        provenance_refs=["study:cta_pe", "cost:starter_us_hospital", "risk:contrast_ckd"],
    ),
    "cxr": CandidateTest(
        slug="cxr",
        name="Chest Radiograph",
        target_diagnoses=["pneumonia", "heart_failure"],
        diagnosis_lrs={
            "pneumonia": LikelihoodRatioRange(positive_lr=3.0, negative_lr=0.4),
            "heart_failure": LikelihoodRatioRange(positive_lr=1.8, negative_lr=0.7),
        },
        direct_cost=120,
        downstream_cost=0,
        radiation=0.15,
        actionability=0.75,
        provenance_refs=["study:cxr_pneumonia", "cost:starter_us_hospital"],
    ),
    "bnp": CandidateTest(
        slug="bnp",
        name="BNP",
        target_diagnoses=["heart_failure"],
        diagnosis_lrs={"heart_failure": LikelihoodRatioRange(positive_lr=3.1, negative_lr=0.3)},
        direct_cost=90,
        downstream_cost=0,
        actionability=0.8,
        bedside=True,
        provenance_refs=["study:bnp_hf", "cost:starter_us_hospital"],
    ),
}


def default_test_catalog(
    *,
    context: ClinicalDecisionContext | None = None,
    registry_entries: list[LREntry] | None = None,
) -> list[CandidateTest]:
    return [
        enrich_candidate_test(candidate, context=context, registry_entries=registry_entries)
        for candidate in TEST_CATALOG.values()
    ]
