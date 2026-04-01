from core.differential import DifferentialEngine
from core.models import ClinicalDecisionContext, ClinicalFinding
from core.next_best_test import NextBestTestEngine
from evidence.cost_catalog import default_test_catalog
from evidence.disease_profiles import default_hypotheses


def test_pe_case_ranks_pe_high_and_prefers_d_dimer_before_cta() -> None:
    context = ClinicalDecisionContext(
        case_id="integration-pe-1",
        findings=[
            ClinicalFinding(key="pleuritic_chest_pain", label="Pleuritic chest pain", present=True),
            ClinicalFinding(key="tachycardia", label="Tachycardia", present=True),
            ClinicalFinding(key="hypoxemia", label="Hypoxemia", present=True),
            ClinicalFinding(key="fever", label="Fever", present=False),
        ],
    )
    differential = DifferentialEngine().rank(context, default_hypotheses(), samples=200, seed=17)
    assert differential.ranked[0].slug == "pe"
    test_plan = NextBestTestEngine().rank(
        {entry.slug: entry.posterior for entry in differential.ranked},
        default_test_catalog(),
        context,
    )
    assert test_plan[0].slug == "d_dimer"

