from core.models import ClinicalDecisionContext
from core.next_best_test import NextBestTestEngine
from evidence.cost_catalog import TEST_CATALOG


def test_renal_impairment_penalizes_contrast_test() -> None:
    context = ClinicalDecisionContext(case_id="synthetic-renal-1", renal_impairment=True)
    recommendations = NextBestTestEngine().rank({"pe": 0.3}, [TEST_CATALOG["cta_pe"]], context)
    assert recommendations[0].stewardship_score < 0.1

