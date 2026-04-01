from core.models import ClinicalDecisionContext
from core.next_best_test import NextBestTestEngine
from evidence.cost_catalog import TEST_CATALOG


def test_already_done_test_is_demoted() -> None:
    context = ClinicalDecisionContext(case_id="synthetic-done-1", completed_tests=["d_dimer"])
    recommendations = NextBestTestEngine().rank({"pe": 0.3}, [TEST_CATALOG["d_dimer"]], context)
    assert recommendations[0].disposition == "already_answered"

