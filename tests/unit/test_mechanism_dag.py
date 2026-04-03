from __future__ import annotations

import pytest

from agent.orchestrator import PRIORIXOrchestrator
from core.mechanism_dag import mechanism_dag_available
from utils.config import Settings


MIXED_CARDIORENAL_BLEED_CASE = (
    "A 72-year-old woman with HFrEF (EF 30%), CKD4, atrial fibrillation on apixaban, diabetes, and CAD "
    "presents with 2 days of worsening dyspnea, dark stools, poor intake, oliguria, and confusion. "
    "She has worsening leg edema and recently had her torsemide increased while also taking ibuprofen. "
    "Vitals show T 37.9°C, HR 122 irregular, BP 88/54, RR 28, and SpO2 90% on 4 L. Exam shows elevated JVP, "
    "bibasilar crackles, 2+ edema, cool legs, delayed capillary refill, and melena. Labs show hemoglobin 7.4 "
    "from a baseline of 10.8, sodium 126, potassium 5.9, bicarbonate 17, BUN 78, creatinine 3.6 from a baseline "
    "of 2.1, lactate 3.8, BNP 3100, and urine sodium 14. Chest x-ray shows bilateral interstitial-alveolar "
    "opacities, small pleural effusions, and cardiomegaly. Bedside ultrasound shows a plethoric IVC, diffuse "
    "B-lines, and reduced LV systolic function. The challenge is to determine the relative contributions of "
    "cardiogenic shock, venous congestion, GI bleeding with low effective arterial volume, medication-associated "
    "AKI, and fluid intolerance."
)


@pytest.mark.skipif(not mechanism_dag_available(), reason="pgmpy is not installed")
def test_pgmpy_dag_refines_mixed_cardiorenal_bleeding_case() -> None:
    base_report = PRIORIXOrchestrator(
        settings=Settings(
            _env_file=None,
            allow_live_llm=False,
            default_model_provider="offline",
            mechanism_dag_enabled=False,
        )
    ).analyze_text_case("dag-base", MIXED_CARDIORENAL_BLEED_CASE)
    dag_report = PRIORIXOrchestrator(
        settings=Settings(
            _env_file=None,
            allow_live_llm=False,
            default_model_provider="offline",
            mechanism_dag_enabled=True,
        )
    ).analyze_text_case("dag-on", MIXED_CARDIORENAL_BLEED_CASE)

    base_mechanisms = {estimate.slug: estimate.posterior for estimate in base_report.mechanism_states.ranked}
    dag_mechanisms = {estimate.slug: estimate.posterior for estimate in dag_report.mechanism_states.ranked}
    top_tests = [recommendation.slug for recommendation in dag_report.next_best_tests[:3]]

    assert dag_mechanisms["impaired_contractility"] > base_mechanisms["impaired_contractility"] + 0.08
    assert dag_mechanisms["fluid_intolerance"] > base_mechanisms["fluid_intolerance"] + 0.1
    assert dag_mechanisms["medication_toxicity_effect"] > base_mechanisms["medication_toxicity_effect"] + 0.1
    assert dag_report.next_best_tests[0].slug == "repeat_hemoglobin"
    assert "bedside_echo" in top_tests
    assert "Mixed cardiogenic and congestive physiology" in dag_report.mechanism_states.summary
    assert "pgmpy DAG refinement" in dag_report.mechanism_states.model_note
