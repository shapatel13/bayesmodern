from core.thresholds import DecisionCostModel, calculate_thresholds, explain_threshold_position


def test_thresholds_are_ordered_and_bounded() -> None:
    thresholds = calculate_thresholds(
        DecisionCostModel(
            benefit_of_treatment=9.0,
            harm_of_treatment=2.0,
            harm_of_missed_disease=8.0,
            harm_of_test=1.0,
        )
    )
    assert 0 < thresholds.defer_threshold <= thresholds.test_threshold
    assert thresholds.test_threshold < thresholds.treatment_threshold


def test_threshold_explanation_returns_treat_for_high_posterior() -> None:
    thresholds = calculate_thresholds(
        DecisionCostModel(
            benefit_of_treatment=10.0,
            harm_of_treatment=2.0,
            harm_of_missed_disease=8.0,
            harm_of_test=1.0,
        )
    )
    decision = explain_threshold_position(0.8, thresholds)
    assert decision.action in {"treat", "icu_consider"}

