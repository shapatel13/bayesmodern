from core.calibration import brier_score, classify_calibration, expected_calibration_error


def test_brier_score_zero_for_perfect_predictions() -> None:
    assert brier_score([0.0, 1.0], [0, 1]) == 0.0


def test_expected_calibration_error_is_non_negative() -> None:
    value = expected_calibration_error([0.1, 0.7, 0.8], [0, 1, 0], bins=3)
    assert value >= 0


def test_calibration_state_fragile_for_wide_interval() -> None:
    assert classify_calibration(0.6, 0.5) == "fragile"

