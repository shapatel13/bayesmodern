from core.bayes import bayes_update, normalize_distribution, sequential_bayes_update


def test_bayes_update_increases_probability_for_supportive_evidence() -> None:
    posterior = bayes_update(0.2, 5.0)
    assert round(posterior, 3) == 0.556


def test_sequential_bayes_update_matches_manual_progression() -> None:
    posterior = sequential_bayes_update(0.1, [4.0, 2.0])
    assert round(posterior, 3) == 0.471


def test_distribution_normalization_preserves_relative_mass() -> None:
    normalized = normalize_distribution({"a": 2.0, "b": 1.0})
    assert normalized["a"] == 2 / 3
    assert normalized["b"] == 1 / 3

