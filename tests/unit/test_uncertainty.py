from core.uncertainty import normalize_sampled_distributions


def test_sampled_distribution_summary_stays_in_bounds() -> None:
    summary = normalize_sampled_distributions(
        [
            {"a": 2.0, "b": 1.0},
            {"a": 1.0, "b": 3.0},
        ]
    )
    assert 0 <= summary["a"].low <= summary["a"].high <= 1
    assert 0 <= summary["b"].low <= summary["b"].high <= 1

