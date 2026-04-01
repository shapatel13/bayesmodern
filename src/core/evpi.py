from __future__ import annotations

from collections.abc import Mapping


def expected_value_current_information(
    state_probabilities: Mapping[str, float],
    action_utilities: Mapping[str, Mapping[str, float]],
) -> float:
    best_value = float("-inf")
    for utilities in action_utilities.values():
        value = sum(state_probabilities[state] * utilities[state] for state in state_probabilities)
        best_value = max(best_value, value)
    return best_value


def expected_value_perfect_information(
    state_probabilities: Mapping[str, float],
    action_utilities: Mapping[str, Mapping[str, float]],
) -> float:
    value = 0.0
    for state, probability in state_probabilities.items():
        best_action_value = max(utilities[state] for utilities in action_utilities.values())
        value += probability * best_action_value
    return value


def expected_value_of_perfect_information(
    state_probabilities: Mapping[str, float],
    action_utilities: Mapping[str, Mapping[str, float]],
) -> float:
    return expected_value_perfect_information(state_probabilities, action_utilities) - expected_value_current_information(
        state_probabilities, action_utilities
    )

