from core.triage import assess_triage


def test_triage_flags_emergent_instability() -> None:
    assessment = assess_triage(
        dangerous_posterior_mass=0.3,
        hemodynamic_instability=True,
        critical_values_present=False,
    )
    assert assessment.urgency == "emergent"

