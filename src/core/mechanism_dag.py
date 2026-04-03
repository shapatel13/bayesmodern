from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product

from core.models import ClinicalDecisionContext

try:
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    from pgmpy.models import DiscreteBayesianNetwork
except Exception:  # pragma: no cover - gracefully handled when pgmpy is unavailable
    DiscreteBayesianNetwork = None
    TabularCPD = None
    VariableElimination = None


_STATE_NODE_TO_SLUG = {
    "impaired_contractility": "impaired_contractility",
    "venous_congestion": "venous_congestion",
    "hemorrhage": "hemorrhagic_tendency_active_blood_loss",
    "low_eav": "low_effective_arterial_volume",
    "medtox": "medication_toxicity_effect",
    "fluid_intolerance": "fluid_intolerance",
}

_DIURETIC_MEDICATIONS = {"torsemide", "furosemide", "bumetanide", "metolazone", "hydrochlorothiazide"}
_NSAID_MEDICATIONS = {"ibuprofen", "naproxen", "diclofenac", "indomethacin", "ketorolac"}


@dataclass(frozen=True)
class MechanismDagRefinement:
    state_posteriors: dict[str, float]
    observed_signals: list[str]
    blend_weight: float
    interval_radius: float
    note: str
    provenance_badges: list[str]


def mechanism_dag_available() -> bool:
    return all(item is not None for item in (DiscreteBayesianNetwork, TabularCPD, VariableElimination))


def _root_cpd(variable: str, probability_true: float) -> TabularCPD:
    bounded_probability = min(max(probability_true, 0.01), 0.99)
    return TabularCPD(variable=variable, variable_card=2, values=[[1.0 - bounded_probability], [bounded_probability]])


def _noisy_or_cpd(
    variable: str,
    parents: list[str],
    *,
    leak_probability: float,
    strengths: list[float],
) -> TabularCPD:
    combinations = list(product([0, 1], repeat=len(parents)))
    false_probabilities: list[float] = []
    true_probabilities: list[float] = []
    for combination in combinations:
        no_activation_probability = 1.0 - leak_probability
        for is_active, strength in zip(combination, strengths, strict=True):
            if is_active:
                no_activation_probability *= 1.0 - strength
        true_probability = 1.0 - no_activation_probability
        true_probabilities.append(min(max(true_probability, 0.01), 0.99))
        false_probabilities.append(1.0 - true_probabilities[-1])
    return TabularCPD(
        variable=variable,
        variable_card=2,
        values=[false_probabilities, true_probabilities],
        evidence=parents,
        evidence_card=[2 for _ in parents],
    )


@lru_cache(maxsize=1)
def _build_inference_engine() -> VariableElimination | None:
    if not mechanism_dag_available():
        return None

    model = DiscreteBayesianNetwork(
        [
            ("anticoagulated_obs", "hemorrhage"),
            ("anticoagulated_obs", "medtox"),
            ("nsaid_obs", "medtox"),
            ("diuretic_obs", "medtox"),
            ("renal_risk_obs", "medtox"),
            ("poor_intake_obs", "low_eav"),
            ("hemorrhage", "low_eav"),
            ("medtox", "low_eav"),
            ("impaired_contractility", "venous_congestion"),
            ("venous_congestion", "fluid_intolerance"),
            ("impaired_contractility", "fluid_intolerance"),
            ("impaired_contractility", "reduced_ef_obs"),
            ("impaired_contractility", "cool_extremities_obs"),
            ("impaired_contractility", "hypotension_obs"),
            ("impaired_contractility", "lactate_obs"),
            ("impaired_contractility", "bnp_obs"),
            ("venous_congestion", "bnp_obs"),
            ("venous_congestion", "jvp_obs"),
            ("venous_congestion", "crackles_obs"),
            ("venous_congestion", "edema_obs"),
            ("venous_congestion", "plethoric_ivc_obs"),
            ("venous_congestion", "blines_obs"),
            ("venous_congestion", "hypoxemia_obs"),
            ("venous_congestion", "hyponatremia_obs"),
            ("venous_congestion", "creatinine_obs"),
            ("hemorrhage", "melena_obs"),
            ("hemorrhage", "severe_anemia_obs"),
            ("hemorrhage", "symptomatic_anemia_obs"),
            ("low_eav", "hypotension_obs"),
            ("low_eav", "lactate_obs"),
            ("low_eav", "oliguria_obs"),
            ("low_eav", "cool_extremities_obs"),
            ("low_eav", "creatinine_obs"),
            ("medtox", "oliguria_obs"),
            ("medtox", "creatinine_obs"),
        ]
    )

    model.add_cpds(
        _root_cpd("anticoagulated_obs", 0.12),
        _root_cpd("nsaid_obs", 0.08),
        _root_cpd("diuretic_obs", 0.14),
        _root_cpd("renal_risk_obs", 0.18),
        _root_cpd("poor_intake_obs", 0.12),
        _root_cpd("impaired_contractility", 0.14),
        _noisy_or_cpd("hemorrhage", ["anticoagulated_obs"], leak_probability=0.08, strengths=[0.32]),
        _noisy_or_cpd(
            "medtox",
            ["anticoagulated_obs", "nsaid_obs", "diuretic_obs", "renal_risk_obs"],
            leak_probability=0.06,
            strengths=[0.14, 0.3, 0.18, 0.24],
        ),
        _noisy_or_cpd(
            "low_eav",
            ["hemorrhage", "poor_intake_obs", "medtox"],
            leak_probability=0.08,
            strengths=[0.58, 0.28, 0.16],
        ),
        _noisy_or_cpd("venous_congestion", ["impaired_contractility"], leak_probability=0.17, strengths=[0.5]),
        _noisy_or_cpd(
            "fluid_intolerance",
            ["venous_congestion", "impaired_contractility"],
            leak_probability=0.08,
            strengths=[0.58, 0.24],
        ),
        _noisy_or_cpd("reduced_ef_obs", ["impaired_contractility"], leak_probability=0.03, strengths=[0.82]),
        _noisy_or_cpd(
            "bnp_obs",
            ["impaired_contractility", "venous_congestion"],
            leak_probability=0.08,
            strengths=[0.22, 0.62],
        ),
        _noisy_or_cpd("jvp_obs", ["venous_congestion"], leak_probability=0.05, strengths=[0.74]),
        _noisy_or_cpd("crackles_obs", ["venous_congestion"], leak_probability=0.08, strengths=[0.6]),
        _noisy_or_cpd("edema_obs", ["venous_congestion"], leak_probability=0.07, strengths=[0.66]),
        _noisy_or_cpd("plethoric_ivc_obs", ["venous_congestion"], leak_probability=0.04, strengths=[0.76]),
        _noisy_or_cpd("blines_obs", ["venous_congestion"], leak_probability=0.05, strengths=[0.78]),
        _noisy_or_cpd("hypoxemia_obs", ["venous_congestion"], leak_probability=0.1, strengths=[0.32]),
        _noisy_or_cpd("hyponatremia_obs", ["venous_congestion"], leak_probability=0.08, strengths=[0.26]),
        _noisy_or_cpd("melena_obs", ["hemorrhage"], leak_probability=0.03, strengths=[0.86]),
        _noisy_or_cpd("severe_anemia_obs", ["hemorrhage"], leak_probability=0.05, strengths=[0.72]),
        _noisy_or_cpd("symptomatic_anemia_obs", ["hemorrhage"], leak_probability=0.08, strengths=[0.65]),
        _noisy_or_cpd(
            "hypotension_obs",
            ["impaired_contractility", "low_eav"],
            leak_probability=0.08,
            strengths=[0.28, 0.58],
        ),
        _noisy_or_cpd(
            "lactate_obs",
            ["impaired_contractility", "low_eav"],
            leak_probability=0.09,
            strengths=[0.22, 0.54],
        ),
        _noisy_or_cpd(
            "cool_extremities_obs",
            ["impaired_contractility", "low_eav"],
            leak_probability=0.08,
            strengths=[0.34, 0.42],
        ),
        _noisy_or_cpd(
            "oliguria_obs",
            ["low_eav", "medtox"],
            leak_probability=0.1,
            strengths=[0.34, 0.3],
        ),
        _noisy_or_cpd(
            "creatinine_obs",
            ["venous_congestion", "low_eav", "medtox"],
            leak_probability=0.12,
            strengths=[0.2, 0.24, 0.38],
        ),
    )
    model.check_model()
    return VariableElimination(model)


def _text_has_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _collect_evidence(context: ClinicalDecisionContext) -> tuple[dict[str, int], list[str]]:
    lowered = (context.symptoms_free_text or "").lower()
    present_keys = {finding.key for finding in context.findings if finding.present}
    absent_keys = {finding.key for finding in context.findings if finding.present is False}
    medications = {item.strip().lower() for item in context.medications}

    evidence: dict[str, int] = {}
    observed_signals: list[str] = []

    def mark(node: str, present: bool, label: str) -> None:
        if present:
            evidence[node] = 1
            observed_signals.append(label)
        elif node not in evidence:
            evidence[node] = 0

    if medications & {"warfarin", "heparin", "apixaban", "rivaroxaban", "dabigatran", "enoxaparin"} or "anticoagulated" in present_keys:
        evidence["anticoagulated_obs"] = 1
        observed_signals.append("anticoagulation exposure")
    if medications & _NSAID_MEDICATIONS:
        evidence["nsaid_obs"] = 1
        observed_signals.append("NSAID exposure")
    if medications & _DIURETIC_MEDICATIONS:
        evidence["diuretic_obs"] = 1
        observed_signals.append("diuretic exposure")
    if context.renal_impairment or "creatinine_elevated" in present_keys or (context.age_years or 0) >= 70:
        evidence["renal_risk_obs"] = 1
        observed_signals.append("renal-risk substrate")
    if _text_has_any(lowered, ("poor intake", "poor po", "decreased intake", "not eating", "reduced intake")):
        evidence["poor_intake_obs"] = 1
        observed_signals.append("poor intake")

    mark("reduced_ef_obs", "reduced_ef" in present_keys or _text_has_any(lowered, ("ef 30%", "reduced lv systolic function", "reduced systolic function")), "reduced systolic function")
    mark("bnp_obs", "bnp_elevated" in present_keys, "natriuretic peptide elevation")
    mark("jvp_obs", "elevated_jvp" in present_keys, "elevated JVP")
    mark("crackles_obs", "crackles" in present_keys, "pulmonary crackles")
    mark("edema_obs", "leg_edema" in present_keys, "peripheral edema")
    mark("plethoric_ivc_obs", _text_has_any(lowered, ("plethoric ivc", "dilated ivc")) or "elevated_jvp" in present_keys, "plethoric IVC")
    mark("blines_obs", _text_has_any(lowered, ("b-lines", "b lines", "diffuse b-lines", "diffuse b lines")), "diffuse B-lines")
    mark("hypoxemia_obs", "hypoxemia" in present_keys, "hypoxemia")
    mark("hyponatremia_obs", "hyponatremia" in present_keys, "hyponatremia")
    mark("melena_obs", "melena" in present_keys or "active_gi_bleeding" in present_keys, "melena / GI blood loss")
    mark("severe_anemia_obs", "severe_anemia" in present_keys, "severe anemia")
    mark("symptomatic_anemia_obs", "symptomatic_anemia" in present_keys, "symptomatic anemia")
    mark("oliguria_obs", "oliguria" in present_keys, "oliguria")
    mark("creatinine_obs", "creatinine_elevated" in present_keys or context.renal_impairment, "creatinine elevation")
    mark("lactate_obs", "elevated_lactate" in present_keys, "lactate elevation")
    mark("cool_extremities_obs", "cool_extremities" in present_keys, "cool extremities")
    mark("hypotension_obs", context.hemodynamic_instability or "hemodynamic_instability" in present_keys, "hypotension / instability")

    for node, finding_key in {
        "melena_obs": "melena",
        "severe_anemia_obs": "severe_anemia",
        "symptomatic_anemia_obs": "symptomatic_anemia",
        "oliguria_obs": "oliguria",
        "cool_extremities_obs": "cool_extremities",
    }.items():
        if node not in evidence and finding_key in absent_keys:
            evidence[node] = 0

    return evidence, observed_signals


def _posterior_true(inference: VariableElimination, variable: str, evidence: dict[str, int]) -> float:
    result = inference.query(variables=[variable], evidence=evidence, show_progress=False)
    return float(result.values[1])


def infer_mechanism_dag_refinement(context: ClinicalDecisionContext) -> MechanismDagRefinement | None:
    inference = _build_inference_engine()
    if inference is None:
        return None

    evidence, observed_signals = _collect_evidence(context)
    if len(observed_signals) < 4:
        return None

    state_posteriors = {
        _STATE_NODE_TO_SLUG[node]: _posterior_true(inference, node, evidence)
        for node in _STATE_NODE_TO_SLUG
    }
    blend_weight = min(0.4, 0.18 + (0.03 * len(observed_signals)))
    interval_radius = max(0.08, 0.22 - (0.008 * len(observed_signals)))

    notes = ["pgmpy DAG refinement jointly weighed congestion, blood-loss, perfusion, and medication-stress clues."]
    if state_posteriors["venous_congestion"] >= 0.7 and state_posteriors["impaired_contractility"] >= 0.6:
        notes.append("The graph favors mixed cardiogenic plus congestive physiology rather than a single isolated state.")
    if state_posteriors["hemorrhagic_tendency_active_blood_loss"] >= 0.65 and state_posteriors["low_effective_arterial_volume"] >= 0.65:
        notes.append("Active blood loss appears to be materially worsening effective arterial volume.")
    if (
        state_posteriors["medication_toxicity_effect"] >= 0.5
        and state_posteriors["low_effective_arterial_volume"] >= 0.55
        and state_posteriors["venous_congestion"] >= 0.55
    ):
        notes.append("Renal stress appears multifactorial, with congestion, hypoperfusion, and medication effect all contributing.")

    return MechanismDagRefinement(
        state_posteriors=state_posteriors,
        observed_signals=observed_signals,
        blend_weight=blend_weight,
        interval_radius=interval_radius,
        note=" ".join(notes),
        provenance_badges=["source:pgmpy-dag", "dag:cardiorenal_hemorrhage_v1"],
    )
