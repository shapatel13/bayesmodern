from __future__ import annotations

from core.models import HypothesisEvidence, LatentStateDefinition, LikelihoodRatioRange


def _evidence(
    finding_key: str,
    label: str,
    *,
    positive_lr: float,
    negative_lr: float,
    rationale: str,
    provenance_refs: list[str] | None = None,
) -> HypothesisEvidence:
    return HypothesisEvidence(
        finding_key=finding_key,
        label=label,
        lr=LikelihoodRatioRange(positive_lr=positive_lr, negative_lr=negative_lr),
        rationale=rationale,
        provenance_refs=provenance_refs or ["mechanism:starter_registry"],
        source_type="hard_coded",
    )


LATENT_STATE_CATALOG: dict[str, LatentStateDefinition] = {
    "distributive_physiology": LatentStateDefinition(
        slug="distributive_physiology",
        name="Distributive Physiology",
        category="hemodynamics",
        prior=0.16,
        description="Relative vasodilation or inflammatory distributive process with reduced effective vascular tone.",
        supporting_findings=[
            _evidence("fever", "Fever", positive_lr=2.8, negative_lr=0.78, rationale="Fever increases the likelihood of a distributive or inflammatory physiology."),
            _evidence("hemodynamic_instability", "Hemodynamic instability", positive_lr=1.9, negative_lr=0.88, rationale="Shock-like instability supports distributive physiology when present."),
            _evidence("tachycardia", "Tachycardia", positive_lr=1.45, negative_lr=0.9, rationale="Compensatory tachycardia often accompanies distributive states."),
            _evidence("warm_extremities", "Warm extremities", positive_lr=2.2, negative_lr=0.9, rationale="Warm vasodilated extremities support distributive physiology."),
            _evidence("purulent_sputum", "Purulent sputum", positive_lr=1.4, negative_lr=0.95, rationale="An infectious pulmonary source can support a distributive picture."),
        ],
        contradicting_findings=[
            _evidence("cool_extremities", "Cool extremities", positive_lr=0.55, negative_lr=1.2, rationale="Cool extremities argue against a warm distributive state."),
        ],
    ),
    "impaired_contractility": LatentStateDefinition(
        slug="impaired_contractility",
        name="Impaired Contractility / Cardiogenic Physiology",
        category="cardiovascular",
        prior=0.14,
        description="Low forward flow due to impaired cardiac pump function or acute ischemic injury.",
        supporting_findings=[
            _evidence("pressure_chest_pain", "Pressure-like chest pain", positive_lr=1.7, negative_lr=0.92, rationale="Ischemic chest pain raises concern for pump failure from myocardial ischemia."),
            _evidence("troponin_positive", "Positive troponin", positive_lr=3.6, negative_lr=0.72, rationale="Troponin elevation increases concern for ischemic or myocardial injury physiology."),
            _evidence("pain_radiation", "Radiation to arm or jaw", positive_lr=1.5, negative_lr=0.95, rationale="Typical ischemic radiation supports cardiogenic physiology."),
            _evidence("diaphoresis", "Diaphoresis", positive_lr=1.4, negative_lr=0.96, rationale="Diaphoresis supports a high-acuity cardiogenic or ischemic picture."),
            _evidence("hemodynamic_instability", "Hemodynamic instability", positive_lr=2.0, negative_lr=0.86, rationale="Instability supports low forward flow states."),
            _evidence("reduced_ef", "Reduced ejection fraction", positive_lr=4.0, negative_lr=0.6, rationale="Reduced EF strongly supports impaired contractility."),
        ],
        contradicting_findings=[
            _evidence("warm_extremities", "Warm extremities", positive_lr=0.7, negative_lr=1.08, rationale="Warm vasodilated extremities slightly argue against primary cardiogenic physiology."),
        ],
    ),
    "low_effective_arterial_volume": LatentStateDefinition(
        slug="low_effective_arterial_volume",
        name="Low Effective Arterial Volume / Low Preload",
        category="hemodynamics",
        prior=0.18,
        description="Low preload or low effective arterial volume from hemorrhage, dehydration, vasodilation, or third spacing.",
        supporting_findings=[
            _evidence("hemodynamic_instability", "Hemodynamic instability", positive_lr=2.2, negative_lr=0.82, rationale="Instability raises concern for low effective arterial volume."),
            _evidence("tachycardia", "Tachycardia", positive_lr=1.6, negative_lr=0.9, rationale="Tachycardia often accompanies low preload states."),
            _evidence("active_gi_bleeding", "Active gastrointestinal bleeding", positive_lr=2.9, negative_lr=0.74, rationale="Active bleeding strongly supports low effective arterial volume."),
            _evidence("melena", "Melena", positive_lr=1.9, negative_lr=0.9, rationale="Melena supports ongoing blood loss and preload depletion."),
            _evidence("symptomatic_anemia", "Symptomatic anemia", positive_lr=2.0, negative_lr=0.84, rationale="Symptomatic anemia can reflect clinically meaningful blood loss."),
            _evidence("oliguria", "Oliguria", positive_lr=1.55, negative_lr=0.92, rationale="Oliguria can reflect poor renal perfusion from low effective arterial volume."),
        ],
        contradicting_findings=[
            _evidence("leg_edema", "Leg edema", positive_lr=0.7, negative_lr=1.12, rationale="Peripheral edema argues against isolated low-preload physiology."),
            _evidence("elevated_jvp", "Elevated JVP", positive_lr=0.6, negative_lr=1.18, rationale="Elevated JVP argues against isolated low effective arterial volume."),
        ],
    ),
    "venous_congestion": LatentStateDefinition(
        slug="venous_congestion",
        name="Venous Congestion",
        category="cardiorenal",
        prior=0.17,
        description="Raised venous pressures and congestive physiology affecting lungs or systemic veins.",
        supporting_findings=[
            _evidence("orthopnea", "Orthopnea", positive_lr=2.8, negative_lr=0.72, rationale="Orthopnea strongly supports congestion physiology."),
            _evidence("crackles", "Crackles", positive_lr=2.1, negative_lr=0.86, rationale="Pulmonary crackles support congestion or edema."),
            _evidence("leg_edema", "Leg edema", positive_lr=2.4, negative_lr=0.8, rationale="Peripheral edema supports venous congestion."),
            _evidence("elevated_jvp", "Elevated JVP", positive_lr=3.1, negative_lr=0.7, rationale="Elevated JVP is a strong venous congestion clue."),
            _evidence("hypoxemia", "Hypoxemia", positive_lr=1.25, negative_lr=0.96, rationale="Hypoxemia can reflect pulmonary venous congestion or edema."),
        ],
        contradicting_findings=[
            _evidence("active_gi_bleeding", "Active gastrointestinal bleeding", positive_lr=0.82, negative_lr=1.04, rationale="Bleeding physiology slightly shifts away from isolated congestion."),
        ],
    ),
    "fluid_responsiveness": LatentStateDefinition(
        slug="fluid_responsiveness",
        name="Fluid Responsiveness",
        category="hemodynamics",
        prior=0.15,
        description="Potential to improve forward flow with preload augmentation.",
        supporting_findings=[
            _evidence("low_preload_signal", "Low preload signal", positive_lr=2.1, negative_lr=0.78, rationale="Low-preload clues raise the chance of fluid responsiveness."),
            _evidence("active_gi_bleeding", "Active gastrointestinal bleeding", positive_lr=1.9, negative_lr=0.86, rationale="Active blood loss increases the chance of fluid responsiveness."),
            _evidence("symptomatic_anemia", "Symptomatic anemia", positive_lr=1.45, negative_lr=0.94, rationale="Blood-loss physiology can raise fluid responsiveness."),
        ],
        contradicting_findings=[
            _evidence("venous_congestion_signal", "Venous congestion signal", positive_lr=0.52, negative_lr=1.2, rationale="Strong congestion argues against useful fluid responsiveness."),
            _evidence("fluid_intolerance_signal", "Fluid intolerance signal", positive_lr=0.58, negative_lr=1.16, rationale="Evidence of fluid intolerance lowers the probability that more fluid will help."),
        ],
    ),
    "fluid_intolerance": LatentStateDefinition(
        slug="fluid_intolerance",
        name="Fluid Intolerance",
        category="hemodynamics",
        prior=0.16,
        description="Risk that additional fluid worsens congestion without improving forward flow.",
        supporting_findings=[
            _evidence("venous_congestion_signal", "Venous congestion signal", positive_lr=2.6, negative_lr=0.72, rationale="Congestion materially raises fluid intolerance."),
            _evidence("orthopnea", "Orthopnea", positive_lr=1.8, negative_lr=0.9, rationale="Orthopnea supports fluid intolerance through congestion."),
            _evidence("crackles", "Crackles", positive_lr=1.7, negative_lr=0.9, rationale="Pulmonary crackles support fluid intolerance."),
        ],
        contradicting_findings=[
            _evidence("low_preload_signal", "Low preload signal", positive_lr=0.72, negative_lr=1.08, rationale="Low-preload evidence argues against fluid intolerance as the dominant state."),
        ],
    ),
    "hemorrhagic_tendency_active_blood_loss": LatentStateDefinition(
        slug="hemorrhagic_tendency_active_blood_loss",
        name="Hemorrhagic Tendency / Active Blood Loss",
        category="hematology",
        prior=0.12,
        description="Clinically significant active bleeding or anticoagulation-amplified hemorrhagic physiology.",
        supporting_findings=[
            _evidence("active_gi_bleeding", "Active gastrointestinal bleeding", positive_lr=3.4, negative_lr=0.7, rationale="Direct bleeding evidence strongly supports active blood loss."),
            _evidence("melena", "Melena", positive_lr=2.8, negative_lr=0.76, rationale="Melena materially increases bleeding-related belief."),
            _evidence("symptomatic_anemia", "Symptomatic anemia", positive_lr=2.1, negative_lr=0.84, rationale="Symptomatic anemia supports clinically meaningful blood loss."),
            _evidence("anticoagulated", "On anticoagulation", positive_lr=1.8, negative_lr=0.94, rationale="Anticoagulation raises the risk of hemorrhagic physiology."),
            _evidence("hemodynamic_instability", "Hemodynamic instability", positive_lr=1.75, negative_lr=0.9, rationale="Instability increases concern for clinically important bleeding."),
        ],
        contradicting_findings=[],
    ),
    "thrombotic_ischemic_tendency": LatentStateDefinition(
        slug="thrombotic_ischemic_tendency",
        name="Thrombotic / Ischemic Tendency",
        category="vascular",
        prior=0.15,
        description="Signal that acute ischemic or thrombotic pathology is active or likely.",
        supporting_findings=[
            _evidence("pressure_chest_pain", "Pressure-like chest pain", positive_lr=1.8, negative_lr=0.92, rationale="Typical ischemic chest pain supports thrombotic or ischemic processes."),
            _evidence("pain_radiation", "Radiation to arm or jaw", positive_lr=1.55, negative_lr=0.94, rationale="Classic radiation supports ischemic physiology."),
            _evidence("diaphoresis", "Diaphoresis", positive_lr=1.35, negative_lr=0.96, rationale="Diaphoresis can accompany acute ischemic physiology."),
            _evidence("troponin_positive", "Positive troponin", positive_lr=3.0, negative_lr=0.74, rationale="Troponin elevation supports ischemic injury."),
            _evidence("pleuritic_chest_pain", "Pleuritic chest pain", positive_lr=1.45, negative_lr=0.96, rationale="Pleuritic chest pain can support thromboembolic ischemic pathology."),
            _evidence("hypoxemia", "Hypoxemia", positive_lr=1.3, negative_lr=0.95, rationale="Hypoxemia supports high-acuity pulmonary vascular processes."),
            _evidence("rv_strain", "RV strain", positive_lr=3.0, negative_lr=0.72, rationale="RV strain strongly supports acute thrombo-obstructive physiology."),
        ],
        contradicting_findings=[],
    ),
    "medication_toxicity_effect": LatentStateDefinition(
        slug="medication_toxicity_effect",
        name="Medication / Toxicity Effect",
        category="iatrogenic",
        prior=0.11,
        description="Medication-related physiology is likely contributing to the presentation.",
        supporting_findings=[
            _evidence("anticoagulated", "On anticoagulation", positive_lr=2.2, negative_lr=0.88, rationale="Anticoagulation increases the chance of medication-driven physiology."),
            _evidence("active_gi_bleeding", "Active gastrointestinal bleeding", positive_lr=1.6, negative_lr=0.96, rationale="Bleeding on anticoagulation can reflect medication effect."),
            _evidence("symptomatic_anemia", "Symptomatic anemia", positive_lr=1.4, negative_lr=0.98, rationale="Medication-related harm may present through anemia or bleeding."),
        ],
        contradicting_findings=[],
    ),
    "obstructive_physiology": LatentStateDefinition(
        slug="obstructive_physiology",
        name="Obstructive Physiology",
        category="hemodynamics",
        prior=0.08,
        description="Forward flow limitation from obstructive cardiopulmonary physiology such as PE with RV strain.",
        supporting_findings=[
            _evidence("pleuritic_chest_pain", "Pleuritic chest pain", positive_lr=1.45, negative_lr=0.97, rationale="Pleuritic pain supports a pulmonary vascular obstructive process."),
            _evidence("hypoxemia", "Hypoxemia", positive_lr=1.4, negative_lr=0.94, rationale="Hypoxemia raises concern for obstructive pulmonary physiology."),
            _evidence("tachycardia", "Tachycardia", positive_lr=1.3, negative_lr=0.96, rationale="Tachycardia supports high-acuity obstructive physiology."),
            _evidence("rv_strain", "RV strain", positive_lr=3.5, negative_lr=0.7, rationale="RV strain strongly supports obstructive cardiopulmonary physiology."),
            _evidence("hemodynamic_instability", "Hemodynamic instability", positive_lr=1.55, negative_lr=0.93, rationale="Instability can accompany severe obstructive states."),
        ],
        contradicting_findings=[],
    ),
}

