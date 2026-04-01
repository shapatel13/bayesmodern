from __future__ import annotations

from core.models import HypothesisEvidence, LikelihoodRatioRange


LR_CATALOG: dict[str, list[HypothesisEvidence]] = {
    "pe": [
        HypothesisEvidence(
            finding_key="pleuritic_chest_pain",
            label="Pleuritic chest pain",
            lr=LikelihoodRatioRange(positive_lr=1.8, negative_lr=0.8, positive_lr_low=1.2, positive_lr_high=2.4),
            rationale="Pleuritic pain increases suspicion for thromboembolic pleural irritation.",
            provenance_refs=["rule:wells_pe"],
        ),
        HypothesisEvidence(
            finding_key="tachycardia",
            label="Tachycardia",
            lr=LikelihoodRatioRange(positive_lr=1.7, negative_lr=0.7, positive_lr_low=1.1, positive_lr_high=2.3),
            rationale="Tachycardia is supportive but non-specific in PE suspicion.",
            provenance_refs=["rule:wells_pe"],
        ),
        HypothesisEvidence(
            finding_key="hypoxemia",
            label="Hypoxemia",
            lr=LikelihoodRatioRange(positive_lr=2.0, negative_lr=0.6, positive_lr_low=1.3, positive_lr_high=2.8),
            rationale="Gas exchange impairment supports clinically meaningful PE.",
            provenance_refs=["rule:wells_pe"],
        ),
        HypothesisEvidence(
            finding_key="fever",
            label="Fever",
            lr=LikelihoodRatioRange(positive_lr=0.7, negative_lr=1.1, positive_lr_low=0.5, positive_lr_high=0.9),
            rationale="Fever modestly lowers PE likelihood relative to infectious alternatives.",
            provenance_refs=["rule:wells_pe"],
        ),
    ],
    "pneumonia": [
        HypothesisEvidence(
            finding_key="fever",
            label="Fever",
            lr=LikelihoodRatioRange(positive_lr=2.1, negative_lr=0.5, positive_lr_low=1.5, positive_lr_high=2.8),
            rationale="Fever materially increases the likelihood of lower respiratory infection.",
            provenance_refs=["study:cxr_pneumonia"],
        ),
        HypothesisEvidence(
            finding_key="crackles",
            label="Focal crackles",
            lr=LikelihoodRatioRange(positive_lr=2.5, negative_lr=0.7, positive_lr_low=1.8, positive_lr_high=3.3),
            rationale="Focal auscultatory findings support parenchymal involvement.",
            provenance_refs=["study:cxr_pneumonia"],
        ),
        HypothesisEvidence(
            finding_key="purulent_sputum",
            label="Purulent sputum",
            lr=LikelihoodRatioRange(positive_lr=1.9, negative_lr=0.8, positive_lr_low=1.3, positive_lr_high=2.6),
            rationale="Purulent sputum increases the likelihood of bacterial pneumonia.",
            provenance_refs=["study:cxr_pneumonia"],
        ),
    ],
    "heart_failure": [
        HypothesisEvidence(
            finding_key="orthopnea",
            label="Orthopnea",
            lr=LikelihoodRatioRange(positive_lr=2.7, negative_lr=0.5, positive_lr_low=1.9, positive_lr_high=3.6),
            rationale="Orthopnea favors cardiogenic pulmonary congestion.",
            provenance_refs=["study:bnp_hf"],
        ),
        HypothesisEvidence(
            finding_key="crackles",
            label="Diffuse crackles",
            lr=LikelihoodRatioRange(positive_lr=2.0, negative_lr=0.7, positive_lr_low=1.3, positive_lr_high=2.8),
            rationale="Pulmonary edema physiology often presents with crackles.",
            provenance_refs=["study:bnp_hf"],
        ),
        HypothesisEvidence(
            finding_key="leg_edema",
            label="Leg edema",
            lr=LikelihoodRatioRange(positive_lr=1.8, negative_lr=0.8, positive_lr_low=1.2, positive_lr_high=2.5),
            rationale="Peripheral edema adds support for systemic congestion.",
            provenance_refs=["study:bnp_hf"],
        ),
    ],
    "acs": [
        HypothesisEvidence(
            finding_key="pressure_chest_pain",
            label="Pressure-like chest pain",
            lr=LikelihoodRatioRange(positive_lr=2.2, negative_lr=0.7, positive_lr_low=1.5, positive_lr_high=2.9),
            rationale="Pressure-like chest discomfort supports ischemic chest pain.",
            provenance_refs=["registry:acs_symptom_profile"],
        ),
        HypothesisEvidence(
            finding_key="troponin_positive",
            label="Positive troponin",
            lr=LikelihoodRatioRange(positive_lr=4.5, negative_lr=0.4, positive_lr_low=3.0, positive_lr_high=6.0),
            rationale="Troponin elevation strongly supports myocardial injury etiologies.",
            provenance_refs=["registry:acs_symptom_profile"],
        ),
    ],
}

