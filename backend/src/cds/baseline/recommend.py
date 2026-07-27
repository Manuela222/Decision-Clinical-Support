"""Phase 8: deterministic, most-common-treatment-by-diagnosis baseline."""
from datetime import datetime, timezone
from typing import Optional

from ..safety import check_hypertension_compatibility
from ..schemas import ClinicalState, MedicationAction, MedicationRecommendation, RecommendationMethod, RecommendationResult
from .stats import DiagnosisMedicationStats

BASELINE_MODEL_VERSION = "baseline-v1"


def recommend_baseline(
    clinical_state: ClinicalState,
    diagnosis_medication_stats: DiagnosisMedicationStats,
    min_frequency: float = 0.3,
    top_k: Optional[int] = None,
    generated_at: Optional[datetime] = None,
) -> RecommendationResult:
    """Recommend the most common discharge medication classes for this
    patient's admission_reason, ignoring every other detail of the patient
    (no labs, no notes, no individual nuance) — that's the point of a
    baseline. Falls back to the training-set-wide statistic if
    `admission_reason` was never seen in training.
    """
    ranked = diagnosis_medication_stats.by_admission_reason.get(clinical_state.admission_reason)
    used_fallback = ranked is None
    if ranked is None:
        ranked = diagnosis_medication_stats.overall

    selected = [(cls, freq) for cls, freq in ranked if freq >= min_frequency]
    if top_k is not None:
        selected = selected[:top_k]

    current_class_names = {m.medication_class for m in clinical_state.current_medications}

    recommendations = []
    for medication_class, frequency in selected:
        compatible, reasoning = check_hypertension_compatibility(medication_class, clinical_state.hypertension_status)
        action = MedicationAction.CONTINUE if medication_class in current_class_names else MedicationAction.START
        fallback_note = (
            " (this admission reason was not seen in training; falling back to the cohort-wide statistic)"
            if used_fallback
            else ""
        )
        rationale = (
            f"'{medication_class}' appears in {frequency * 100:.0f}% of training-set discharges for "
            f"admission reason '{clinical_state.admission_reason}'{fallback_note}. Deterministic "
            f"population-level baseline — no individual patient evidence considered."
        )
        recommendations.append(
            MedicationRecommendation(
                medication_class=medication_class,
                action=action,
                rationale=rationale,
                evidence_ids=[],
                confidence=frequency,
                hypertension_compatible=compatible,
                hypertension_reasoning=reasoning,
            )
        )

    return RecommendationResult(
        subject_id=clinical_state.subject_id,
        hadm_id=clinical_state.hadm_id,
        method=RecommendationMethod.BASELINE,
        recommended_medications=recommendations,
        safety_warnings=[],
        reasoning_trace=[],
        model_version=BASELINE_MODEL_VERSION,
        generated_at=generated_at or datetime.now(timezone.utc),
    )
