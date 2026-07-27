"""Phase 9: turn the trained model's per-class probabilities into a
RecommendationResult, in the same shape as the baseline (Phase 8) and
agent (Phase 13) so Phase 14 can score all three identically."""
from datetime import datetime, timezone
from typing import Optional

from ..safety import check_hypertension_compatibility
from ..schemas import ClinicalState, MedicationAction, MedicationRecommendation, RecommendationMethod, RecommendationResult
from .features import ALL_FEATURE_COLUMNS, build_structured_features
from .train import TrainedModelArtifact


def recommend_trained_model(
    clinical_state: ClinicalState,
    artifact: TrainedModelArtifact,
    threshold: float = 0.3,
    top_k: Optional[int] = None,
    generated_at: Optional[datetime] = None,
) -> RecommendationResult:
    features_df = build_structured_features([clinical_state])[ALL_FEATURE_COLUMNS]
    proba_per_label = artifact.pipeline.predict_proba(features_df)  # list of (1, 2) arrays, one per label

    class_confidences = [
        (label, float(proba_per_label[i][0, 1])) for i, label in enumerate(artifact.label_binarizer.classes_)
    ]
    selected = sorted(
        ((cls, conf) for cls, conf in class_confidences if conf >= threshold),
        key=lambda item: (-item[1], item[0]),
    )
    if top_k is not None:
        selected = selected[:top_k]

    current_class_names = {m.medication_class for m in clinical_state.current_medications}

    recommendations = []
    for medication_class, confidence in selected:
        compatible, reasoning = check_hypertension_compatibility(medication_class, clinical_state.hypertension_status)
        action = MedicationAction.CONTINUE if medication_class in current_class_names else MedicationAction.START
        rationale = (
            f"Trained model ({artifact.model_card.model_version}) predicts '{medication_class}' with "
            f"probability {confidence:.2f}, based on this patient's structured features and note text."
        )
        recommendations.append(
            MedicationRecommendation(
                medication_class=medication_class,
                action=action,
                rationale=rationale,
                evidence_ids=[],
                confidence=confidence,
                hypertension_compatible=compatible,
                hypertension_reasoning=reasoning,
            )
        )

    return RecommendationResult(
        subject_id=clinical_state.subject_id,
        hadm_id=clinical_state.hadm_id,
        method=RecommendationMethod.TRAINED_MODEL,
        recommended_medications=recommendations,
        safety_warnings=[],
        reasoning_trace=[],
        model_version=artifact.model_card.model_version,
        generated_at=generated_at or datetime.now(timezone.utc),
    )
