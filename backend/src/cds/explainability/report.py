"""Phase 15: build an ExplainabilityReport for an EXISTING RecommendationResult.

No new recommendation logic lives here — each `explain_*` function only
reorganizes/derives from what Phase 8/9/13 already produced (plus, for the
trained model, its saved artifact's feature importances; for the agent, its
own reasoning trace) into one structured, UI-renderable report.
"""
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd

from ..baseline import DiagnosisMedicationStats
from ..model import ALL_FEATURE_COLUMNS, TrainedModelArtifact, build_structured_features
from ..retrieval import PatientProfileIndex, SimilarPatientProfile
from ..safety import check_recommendation_safety
from ..schemas import ClinicalState, EvidenceItem, ReasoningStepType, RecommendationResult
from .feature_importance import get_feature_importances
from .schemas import ExplainabilityReport, MedicationCandidate


def _matched_missed_extra(predicted: List[str], ground_truth: List[str]):
    predicted_set, truth_set = set(predicted), set(ground_truth)
    return sorted(predicted_set & truth_set), sorted(truth_set - predicted_set), sorted(predicted_set - truth_set)


def _base_report(
    clinical_state: ClinicalState,
    result: RecommendationResult,
    ground_truth_medications: List[str],
    medication_candidates: List[MedicationCandidate],
    retrieved_similar_profiles: Optional[List[SimilarPatientProfile]] = None,
    feature_importances=None,
    evidence_citations=None,
) -> ExplainabilityReport:
    predicted = [rec.medication_class for rec in result.recommended_medications]
    matched, missed, extra = _matched_missed_extra(predicted, ground_truth_medications)

    return ExplainabilityReport(
        subject_id=clinical_state.subject_id,
        hadm_id=clinical_state.hadm_id,
        method=result.method,
        patient_profile=clinical_state,
        retrieved_similar_profiles=retrieved_similar_profiles or [],
        feature_importances=feature_importances or [],
        medication_candidates=medication_candidates,
        safety_warnings=result.safety_warnings or check_recommendation_safety(clinical_state, result.recommended_medications),
        evidence_citations=evidence_citations or [],
        ground_truth_medications=list(ground_truth_medications),
        matched_classes=matched,
        missed_classes=missed,
        extra_classes=extra,
        generated_at=datetime.now(timezone.utc),
    )


def explain_baseline(
    clinical_state: ClinicalState,
    result: RecommendationResult,
    diagnosis_medication_stats: DiagnosisMedicationStats,
    ground_truth_medications: List[str],
) -> ExplainabilityReport:
    accepted_classes = {rec.medication_class for rec in result.recommended_medications}
    ranked = diagnosis_medication_stats.by_admission_reason.get(clinical_state.admission_reason)
    if ranked is None:
        ranked = diagnosis_medication_stats.overall

    candidates = [
        MedicationCandidate(
            medication_class=cls,
            accepted=cls in accepted_classes,
            score=frequency,
            reason=None if cls in accepted_classes else "training-set frequency for this admission condition did not meet the recommendation threshold",
        )
        for cls, frequency in ranked
    ]
    return _base_report(clinical_state, result, ground_truth_medications, candidates)


def explain_trained_model(
    clinical_state: ClinicalState,
    result: RecommendationResult,
    artifact: TrainedModelArtifact,
    ground_truth_medications: List[str],
) -> ExplainabilityReport:
    accepted_classes = {rec.medication_class for rec in result.recommended_medications}
    features_df = build_structured_features([clinical_state])[ALL_FEATURE_COLUMNS]
    proba_per_label = artifact.pipeline.predict_proba(features_df)

    candidates = [
        MedicationCandidate(
            medication_class=label,
            accepted=label in accepted_classes,
            score=float(proba_per_label[i][0, 1]),
            reason=None if label in accepted_classes else "predicted probability below the recommendation threshold",
        )
        for i, label in enumerate(artifact.label_binarizer.classes_)
    ]
    feature_importances = get_feature_importances(artifact)
    return _base_report(
        clinical_state, result, ground_truth_medications, candidates, feature_importances=feature_importances
    )


def explain_agent(
    clinical_state: ClinicalState,
    result: RecommendationResult,
    ground_truth_medications: List[str],
) -> ExplainabilityReport:
    accepted_classes = {rec.medication_class for rec in result.recommended_medications}

    considered_classes = set()
    retrieved_similar_profiles: List[SimilarPatientProfile] = []
    evidence_citations: List[EvidenceItem] = []

    for step in result.reasoning_trace:
        if step.tool_name == "check_medication_compatibility" and step.tool_input:
            cls = step.tool_input.get("medication_class")
            if cls:
                considered_classes.add(cls)
        if step.tool_name == "get_evidence_citations" and step.tool_output:
            items = step.tool_output.get("result", [])
            for item in items:
                try:
                    evidence_citations.append(EvidenceItem.model_validate(item))
                except Exception:  # noqa: BLE001 - a malformed trace entry shouldn't break the whole report
                    continue
        if step.step_type == ReasoningStepType.RETRIEVAL and step.tool_output:
            profiles = step.tool_output.get("result", [])
            for profile in profiles:
                try:
                    retrieved_similar_profiles.append(SimilarPatientProfile.model_validate(profile))
                except Exception:  # noqa: BLE001 - a malformed trace entry shouldn't break the whole report
                    continue

    considered_classes |= accepted_classes
    candidates = [
        MedicationCandidate(
            medication_class=cls,
            accepted=cls in accepted_classes,
            score=None,
            reason=None if cls in accepted_classes else "considered (compatibility checked) but not included in the agent's final answer",
        )
        for cls in sorted(considered_classes)
    ]

    return _base_report(
        clinical_state,
        result,
        ground_truth_medications,
        candidates,
        retrieved_similar_profiles=retrieved_similar_profiles,
        evidence_citations=evidence_citations,
    )
