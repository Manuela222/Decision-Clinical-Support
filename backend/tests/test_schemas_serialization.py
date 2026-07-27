"""Every schema must round-trip through JSON without loss: model -> JSON string
-> parsed back into an equal model. This is the Phase 1 requirement that all
structured objects be JSON-serializable."""
import json

import pytest
from pydantic import BaseModel

from cds.schemas import (
    ClinicalState,
    EvaluationResult,
    MethodResultState,
    PatientTimeline,
    PerAdmissionEvaluation,
    RecommendationMethod,
    RecommendationResult,
    RequestStatus,
    TimelineEvent,
    UIPage,
    UIState,
)


def assert_round_trips(model: BaseModel) -> None:
    json_str = model.model_dump_json()
    parsed_dict = json.loads(json_str)  # must be plain JSON, no leftover Python objects
    rebuilt = type(model).model_validate_json(json_str)
    assert rebuilt == model
    assert parsed_dict == json.loads(rebuilt.model_dump_json())


def test_evidence_item_round_trips(evidence_item):
    assert_round_trips(evidence_item)


def test_timeline_event_round_trips(timeline_event: TimelineEvent):
    assert_round_trips(timeline_event)


def test_patient_timeline_round_trips(patient_timeline: PatientTimeline):
    assert_round_trips(patient_timeline)


def test_clinical_state_round_trips(clinical_state: ClinicalState):
    assert_round_trips(clinical_state)


def test_clinical_state_exposes_hypertension_fields(clinical_state: ClinicalState):
    assert clinical_state.hypertension_status.value == "confirmed_chronic"
    assert clinical_state.current_antihypertensive_medications[0].is_antihypertensive is True


def test_medication_recommendation_round_trips(medication_recommendation):
    assert_round_trips(medication_recommendation)


def test_medication_recommendation_requires_hypertension_reasoning():
    from cds.schemas import MedicationAction, MedicationRecommendation

    with pytest.raises(Exception):
        MedicationRecommendation(
            medication_class="nsaid",
            action=MedicationAction.AVOID,
            rationale="pain control",
            evidence_ids=[],
            confidence=0.5,
            hypertension_compatible=False,
            # hypertension_reasoning omitted on purpose
        )


def test_safety_warning_round_trips(safety_warning):
    assert_round_trips(safety_warning)


def test_reasoning_step_round_trips(reasoning_step):
    assert_round_trips(reasoning_step)


def test_recommendation_result_round_trips(recommendation_result: RecommendationResult):
    assert_round_trips(recommendation_result)


def test_evaluation_result_round_trips():
    per_admission = PerAdmissionEvaluation(
        subject_id=42,
        hadm_id=100001,
        method=RecommendationMethod.AGENT,
        predicted_classes=["beta blocker", "statin"],
        ground_truth_classes=["beta blocker", "anticoagulant"],
        matched_classes=["beta blocker"],
        missed_classes=["anticoagulant"],
        extra_classes=["statin"],
        precision=0.5,
        recall=0.5,
        f1=0.5,
    )
    result = EvaluationResult(
        method=RecommendationMethod.AGENT,
        n_admissions_evaluated=1,
        micro_precision=0.5,
        micro_recall=0.5,
        micro_f1=0.5,
        macro_precision=0.5,
        macro_recall=0.5,
        macro_f1=0.5,
        per_admission=[per_admission],
        notes="smoke test",
    )
    assert_round_trips(per_admission)
    assert_round_trips(result)


def test_ui_state_round_trips(clinical_state: ClinicalState, recommendation_result: RecommendationResult):
    state = UIState(
        page=UIPage.EXISTING_PATIENT,
        selected_subject_id=42,
        selected_hadm_id=100001,
        clinical_state=clinical_state,
        agent_result=MethodResultState(status=RequestStatus.SUCCESS, result=recommendation_result),
        ground_truth_medications=["beta blocker", "anticoagulant"],
    )
    assert_round_trips(state)


def test_ui_state_defaults_are_idle():
    state = UIState(page=UIPage.NEW_PATIENT)
    assert state.baseline_result.status == RequestStatus.IDLE
    assert state.baseline_result.result is None
    assert_round_trips(state)
