"""Service layer: route handlers (routes.py) call these; these call `cds`'s
business logic. No FastAPI-specific concerns (status codes, HTTP exceptions)
live here — routes.py translates the exceptions below into HTTP responses.
"""
from datetime import datetime, timezone
from typing import List, Optional

from cds.agent import recommend_agent
from cds.baseline import recommend_baseline
from cds.evaluation import ComparisonReport, evaluate_all_methods
from cds.explainability import ExplainabilityReport, explain_agent, explain_baseline, explain_trained_model
from cds.mcp_tools import MCPToolContext
from cds.medications import default_dictionary, is_antihypertensive_class
from cds.model import recommend_trained_model
from cds.synthetic import generate_synthetic_clinical_state
from cds.schemas import (
    ClinicalState,
    Gender,
    HypertensionStatus,
    LabValueSummary,
    MedicationMention,
    PatientTimeline,
    RecommendationMethod,
    RecommendationResult,
)

from .app_state import AppState, CohortEntry
from .schemas import NewPatientRequest


class NotFoundError(Exception):
    pass


class ValidationError(Exception):
    pass


def get_cohort(state: AppState) -> List[CohortEntry]:
    return state.test_cohort()


def get_timeline(state: AppState, subject_id: int, hadm_id: int) -> PatientTimeline:
    timeline = state.get_timeline(subject_id, hadm_id)
    if timeline is None:
        raise NotFoundError(f"No timeline found for subject_id={subject_id}, hadm_id={hadm_id}")
    return timeline


def get_clinical_state(state: AppState, subject_id: int, hadm_id: int) -> ClinicalState:
    clinical_state = state.get_clinical_state(subject_id, hadm_id)
    if clinical_state is None:
        raise NotFoundError(f"No clinical state found for subject_id={subject_id}, hadm_id={hadm_id}")
    return clinical_state


def _build_mcp_context(state: AppState, clinical_state: ClinicalState, timeline: PatientTimeline) -> MCPToolContext:
    return MCPToolContext(
        clinical_state=clinical_state, patient_timeline=timeline, patient_profile_index=state.patient_profile_index
    )


def predict_baseline(state: AppState, subject_id: int, hadm_id: int) -> RecommendationResult:
    clinical_state = get_clinical_state(state, subject_id, hadm_id)
    return recommend_baseline(clinical_state, state.diagnosis_medication_stats)


def predict_model(state: AppState, subject_id: int, hadm_id: int) -> RecommendationResult:
    clinical_state = get_clinical_state(state, subject_id, hadm_id)
    return recommend_trained_model(clinical_state, state.trained_model_artifact)


def predict_agent(state: AppState, subject_id: int, hadm_id: int) -> RecommendationResult:
    clinical_state = get_clinical_state(state, subject_id, hadm_id)
    timeline = get_timeline(state, subject_id, hadm_id)
    ctx = _build_mcp_context(state, clinical_state, timeline)
    return recommend_agent(clinical_state, ctx, state.llm_provider_factory())


def _build_new_patient_clinical_state(request: NewPatientRequest) -> ClinicalState:
    dictionary = default_dictionary()
    known_classes = set(dictionary.classes.values())
    unknown = [c for c in request.current_medication_classes if c not in known_classes]
    if unknown:
        raise ValidationError(f"Unknown medication class(es), not in the fixed vocabulary: {unknown}")

    try:
        hypertension_status = HypertensionStatus(request.hypertension_status)
    except ValueError as e:
        raise ValidationError(f"Invalid hypertension_status: '{request.hypertension_status}'") from e
    try:
        gender = Gender(request.gender)
    except ValueError as e:
        raise ValidationError(f"Invalid gender: '{request.gender}'") from e

    current_medications = [
        MedicationMention(
            normalized_name=cls,
            medication_class=cls,
            is_antihypertensive=is_antihypertensive_class(cls),
            evidence_id=f"new-patient-med-{i}",
        )
        for i, cls in enumerate(request.current_medication_classes)
    ]
    recent_labs = []
    if request.recent_creatinine is not None:
        recent_labs.append(
            LabValueSummary(
                itemid=-1, label="Creatinine", value=request.recent_creatinine, unit="mg/dL",
                flag=None, charttime=None, evidence_id="new-patient-lab-0",
            )
        )

    active_conditions = [request.admission_reason]
    if hypertension_status == HypertensionStatus.CONFIRMED_CHRONIC:
        active_conditions.append("Hypertension")

    return ClinicalState(
        subject_id=-1,
        hadm_id=-1,
        age=request.age,
        gender=gender,
        admission_reason=request.admission_reason,
        hypertension_status=hypertension_status,
        hypertension_evidence=[],
        recent_blood_pressure_labs=[],
        current_antihypertensive_medications=[m for m in current_medications if m.is_antihypertensive],
        active_conditions=active_conditions,
        chronic_conditions=active_conditions,
        recent_labs=recent_labs,
        current_medications=current_medications,
        prior_medications=[],
        relevant_note_sections=[],
        missing_information=[],
        generated_at=datetime.now(timezone.utc),
    )


def predict_agent_new_patient(state: AppState, request: NewPatientRequest) -> RecommendationResult:
    clinical_state = _build_new_patient_clinical_state(request)
    timeline = PatientTimeline(subject_id=-1, hadm_id=-1, events=[], generated_at=datetime.now(timezone.utc))
    ctx = _build_mcp_context(state, clinical_state, timeline)
    return recommend_agent(clinical_state, ctx, state.llm_provider_factory())


def get_synthetic_new_patient_example() -> NewPatientRequest:
    """Not one of Phase 16's originally-listed endpoints -- added because
    Phase 17's new-patient page explicitly needs a way to auto-fill the
    form with a synthetic example from Phase 7, and that generator's logic
    belongs in `cds`, not reimplemented in frontend JS."""
    synthetic = generate_synthetic_clinical_state()
    creatinine = next(
        (lab.value for lab in synthetic.recent_labs if "creatinine" in lab.label.lower()), None
    )
    return NewPatientRequest(
        age=synthetic.age,
        gender=synthetic.gender.value,
        admission_reason=synthetic.admission_reason,
        hypertension_status=synthetic.hypertension_status.value,
        current_medication_classes=[m.medication_class for m in synthetic.current_medications],
        recent_creatinine=creatinine,
    )


def _placeholder_agent_result(subject_id: int, hadm_id: int) -> RecommendationResult:
    return RecommendationResult(
        subject_id=subject_id,
        hadm_id=hadm_id,
        method=RecommendationMethod.AGENT,
        recommended_medications=[],
        safety_warnings=[],
        reasoning_trace=[],
        model_version="not-run",
        generated_at=datetime.now(timezone.utc),
    )


def run_evaluation(state: AppState, hadm_ids: Optional[List[int]], include_agent: bool) -> ComparisonReport:
    test_entries = [c for c in state.test_cohort() if hadm_ids is None or c.hadm_id in hadm_ids]
    if not test_entries:
        raise ValidationError("No matching test-set admissions to evaluate.")

    ground_truth = [state.get_ground_truth(c.subject_id, c.hadm_id) for c in test_entries]
    baseline_results = [predict_baseline(state, c.subject_id, c.hadm_id) for c in test_entries]
    model_results = [predict_model(state, c.subject_id, c.hadm_id) for c in test_entries]
    agent_results = (
        [predict_agent(state, c.subject_id, c.hadm_id) for c in test_entries]
        if include_agent
        else [_placeholder_agent_result(c.subject_id, c.hadm_id) for c in test_entries]
    )

    return evaluate_all_methods(ground_truth, baseline_results, model_results, agent_results)


_EXPLAIN_METHODS = {"baseline", "trained_model", "agent"}


def explain(state: AppState, subject_id: int, hadm_id: int, method: str) -> ExplainabilityReport:
    if method not in _EXPLAIN_METHODS:
        raise ValidationError(f"Unknown method '{method}'. Must be one of: {sorted(_EXPLAIN_METHODS)}.")

    clinical_state = get_clinical_state(state, subject_id, hadm_id)
    ground_truth = state.get_ground_truth(subject_id, hadm_id)

    if method == "baseline":
        result = predict_baseline(state, subject_id, hadm_id)
        return explain_baseline(clinical_state, result, state.diagnosis_medication_stats, ground_truth)
    if method == "trained_model":
        result = predict_model(state, subject_id, hadm_id)
        return explain_trained_model(clinical_state, result, state.trained_model_artifact, ground_truth)
    result = predict_agent(state, subject_id, hadm_id)
    return explain_agent(clinical_state, result, ground_truth)
