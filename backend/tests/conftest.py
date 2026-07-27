"""Shared fake-data fixtures for schema serialization tests."""
from datetime import datetime

import pytest

from cds.schemas import (
    ClinicalState,
    EvidenceItem,
    Gender,
    HypertensionStatus,
    LabValueSummary,
    MedicationAction,
    MedicationMention,
    MedicationRecommendation,
    MissingInformationFlag,
    NoteSection,
    PatientTimeline,
    ReasoningStep,
    ReasoningStepType,
    RecommendationMethod,
    RecommendationResult,
    SafetyCategory,
    SafetySeverity,
    SafetyWarning,
    SourceTable,
    TimelineEvent,
    TimelineEventType,
)


@pytest.fixture
def evidence_item() -> EvidenceItem:
    return EvidenceItem(
        evidence_id="ev-1",
        source_table=SourceTable.PRESCRIPTIONS,
        source_row_id="123456",
        subject_id=42,
        hadm_id=100001,
        description="Lisinopril 10mg ordered, ongoing at discharge",
        value="lisinopril",
        timestamp=datetime(2150, 1, 5, 8, 0, 0),
        text_excerpt=None,
    )


@pytest.fixture
def timeline_event(evidence_item: EvidenceItem) -> TimelineEvent:
    return TimelineEvent(
        event_id="evt-1",
        subject_id=42,
        hadm_id=100001,
        event_type=TimelineEventType.MEDICATION_ORDER,
        timestamp=datetime(2150, 1, 5, 8, 0, 0),
        description="Lisinopril 10mg PO daily started",
        details={"drug": "lisinopril", "dose": "10mg", "route": "PO"},
        evidence=evidence_item,
    )


@pytest.fixture
def patient_timeline(timeline_event: TimelineEvent) -> PatientTimeline:
    return PatientTimeline(
        subject_id=42,
        hadm_id=100001,
        events=[timeline_event],
        generated_at=datetime(2150, 1, 10, 12, 0, 0),
    )


@pytest.fixture
def clinical_state(evidence_item: EvidenceItem) -> ClinicalState:
    lab = LabValueSummary(
        itemid=50912,
        label="Creatinine",
        value=1.1,
        unit="mg/dL",
        flag=None,
        charttime=datetime(2150, 1, 6, 6, 0, 0),
        evidence_id="ev-2",
    )
    med = MedicationMention(
        normalized_name="lisinopril",
        medication_class="ace inhibitor",
        is_antihypertensive=True,
        evidence_id="ev-1",
    )
    note = NoteSection(
        category="Discharge summary",
        section_title="History of Present Illness",
        excerpt="Patient with hx of HTN admitted for new-onset atrial fibrillation...",
        evidence_id="ev-3",
    )
    return ClinicalState(
        subject_id=42,
        hadm_id=100001,
        age=67,
        gender=Gender.F,
        admission_reason="Cardiac arrhythmias",
        hypertension_status=HypertensionStatus.CONFIRMED_CHRONIC,
        hypertension_evidence=[evidence_item],
        recent_blood_pressure_labs=[],
        current_antihypertensive_medications=[med],
        active_conditions=["Cardiac arrhythmias", "Hypertension"],
        chronic_conditions=["Hypertension"],
        recent_labs=[lab],
        current_medications=[med],
        prior_medications=[],
        relevant_note_sections=[note],
        missing_information=[MissingInformationFlag.PRIOR_ADMISSIONS_MISSING],
        generated_at=datetime(2150, 1, 10, 12, 0, 0),
    )


@pytest.fixture
def medication_recommendation() -> MedicationRecommendation:
    return MedicationRecommendation(
        medication_class="beta blocker",
        action=MedicationAction.START,
        rationale="Rate control indicated for new-onset atrial fibrillation.",
        evidence_ids=["ev-1", "ev-2"],
        confidence=0.82,
        hypertension_compatible=True,
        hypertension_reasoning="Beta blockers are appropriate and do not conflict with the patient's chronic hypertension.",
    )


@pytest.fixture
def safety_warning() -> SafetyWarning:
    return SafetyWarning(
        warning_id="sw-1",
        severity=SafetySeverity.CRITICAL,
        category=SafetyCategory.HYPERTENSION_UNSAFE,
        message="NSAID recommended despite chronic hypertension.",
        related_medication_class="nsaid",
        evidence_ids=["ev-4"],
    )


@pytest.fixture
def reasoning_step() -> ReasoningStep:
    return ReasoningStep(
        step_index=0,
        step_type=ReasoningStepType.HYPERTENSION_COMPATIBILITY_CHECK,
        description="Checked candidate class 'beta blocker' against hypertension_status=confirmed_chronic: compatible.",
        tool_name="check_medication_compatibility",
        tool_input={"medication_class": "beta blocker"},
        tool_output={"compatible": True},
        evidence_ids=["ev-1"],
        timestamp=datetime(2150, 1, 10, 12, 0, 1),
    )


@pytest.fixture
def recommendation_result(
    medication_recommendation: MedicationRecommendation,
    safety_warning: SafetyWarning,
    reasoning_step: ReasoningStep,
) -> RecommendationResult:
    return RecommendationResult(
        subject_id=42,
        hadm_id=100001,
        method=RecommendationMethod.AGENT,
        recommended_medications=[medication_recommendation],
        safety_warnings=[safety_warning],
        reasoning_trace=[reasoning_step],
        model_version="gpt-test-mock",
        generated_at=datetime(2150, 1, 10, 12, 0, 2),
    )
