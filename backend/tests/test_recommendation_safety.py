"""Phase 10, part 1: rule-based recommendation safety checks."""
from datetime import datetime, timezone

from cds.safety import check_recommendation_safety
from cds.schemas import (
    ClinicalState,
    Gender,
    HypertensionStatus,
    LabValueSummary,
    MedicationAction,
    MedicationRecommendation,
    SafetyCategory,
    SafetySeverity,
)


def _clinical_state(recent_labs=None) -> ClinicalState:
    return ClinicalState(
        subject_id=1,
        hadm_id=100,
        age=67,
        gender=Gender.F,
        admission_reason="Cardiac arrhythmias",
        hypertension_status=HypertensionStatus.CONFIRMED_CHRONIC,
        hypertension_evidence=[],
        recent_blood_pressure_labs=[],
        current_antihypertensive_medications=[],
        active_conditions=["Cardiac arrhythmias", "Hypertension"],
        chronic_conditions=["Cardiac arrhythmias", "Hypertension"],
        recent_labs=recent_labs or [],
        current_medications=[],
        prior_medications=[],
        relevant_note_sections=[],
        missing_information=[],
        generated_at=datetime.now(timezone.utc),
    )


def _rec(
    medication_class="statin",
    rationale="Indicated for rate control.",
    evidence_ids=None,
    hypertension_compatible=True,
    hypertension_reasoning="No known interaction.",
) -> MedicationRecommendation:
    return MedicationRecommendation(
        medication_class=medication_class,
        action=MedicationAction.START,
        rationale=rationale,
        evidence_ids=evidence_ids if evidence_ids is not None else ["ev-1"],
        confidence=0.8,
        hypertension_compatible=hypertension_compatible,
        hypertension_reasoning=hypertension_reasoning,
    )


def test_clean_recommendation_has_no_warnings():
    state = _clinical_state()
    warnings = check_recommendation_safety(state, [_rec()])
    assert warnings == []


def test_duplicate_medication_class_flagged():
    state = _clinical_state()
    warnings = check_recommendation_safety(state, [_rec(), _rec()])
    duplicate_warnings = [w for w in warnings if w.category == SafetyCategory.DUPLICATE_MEDICATION_CLASS]
    assert len(duplicate_warnings) == 1
    assert duplicate_warnings[0].severity == SafetySeverity.WARNING


def test_hypertension_unsafe_flagged_critical():
    state = _clinical_state()
    rec = _rec(
        medication_class="nsaid",
        hypertension_compatible=False,
        hypertension_reasoning="NSAIDs can raise blood pressure.",
    )
    warnings = check_recommendation_safety(state, [rec])
    htn_warnings = [w for w in warnings if w.category == SafetyCategory.HYPERTENSION_UNSAFE]
    assert len(htn_warnings) == 1
    assert htn_warnings[0].severity == SafetySeverity.CRITICAL
    assert htn_warnings[0].related_medication_class == "nsaid"


def test_renal_sensitive_missing_creatinine_flagged_warning():
    state = _clinical_state(recent_labs=[])
    rec = _rec(medication_class="ace inhibitor")
    warnings = check_recommendation_safety(state, [rec])
    renal_warnings = [w for w in warnings if w.category == SafetyCategory.RENAL_SENSITIVE]
    assert len(renal_warnings) == 1
    assert renal_warnings[0].severity == SafetySeverity.WARNING


def test_renal_sensitive_abnormal_creatinine_flagged_critical():
    lab = LabValueSummary(
        itemid=1, label="Creatinine", value=2.4, unit="mg/dL", flag="abnormal",
        charttime=None, evidence_id="ev-cr",
    )
    state = _clinical_state(recent_labs=[lab])
    rec = _rec(medication_class="ace inhibitor")
    warnings = check_recommendation_safety(state, [rec])
    renal_warnings = [w for w in warnings if w.category == SafetyCategory.RENAL_SENSITIVE]
    assert len(renal_warnings) == 1
    assert renal_warnings[0].severity == SafetySeverity.CRITICAL
    assert renal_warnings[0].evidence_ids == ["ev-cr"]


def test_renal_sensitive_normal_creatinine_not_flagged():
    lab = LabValueSummary(
        itemid=1, label="Creatinine", value=0.9, unit="mg/dL", flag=None,
        charttime=None, evidence_id="ev-cr",
    )
    state = _clinical_state(recent_labs=[lab])
    rec = _rec(medication_class="ace inhibitor")
    warnings = check_recommendation_safety(state, [rec])
    assert not any(w.category == SafetyCategory.RENAL_SENSITIVE for w in warnings)


def test_non_renal_class_not_flagged_even_with_abnormal_creatinine():
    lab = LabValueSummary(
        itemid=1, label="Creatinine", value=2.4, unit="mg/dL", flag="abnormal",
        charttime=None, evidence_id="ev-cr",
    )
    state = _clinical_state(recent_labs=[lab])
    rec = _rec(medication_class="statin")
    warnings = check_recommendation_safety(state, [rec])
    assert not any(w.category == SafetyCategory.RENAL_SENSITIVE for w in warnings)


def test_missing_indication_flagged():
    state = _clinical_state()
    rec = _rec(rationale="")
    warnings = check_recommendation_safety(state, [rec])
    assert any(w.category == SafetyCategory.MISSING_INDICATION for w in warnings)


def test_missing_evidence_flagged_info():
    state = _clinical_state()
    rec = _rec(evidence_ids=[])
    warnings = check_recommendation_safety(state, [rec])
    missing_evidence = [w for w in warnings if w.category == SafetyCategory.MISSING_EVIDENCE]
    assert len(missing_evidence) == 1
    assert missing_evidence[0].severity == SafetySeverity.INFO


def test_warning_ids_are_unique():
    state = _clinical_state()
    rec = _rec(medication_class="nsaid", hypertension_compatible=False, rationale="", evidence_ids=[])
    warnings = check_recommendation_safety(state, [rec, rec])
    ids = [w.warning_id for w in warnings]
    assert len(ids) == len(set(ids))
