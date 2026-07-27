"""Phase 8: baseline recommender tests."""
from datetime import datetime, timezone

import pytest

from cds.baseline import compute_diagnosis_medication_stats, recommend_baseline
from cds.schemas import ClinicalState, Gender, HypertensionStatus, MedicationAction, MedicationMention, RecommendationMethod


def _clinical_state(
    admission_reason="Cardiac arrhythmias",
    hypertension_status=HypertensionStatus.CONFIRMED_CHRONIC,
    current_medications=None,
) -> ClinicalState:
    return ClinicalState(
        subject_id=1,
        hadm_id=100,
        age=67,
        gender=Gender.F,
        admission_reason=admission_reason,
        hypertension_status=hypertension_status,
        hypertension_evidence=[],
        recent_blood_pressure_labs=[],
        current_antihypertensive_medications=[],
        active_conditions=[admission_reason],
        chronic_conditions=[admission_reason],
        recent_labs=[],
        current_medications=current_medications or [],
        prior_medications=[],
        relevant_note_sections=[],
        missing_information=[],
        generated_at=datetime.now(timezone.utc),
    )


# --- compute_diagnosis_medication_stats -----------------------------------

def test_compute_stats_frequencies_and_sort_order():
    reasons = ["Cardiac arrhythmias"] * 4 + ["Diabetes, uncomplicated"] * 2
    classes = [
        ["antiarrhythmic", "statin"],
        ["antiarrhythmic"],
        ["antiarrhythmic", "anticoagulant"],
        ["statin"],
        ["antidiabetic - biguanide"],
        ["antidiabetic - biguanide", "statin"],
    ]
    stats = compute_diagnosis_medication_stats(reasons, classes)

    arrhythmia_stats = dict(stats.by_admission_reason["Cardiac arrhythmias"])
    assert arrhythmia_stats["antiarrhythmic"] == pytest.approx(3 / 4)
    assert arrhythmia_stats["statin"] == pytest.approx(2 / 4)
    assert arrhythmia_stats["anticoagulant"] == pytest.approx(1 / 4)
    # sorted descending by frequency
    ranked_classes = [cls for cls, _ in stats.by_admission_reason["Cardiac arrhythmias"]]
    assert ranked_classes[0] == "antiarrhythmic"

    diabetes_stats = dict(stats.by_admission_reason["Diabetes, uncomplicated"])
    assert diabetes_stats["antidiabetic - biguanide"] == pytest.approx(1.0)
    assert diabetes_stats["statin"] == pytest.approx(0.5)


def test_compute_stats_overall_fallback_covers_all_admissions():
    reasons = ["A", "B"]
    classes = [["x"], ["x", "y"]]
    stats = compute_diagnosis_medication_stats(reasons, classes)
    overall = dict(stats.overall)
    assert overall["x"] == pytest.approx(1.0)
    assert overall["y"] == pytest.approx(0.5)


def test_compute_stats_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        compute_diagnosis_medication_stats(["A", "B"], [["x"]])


def test_compute_stats_empty_input():
    stats = compute_diagnosis_medication_stats([], [])
    assert stats.by_admission_reason == {}
    assert stats.overall == []


# --- recommend_baseline ---------------------------------------------------

@pytest.fixture
def stats():
    reasons = ["Cardiac arrhythmias"] * 4 + ["Diabetes, uncomplicated"] * 2
    classes = [
        ["antiarrhythmic", "statin"],
        ["antiarrhythmic"],
        ["antiarrhythmic", "anticoagulant"],
        ["statin"],
        ["antidiabetic - biguanide"],
        ["nsaid"],  # only 1/2 of diabetes admissions -- below default threshold
    ]
    return compute_diagnosis_medication_stats(reasons, classes)


def test_recommend_baseline_selects_above_min_frequency(stats):
    state = _clinical_state(admission_reason="Cardiac arrhythmias")
    result = recommend_baseline(state, stats, min_frequency=0.6)
    classes = {r.medication_class for r in result.recommended_medications}
    assert classes == {"antiarrhythmic"}  # 3/4 = 0.75 >= 0.6; statin 0.5 and anticoagulant 0.25 excluded


def test_recommend_baseline_falls_back_for_unseen_admission_reason(stats):
    state = _clinical_state(admission_reason="Never Seen Condition")
    result = recommend_baseline(state, stats, min_frequency=0.4)
    assert len(result.recommended_medications) > 0
    assert "not seen in training" in result.recommended_medications[0].rationale


def test_recommend_baseline_respects_top_k(stats):
    state = _clinical_state(admission_reason="Cardiac arrhythmias")
    result = recommend_baseline(state, stats, min_frequency=0.0, top_k=1)
    assert len(result.recommended_medications) == 1
    assert result.recommended_medications[0].medication_class == "antiarrhythmic"


def test_recommend_baseline_action_continue_vs_start(stats):
    existing_med = MedicationMention(
        normalized_name="amiodarone", medication_class="antiarrhythmic", is_antihypertensive=False, evidence_id="ev-1"
    )
    state = _clinical_state(admission_reason="Cardiac arrhythmias", current_medications=[existing_med])
    result = recommend_baseline(state, stats, min_frequency=0.6)
    rec = result.recommended_medications[0]
    assert rec.medication_class == "antiarrhythmic"
    assert rec.action == MedicationAction.CONTINUE

    state_no_med = _clinical_state(admission_reason="Cardiac arrhythmias")
    result_no_med = recommend_baseline(state_no_med, stats, min_frequency=0.6)
    assert result_no_med.recommended_medications[0].action == MedicationAction.START


def test_recommend_baseline_flags_hypertension_unsafe_class(stats):
    state = _clinical_state(
        admission_reason="Diabetes, uncomplicated", hypertension_status=HypertensionStatus.CONFIRMED_CHRONIC
    )
    result = recommend_baseline(state, stats, min_frequency=0.4)  # nsaid at 0.5 makes the cut
    nsaid_rec = next(r for r in result.recommended_medications if r.medication_class == "nsaid")
    assert nsaid_rec.hypertension_compatible is False
    assert "hypertension" in nsaid_rec.hypertension_reasoning.lower()


def test_recommend_baseline_hypertension_not_present_marks_compatible(stats):
    state = _clinical_state(
        admission_reason="Diabetes, uncomplicated", hypertension_status=HypertensionStatus.NOT_PRESENT
    )
    result = recommend_baseline(state, stats, min_frequency=0.4)
    nsaid_rec = next(r for r in result.recommended_medications if r.medication_class == "nsaid")
    assert nsaid_rec.hypertension_compatible is True


def test_recommend_baseline_result_shape(stats):
    state = _clinical_state()
    result = recommend_baseline(state, stats)
    assert result.method == RecommendationMethod.BASELINE
    assert result.subject_id == state.subject_id
    assert result.hadm_id == state.hadm_id
    assert result.safety_warnings == []
    assert result.reasoning_trace == []
    assert all(r.evidence_ids == [] for r in result.recommended_medications)


def test_recommend_baseline_is_deterministic(stats):
    state = _clinical_state()
    a = recommend_baseline(state, stats)
    b = recommend_baseline(state, stats)
    assert [r.model_dump() for r in a.recommended_medications] == [r.model_dump() for r in b.recommended_medications]
