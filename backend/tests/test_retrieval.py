"""Phase 11: RAG index + focused evidence-cited retrieval function tests."""
from datetime import datetime, timezone

import pandas as pd
import pytest

from cds.retrieval import (
    build_patient_profile_index,
    find_similar_patient_profiles,
    get_diagnoses,
    get_evidence_by_ids,
    get_medications,
    get_recent_labs,
    search_notes,
)
from cds.schemas import ClinicalState, Gender, HypertensionStatus, LabValueSummary
from cds.timeline import build_patient_timeline


def _clinical_state(idx: int, admission_reason: str, age: int) -> ClinicalState:
    return ClinicalState(
        subject_id=idx,
        hadm_id=1000 + idx,
        age=age,
        gender=Gender.F,
        admission_reason=admission_reason,
        hypertension_status=HypertensionStatus.CONFIRMED_CHRONIC,
        hypertension_evidence=[],
        recent_blood_pressure_labs=[],
        current_antihypertensive_medications=[],
        active_conditions=[admission_reason, "Hypertension"],
        chronic_conditions=[admission_reason, "Hypertension"],
        recent_labs=[
            LabValueSummary(
                itemid=1, label="Creatinine", value=1.0, unit="mg/dL", flag=None,
                charttime=None, evidence_id=f"ev-{idx}",
            )
        ],
        current_medications=[],
        prior_medications=[],
        relevant_note_sections=[],
        missing_information=[],
        generated_at=datetime.now(timezone.utc),
    )


# --- build_patient_profile_index / find_similar_patient_profiles -----------

def test_build_index_mismatched_lengths_raises():
    states = [_clinical_state(1, "Cardiac arrhythmias", 60)]
    with pytest.raises(ValueError, match="same length"):
        build_patient_profile_index(states, [])


def test_build_index_empty_raises():
    with pytest.raises(ValueError, match="zero patients"):
        build_patient_profile_index([], [])


def test_find_similar_patient_profiles_ranks_matching_condition_highest():
    train_states = [
        _clinical_state(1, "Cardiac arrhythmias", 65),
        _clinical_state(2, "Cardiac arrhythmias", 63),
        _clinical_state(3, "Diabetes, uncomplicated", 40),
        _clinical_state(4, "Diabetes, uncomplicated", 42),
    ]
    train_labels = [
        ["antiarrhythmic"],
        ["antiarrhythmic", "anticoagulant"],
        ["antidiabetic - biguanide"],
        ["antidiabetic - biguanide", "statin"],
    ]
    index = build_patient_profile_index(train_states, train_labels)

    query = _clinical_state(999, "Cardiac arrhythmias", 64)
    results = find_similar_patient_profiles(query, index, top_k=2)

    assert len(results) == 2
    assert {r.subject_id for r in results} == {1, 2}
    # descending order by similarity
    assert results[0].similarity >= results[1].similarity


def test_find_similar_patient_profiles_carries_ground_truth_medications():
    train_states = [_clinical_state(1, "Cardiac arrhythmias", 65)]
    train_labels = [["antiarrhythmic", "anticoagulant"]]
    index = build_patient_profile_index(train_states, train_labels)

    query = _clinical_state(999, "Cardiac arrhythmias", 65)
    results = find_similar_patient_profiles(query, index, top_k=1)

    assert results[0].ground_truth_medication_classes == ["antiarrhythmic", "anticoagulant"]
    assert results[0].hadm_id == 1001
    assert results[0].evidence_id == "train-profile-1-1001"


def test_find_similar_patient_profiles_respects_top_k():
    train_states = [_clinical_state(i, "Cardiac arrhythmias", 60 + i) for i in range(1, 6)]
    train_labels = [["antiarrhythmic"]] * 5
    index = build_patient_profile_index(train_states, train_labels)

    query = _clinical_state(999, "Cardiac arrhythmias", 62)
    results = find_similar_patient_profiles(query, index, top_k=3)
    assert len(results) == 3


# --- timeline query helpers --------------------------------------------------

@pytest.fixture
def timeline():
    admittime = pd.Timestamp("2150-01-01 08:00:00")
    dischtime = pd.Timestamp("2150-01-05 12:00:00")
    admissions = pd.DataFrame(
        {
            "ROW_ID": [1], "SUBJECT_ID": [1], "HADM_ID": [100],
            "ADMITTIME": [admittime], "DISCHTIME": [dischtime],
            "ADMISSION_TYPE": ["EMERGENCY"], "ADMISSION_LOCATION": ["ER"],
            "DISCHARGE_LOCATION": ["HOME"], "DIAGNOSIS": ["AFIB"],
        }
    )
    diagnoses = pd.DataFrame(
        {"ROW_ID": [10], "SUBJECT_ID": [1], "HADM_ID": [100], "SEQ_NUM": [1], "ICD9_CODE": ["42731"]}
    )
    d_icd = pd.DataFrame({"ICD9_CODE": ["42731"], "SHORT_TITLE": ["Atrial fibrillation"]})
    prescriptions = pd.DataFrame(
        {
            "ROW_ID": [20], "SUBJECT_ID": [1], "HADM_ID": [100],
            "DRUG": ["Metoprolol"], "ROUTE": ["PO"],
            "STARTDATE": [admittime], "ENDDATE": [dischtime],
        }
    )
    labs = pd.DataFrame(
        {
            "ROW_ID": [30, 31], "SUBJECT_ID": [1, 1], "HADM_ID": [100, 100],
            "ITEMID": [50912, 50912],
            "CHARTTIME": [admittime + pd.Timedelta(days=1), admittime + pd.Timedelta(days=2)],
            "VALUE": ["1.0", "1.2"], "VALUENUM": [1.0, 1.2], "VALUEUOM": ["mg/dL", "mg/dL"], "FLAG": [None, None],
        }
    )
    d_labitems = pd.DataFrame({"ITEMID": [50912], "LABEL": ["Creatinine"]})
    notes = pd.DataFrame(
        {
            "ROW_ID": [40], "SUBJECT_ID": [1], "HADM_ID": [100],
            "CHARTDATE": [dischtime], "CHARTTIME": [dischtime],
            "CATEGORY": ["Nursing"], "DESCRIPTION": ["Note"],
            "TEXT": ["Patient reports palpitations and dizziness overnight."],
        }
    )
    return build_patient_timeline(
        subject_id=1, hadm_id=100,
        admissions=admissions, diagnoses_icd=diagnoses, prescriptions=prescriptions,
        noteevents=notes, labevents=labs, d_labitems=d_labitems, d_icd_diagnoses=d_icd,
    )


def test_get_evidence_by_ids_returns_matching_items(timeline):
    all_ids = [e.evidence.evidence_id for e in timeline.events]
    result = get_evidence_by_ids(timeline, all_ids[:2])
    assert len(result) == 2
    assert {e.evidence_id for e in result} == set(all_ids[:2])


def test_get_evidence_by_ids_ignores_unknown_ids(timeline):
    result = get_evidence_by_ids(timeline, ["not-a-real-id"])
    assert result == []


def test_search_notes_case_insensitive_keyword_match(timeline):
    result = search_notes(timeline, "PALPITATIONS")
    assert len(result) == 1
    assert result[0].source_table.value == "NOTEEVENTS"


def test_search_notes_no_match_returns_empty(timeline):
    assert search_notes(timeline, "nonexistent_keyword_xyz") == []


def test_get_recent_labs_sorted_most_recent_first(timeline):
    result = get_recent_labs(timeline)
    assert len(result) == 2
    assert result[0].timestamp > result[1].timestamp


def test_get_recent_labs_respects_limit(timeline):
    result = get_recent_labs(timeline, limit=1)
    assert len(result) == 1


def test_get_medications_returns_prescription_evidence(timeline):
    result = get_medications(timeline)
    assert len(result) == 1
    assert result[0].source_table.value == "PRESCRIPTIONS"


def test_get_diagnoses_returns_diagnosis_evidence(timeline):
    result = get_diagnoses(timeline)
    assert len(result) == 1
    assert result[0].source_table.value == "DIAGNOSES_ICD"
