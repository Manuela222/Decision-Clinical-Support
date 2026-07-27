"""Phase 5: build_patient_timeline / build_clinical_state tests, against
small fully-fake DataFrames shaped like the Phase 2 loader's output."""
import pandas as pd
import pytest

from cds.schemas import HypertensionStatus, MissingInformationFlag, TimelineEventType
from cds.timeline import build_clinical_state, build_patient_timeline

ADMITTIME = pd.Timestamp("2150-01-01 08:00:00")
DISCHTIME = pd.Timestamp("2150-01-05 12:00:00")


def _admissions_df():
    return pd.DataFrame(
        {
            "ROW_ID": [1],
            "SUBJECT_ID": [1],
            "HADM_ID": [100],
            "ADMITTIME": [ADMITTIME],
            "DISCHTIME": [DISCHTIME],
            "ADMISSION_TYPE": ["EMERGENCY"],
            "ADMISSION_LOCATION": ["ER"],
            "DISCHARGE_LOCATION": ["HOME"],
            "DIAGNOSIS": ["ATRIAL FIBRILLATION"],
        }
    )


def _diagnoses_df():
    return pd.DataFrame(
        {
            "ROW_ID": [10, 11],
            "SUBJECT_ID": [1, 1],
            "HADM_ID": [100, 100],
            "SEQ_NUM": [1, 2],
            "ICD9_CODE": ["4019", "42731"],  # Hypertension, Cardiac arrhythmias
        }
    )


def _d_icd_diagnoses_df():
    return pd.DataFrame(
        {
            "ICD9_CODE": ["4019", "42731"],
            "SHORT_TITLE": ["Hypertension NOS", "Atrial fibrillation"],
        }
    )


def _prescriptions_df():
    return pd.DataFrame(
        {
            "ROW_ID": [20, 21, 22],
            "SUBJECT_ID": [1, 1, 1],
            "HADM_ID": [100, 100, 100],
            "DRUG": ["Metoprolol", "Potassium Chloride", "Warfarin"],
            "ROUTE": ["PO", "IV", "PO"],
            # Metoprolol + Potassium Chloride start at admission (within window);
            # Warfarin starts near discharge (outside the 48h window) -- this is
            # the leakage-guard case: it must NOT show up in current_medications.
            "STARTDATE": [ADMITTIME, ADMITTIME, DISCHTIME - pd.Timedelta(hours=1)],
            "ENDDATE": [DISCHTIME, ADMITTIME + pd.Timedelta(hours=6), DISCHTIME],
        }
    )


def _labevents_df():
    return pd.DataFrame(
        {
            "ROW_ID": [30],
            "SUBJECT_ID": [1],
            "HADM_ID": [100],
            "ITEMID": [50912],
            "CHARTTIME": [ADMITTIME + pd.Timedelta(days=1)],
            "VALUE": ["1.1"],
            "VALUENUM": [1.1],
            "VALUEUOM": ["mg/dL"],
            "FLAG": [None],
        }
    )


def _d_labitems_df():
    return pd.DataFrame({"ITEMID": [50912], "LABEL": ["Creatinine"]})


def _noteevents_df():
    text = (
        "History of Present Illness: patient with hx of HTN presents with palpitations "
        "and is found to be in atrial fibrillation.\n\n"
        "Discharge Medications:\n1. Warfarin 5mg daily\n2. Metoprolol 25mg BID\n"
    )
    return pd.DataFrame(
        {
            "ROW_ID": [40],
            "SUBJECT_ID": [1],
            "HADM_ID": [100],
            "CHARTDATE": [DISCHTIME],
            "CHARTTIME": [DISCHTIME],
            "CATEGORY": ["Discharge summary"],
            "DESCRIPTION": ["Report"],
            "TEXT": [text],
        }
    )


@pytest.fixture
def timeline():
    return build_patient_timeline(
        subject_id=1,
        hadm_id=100,
        admissions=_admissions_df(),
        diagnoses_icd=_diagnoses_df(),
        prescriptions=_prescriptions_df(),
        noteevents=_noteevents_df(),
        labevents=_labevents_df(),
        d_labitems=_d_labitems_df(),
        d_icd_diagnoses=_d_icd_diagnoses_df(),
    )


# --- build_patient_timeline ---------------------------------------------

def test_timeline_is_chronologically_sorted(timeline):
    timestamps = [e.timestamp for e in timeline.events]
    non_none = [t for t in timestamps if t is not None]
    assert non_none == sorted(non_none)


def test_timeline_events_have_evidence_citations(timeline):
    for event in timeline.events:
        assert event.evidence.evidence_id
        assert event.evidence.subject_id == 1


def test_timeline_includes_all_event_types(timeline):
    types = {e.event_type for e in timeline.events}
    assert types == {
        TimelineEventType.ADMISSION,
        TimelineEventType.DISCHARGE,
        TimelineEventType.DIAGNOSIS,
        TimelineEventType.MEDICATION_ORDER,
        TimelineEventType.LAB_RESULT,
        TimelineEventType.NOTE,
    }


def test_timeline_raises_for_unknown_admission():
    with pytest.raises(ValueError, match="No ADMISSIONS row found"):
        build_patient_timeline(
            subject_id=999,
            hadm_id=999,
            admissions=_admissions_df(),
            diagnoses_icd=_diagnoses_df(),
            prescriptions=_prescriptions_df(),
            noteevents=_noteevents_df(),
            labevents=_labevents_df(),
            d_labitems=_d_labitems_df(),
            d_icd_diagnoses=_d_icd_diagnoses_df(),
        )


# --- build_clinical_state ------------------------------------------------

def test_clinical_state_leakage_guard_excludes_late_medication(timeline):
    state = build_clinical_state(
        timeline, age=67, gender="F",
        condition_groups=["Hypertension", "Cardiac arrhythmias"],
        admission_reason="Cardiac arrhythmias",
    )
    normalized_names = {m.normalized_name for m in state.current_medications}
    assert "metoprolol" in normalized_names
    assert "warfarin" not in normalized_names  # started too close to discharge


def test_clinical_state_excludes_iv_fluids(timeline):
    state = build_clinical_state(
        timeline, age=67, gender="F",
        condition_groups=["Hypertension", "Cardiac arrhythmias"],
        admission_reason="Cardiac arrhythmias",
    )
    normalized_names = {m.normalized_name for m in state.current_medications}
    assert "potassium chloride" not in normalized_names


def test_clinical_state_hypertension_confirmed_from_condition_group(timeline):
    state = build_clinical_state(
        timeline, age=67, gender="F",
        condition_groups=["Hypertension", "Cardiac arrhythmias"],
        admission_reason="Cardiac arrhythmias",
    )
    assert state.hypertension_status == HypertensionStatus.CONFIRMED_CHRONIC
    assert len(state.hypertension_evidence) == 1


def test_clinical_state_hypertension_suspected_from_medication_only(timeline):
    state = build_clinical_state(
        timeline, age=67, gender="F",
        condition_groups=["Cardiac arrhythmias"],  # no Hypertension group this time
        admission_reason="Cardiac arrhythmias",
    )
    # Metoprolol (beta blocker, antihypertensive-tagged) is in the window -> SUSPECTED
    assert state.hypertension_status == HypertensionStatus.SUSPECTED


def test_clinical_state_hypertension_not_present(timeline):
    # Remove the antihypertensive-tagged medication from the picture entirely
    # by dropping it from the fixture's timeline instead of the class dict.
    timeline.events = [e for e in timeline.events if "metoprolol" not in e.description.lower()]
    state = build_clinical_state(
        timeline, age=67, gender="F",
        condition_groups=["Cardiac arrhythmias"],
        admission_reason="Cardiac arrhythmias",
    )
    assert state.hypertension_status == HypertensionStatus.NOT_PRESENT


def test_clinical_state_note_excerpt_truncated_before_discharge_meds(timeline):
    state = build_clinical_state(
        timeline, age=67, gender="F",
        condition_groups=["Hypertension", "Cardiac arrhythmias"],
        admission_reason="Cardiac arrhythmias",
    )
    assert len(state.relevant_note_sections) == 1
    excerpt = state.relevant_note_sections[0].excerpt.lower()
    assert "history of present illness" in excerpt
    assert "warfarin" not in excerpt
    assert "discharge medications" not in excerpt


def test_clinical_state_missing_information_flags(timeline):
    state = build_clinical_state(
        timeline, age=67, gender="F",
        condition_groups=["Hypertension", "Cardiac arrhythmias"],
        admission_reason="Cardiac arrhythmias",
    )
    assert MissingInformationFlag.PRIOR_ADMISSIONS_MISSING in state.missing_information
    assert MissingInformationFlag.HYPERTENSION_LABS_MISSING in state.missing_information
    assert MissingInformationFlag.DISCHARGE_SUMMARY_MISSING not in state.missing_information
    assert MissingInformationFlag.LABS_MISSING not in state.missing_information
    assert MissingInformationFlag.NOTES_MISSING not in state.missing_information


def test_clinical_state_recent_labs_present(timeline):
    state = build_clinical_state(
        timeline, age=67, gender="F",
        condition_groups=["Hypertension", "Cardiac arrhythmias"],
        admission_reason="Cardiac arrhythmias",
    )
    assert len(state.recent_labs) == 1
    assert state.recent_labs[0].label == "Creatinine"
    assert state.recent_labs[0].value == 1.1


def test_clinical_state_prior_medications_from_prior_timeline(timeline):
    prior_admittime = ADMITTIME - pd.Timedelta(days=90)
    prior_dischtime = prior_admittime + pd.Timedelta(days=3)
    prior_admissions = pd.DataFrame(
        {
            "ROW_ID": [2], "SUBJECT_ID": [1], "HADM_ID": [99],
            "ADMITTIME": [prior_admittime], "DISCHTIME": [prior_dischtime],
            "ADMISSION_TYPE": ["ELECTIVE"], "ADMISSION_LOCATION": ["CLINIC"],
            "DISCHARGE_LOCATION": ["HOME"], "DIAGNOSIS": ["HYPERTENSION"],
        }
    )
    prior_prescriptions = pd.DataFrame(
        {
            "ROW_ID": [50], "SUBJECT_ID": [1], "HADM_ID": [99],
            "DRUG": ["Lisinopril"], "ROUTE": ["PO"],
            "STARTDATE": [prior_admittime], "ENDDATE": [prior_dischtime],
        }
    )
    empty_diag = _diagnoses_df().iloc[0:0]
    empty_labs = _labevents_df().iloc[0:0]
    empty_notes = _noteevents_df().iloc[0:0]

    prior_timeline = build_patient_timeline(
        subject_id=1, hadm_id=99,
        admissions=prior_admissions, diagnoses_icd=empty_diag,
        prescriptions=prior_prescriptions, noteevents=empty_notes,
        labevents=empty_labs, d_labitems=_d_labitems_df(), d_icd_diagnoses=_d_icd_diagnoses_df(),
    )

    state = build_clinical_state(
        timeline, age=67, gender="F",
        condition_groups=["Hypertension", "Cardiac arrhythmias"],
        admission_reason="Cardiac arrhythmias",
        prior_admission_timeline=prior_timeline,
    )
    assert MissingInformationFlag.PRIOR_ADMISSIONS_MISSING not in state.missing_information
    assert {m.normalized_name for m in state.prior_medications} == {"lisinopril"}
