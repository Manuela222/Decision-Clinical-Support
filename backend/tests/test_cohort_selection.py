"""Phase 3: select_cohort() tests against small, fully synthetic DataFrames.

Subject IDs used, and why each should (or should not) end up in the cohort:
  1  -> qualifies: adult, HTN + Cardiac arrhythmias (exactly 2 groups), has a prescription
  2  -> excluded: age 10 at admission (pediatric)
  3  -> excluded: HTN + Cardiac arrhythmias + CHF (3 groups, not exactly 2)
  4  -> excluded: Cardiac arrhythmias + Diabetes (2 groups, neither is Hypertension)
  5  -> excluded: HTN + Cardiac arrhythmias (2 groups) but no prescriptions at all
  6  -> qualifies: deidentified-elderly (DOB shifted per MIMIC-III >89 convention),
       HTN + Diabetes uncomplicated, has a prescription
"""
import pandas as pd
import pytest

from cds.cohort import select_cohort

HTN = "4019"  # Hypertension
ARRHYTHMIA = "42731"  # Cardiac arrhythmias
CHF = "4280"  # Congestive heart failure
DIABETES = "25000"  # Diabetes, uncomplicated


@pytest.fixture
def patients() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SUBJECT_ID": [1, 2, 3, 4, 5, 6],
            "GENDER": ["F", "M", "F", "M", "F", "M"],
            "DOB": [
                "2100-01-01",  # subject 1: age 50 at admission
                "2140-01-01",  # subject 2: age 10 at admission
                "2100-01-01",  # subject 3: age 50
                "2100-01-01",  # subject 4: age 50
                "2100-01-01",  # subject 5: age 50
                "1850-01-01",  # subject 6: deidentified-elderly (raw age ~300)
            ],
        }
    )


@pytest.fixture
def admissions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SUBJECT_ID": [1, 2, 3, 4, 5, 6],
            "HADM_ID": [101, 102, 103, 104, 105, 106],
            "ADMITTIME": ["2150-01-01"] * 6,
            "DISCHTIME": ["2150-01-10"] * 6,
        }
    )


@pytest.fixture
def diagnoses_icd() -> pd.DataFrame:
    rows = [
        (1, 101, HTN),
        (1, 101, ARRHYTHMIA),
        (2, 102, HTN),
        (2, 102, ARRHYTHMIA),
        (3, 103, HTN),
        (3, 103, ARRHYTHMIA),
        (3, 103, CHF),
        (4, 104, ARRHYTHMIA),
        (4, 104, DIABETES),
        (5, 105, HTN),
        (5, 105, ARRHYTHMIA),
        (6, 106, HTN),
        (6, 106, DIABETES),
    ]
    return pd.DataFrame(rows, columns=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"])


@pytest.fixture
def prescriptions() -> pd.DataFrame:
    # subject 5 / hadm 105 deliberately has no prescriptions at all.
    return pd.DataFrame(
        {
            "SUBJECT_ID": [1, 2, 3, 4, 6],
            "HADM_ID": [101, 102, 103, 104, 106],
            "DRUG": ["Metoprolol", "Metoprolol", "Metoprolol", "Metformin", "Lisinopril"],
        }
    )


def test_select_cohort_default_filters(patients, admissions, diagnoses_icd, prescriptions):
    result = select_cohort(patients, admissions, diagnoses_icd, prescriptions)

    assert sorted(result["subject_id"].tolist()) == [1, 6]
    assert set(result["hadm_id"]) == {101, 106}


def test_select_cohort_deidentified_elderly_treated_as_adult(patients, admissions, diagnoses_icd, prescriptions):
    result = select_cohort(patients, admissions, diagnoses_icd, prescriptions)
    row6 = result[result["subject_id"] == 6].iloc[0]
    assert row6["age"] >= 18


def test_select_cohort_excludes_pediatric(patients, admissions, diagnoses_icd, prescriptions):
    result = select_cohort(patients, admissions, diagnoses_icd, prescriptions)
    assert 2 not in result["subject_id"].tolist()


def test_select_cohort_excludes_three_condition_groups(patients, admissions, diagnoses_icd, prescriptions):
    result = select_cohort(patients, admissions, diagnoses_icd, prescriptions)
    assert 3 not in result["subject_id"].tolist()


def test_select_cohort_excludes_missing_required_group(patients, admissions, diagnoses_icd, prescriptions):
    result = select_cohort(patients, admissions, diagnoses_icd, prescriptions)
    assert 4 not in result["subject_id"].tolist()


def test_select_cohort_excludes_no_prescriptions(patients, admissions, diagnoses_icd, prescriptions):
    result = select_cohort(patients, admissions, diagnoses_icd, prescriptions)
    assert 5 not in result["subject_id"].tolist()


def test_select_cohort_condition_groups_and_other_groups(patients, admissions, diagnoses_icd, prescriptions):
    result = select_cohort(patients, admissions, diagnoses_icd, prescriptions)
    row1 = result[result["subject_id"] == 1].iloc[0]
    assert row1["condition_groups"] == ["Cardiac arrhythmias", "Hypertension"]
    assert row1["other_condition_groups"] == ["Cardiac arrhythmias"]

    row6 = result[result["subject_id"] == 6].iloc[0]
    assert row6["other_condition_groups"] == ["Diabetes, uncomplicated"]


def test_select_cohort_is_deterministic(patients, admissions, diagnoses_icd, prescriptions):
    result_a = select_cohort(patients, admissions, diagnoses_icd, prescriptions)
    result_b = select_cohort(patients, admissions, diagnoses_icd, prescriptions)
    pd.testing.assert_frame_equal(result_a, result_b)


def test_select_cohort_returns_expected_columns(patients, admissions, diagnoses_icd, prescriptions):
    result = select_cohort(patients, admissions, diagnoses_icd, prescriptions)
    assert list(result.columns) == [
        "subject_id",
        "hadm_id",
        "age",
        "gender",
        "admittime",
        "dischtime",
        "condition_groups",
        "other_condition_groups",
    ]


def test_select_cohort_custom_min_age_includes_pediatric(patients, admissions, diagnoses_icd, prescriptions):
    result = select_cohort(patients, admissions, diagnoses_icd, prescriptions, min_age=5)
    assert 2 in result["subject_id"].tolist()


def test_select_cohort_custom_n_condition_groups(patients, admissions, diagnoses_icd, prescriptions):
    # subject 3 has 3 groups incl. Hypertension -- should qualify when n_condition_groups=3
    result = select_cohort(
        patients, admissions, diagnoses_icd, prescriptions,
        required_condition_groups=["Hypertension"], n_condition_groups=3,
    )
    assert result["subject_id"].tolist() == [3]


def test_select_cohort_custom_required_groups(patients, admissions, diagnoses_icd, prescriptions):
    # require Diabetes instead of Hypertension, still exactly 2 groups -> only subject 4 (Arrhythmia + Diabetes)
    result = select_cohort(
        patients, admissions, diagnoses_icd, prescriptions,
        required_condition_groups=["Diabetes, uncomplicated"], n_condition_groups=2,
    )
    assert sorted(result["subject_id"].tolist()) == [4, 6]
