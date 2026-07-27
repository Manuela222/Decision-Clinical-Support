"""Phase 2: data loading tests, using tiny fake CSVs (2-3 rows each) written
to a tmp_path so no real MIMIC-III data is required to run the test suite."""
import pandas as pd
import pytest

from cds.data import MimicDataError, load_mimic_tables, load_table

FAKE_TABLES = {
    "PATIENTS": pd.DataFrame(
        {
            "ROW_ID": [1, 2],
            "SUBJECT_ID": [10, 11],
            "GENDER": ["F", "M"],
            "DOB": ["2080-01-01", "2075-06-15"],
            "DOD": ["", ""],
            "DOD_HOSP": ["", ""],
            "DOD_SSN": ["", ""],
            "EXPIRE_FLAG": [0, 0],
        }
    ),
    "ADMISSIONS": pd.DataFrame(
        {
            "ROW_ID": [1, 2],
            "SUBJECT_ID": [10, 11],
            "HADM_ID": [100, 101],
            "ADMITTIME": ["2150-01-01 08:00:00", "2150-02-01 09:00:00"],
            "DISCHTIME": ["2150-01-05 12:00:00", "2150-02-05 12:00:00"],
            "DEATHTIME": ["", ""],
            "ADMISSION_TYPE": ["EMERGENCY", "ELECTIVE"],
            "ADMISSION_LOCATION": ["ER", "CLINIC"],
            "DISCHARGE_LOCATION": ["HOME", "HOME"],
            "INSURANCE": ["Medicare", "Private"],
            "LANGUAGE": ["ENGL", "ENGL"],
            "RELIGION": ["", ""],
            "MARITAL_STATUS": ["MARRIED", "SINGLE"],
            "ETHNICITY": ["WHITE", "WHITE"],
            "EDREGTIME": ["", ""],
            "EDOUTTIME": ["", ""],
            "DIAGNOSIS": ["HYPERTENSION", "ARRHYTHMIA"],
            "HOSPITAL_EXPIRE_FLAG": [0, 0],
            "HAS_CHARTEVENTS_DATA": [1, 1],
        }
    ),
    "DIAGNOSES_ICD": pd.DataFrame(
        {
            "ROW_ID": [1, 2],
            "SUBJECT_ID": [10, 11],
            "HADM_ID": [100, 101],
            "SEQ_NUM": [1, 1],
            "ICD9_CODE": ["4019", "4271"],
        }
    ),
    "PRESCRIPTIONS": pd.DataFrame(
        {
            "ROW_ID": [1, 2],
            "SUBJECT_ID": [10, 11],
            "HADM_ID": [100, 101],
            "ICUSTAY_ID": ["", ""],
            "STARTDATE": ["2150-01-02", "2150-02-02"],
            "ENDDATE": ["2150-01-05", "2150-02-05"],
            "DRUG_TYPE": ["MAIN", "MAIN"],
            "DRUG": ["Lisinopril", "Metoprolol"],
            "DRUG_NAME_POE": ["", ""],
            "DRUG_NAME_GENERIC": ["", ""],
            "FORMULARY_DRUG_CD": ["", ""],
            "GSN": ["", ""],
            "NDC": ["", ""],
            "PROD_STRENGTH": ["10mg", "25mg"],
            "DOSE_VAL_RX": ["10", "25"],
            "DOSE_UNIT_RX": ["mg", "mg"],
            "FORM_VAL_DISP": ["1", "1"],
            "FORM_UNIT_DISP": ["TAB", "TAB"],
            "ROUTE": ["PO", "PO"],
        }
    ),
    "NOTEEVENTS": pd.DataFrame(
        {
            "ROW_ID": [1, 2],
            "SUBJECT_ID": [10, 11],
            "HADM_ID": [100, 101],
            "CHARTDATE": ["2150-01-05", "2150-02-05"],
            "CHARTTIME": ["2150-01-05 10:00:00", "2150-02-05 10:00:00"],
            "STORETIME": ["2150-01-05 11:00:00", "2150-02-05 11:00:00"],
            "CATEGORY": ["Discharge summary", "Discharge summary"],
            "DESCRIPTION": ["Report", "Report"],
            "CGID": ["", ""],
            "ISERROR": ["", ""],
            "TEXT": ["Patient with history of hypertension...", "Patient with arrhythmia..."],
        }
    ),
    "LABEVENTS": pd.DataFrame(
        {
            "ROW_ID": [1, 2],
            "SUBJECT_ID": [10, 11],
            "HADM_ID": [100, 101],
            "ITEMID": [50912, 50912],
            "CHARTTIME": ["2150-01-03 06:00:00", "2150-02-03 06:00:00"],
            "VALUE": ["1.1", "0.9"],
            "VALUENUM": [1.1, 0.9],
            "VALUEUOM": ["mg/dL", "mg/dL"],
            "FLAG": ["", ""],
        }
    ),
    "D_LABITEMS": pd.DataFrame(
        {
            "ROW_ID": [1],
            "ITEMID": [50912],
            "LABEL": ["Creatinine"],
            "FLUID": ["Blood"],
            "CATEGORY": ["Chemistry"],
            "LOINC_CODE": ["2160-0"],
        }
    ),
    "D_ICD_DIAGNOSES": pd.DataFrame(
        {
            "ROW_ID": [1, 2],
            "ICD9_CODE": ["4019", "4271"],
            "SHORT_TITLE": ["Hypertension NOS", "Atrial fibrillation"],
            "LONG_TITLE": ["Unspecified essential hypertension", "Atrial fibrillation"],
        }
    ),
}


@pytest.fixture
def fake_mimic_dir(tmp_path):
    for table_name, df in FAKE_TABLES.items():
        df.to_csv(tmp_path / f"{table_name}.csv", index=False)
    return tmp_path


def test_load_mimic_tables_happy_path(fake_mimic_dir):
    tables = load_mimic_tables(fake_mimic_dir)

    assert len(tables.patients) == 2
    assert pd.api.types.is_datetime64_any_dtype(tables.patients["DOB"])

    assert len(tables.admissions) == 2
    assert pd.api.types.is_datetime64_any_dtype(tables.admissions["ADMITTIME"])

    assert len(tables.diagnoses_icd) == 2
    assert len(tables.prescriptions) == 2
    assert len(tables.noteevents) == 2
    assert len(tables.labevents) == 2
    assert len(tables.d_labitems) == 1
    assert len(tables.d_icd_diagnoses) == 2


def test_load_table_lowercase_filename_also_found(tmp_path):
    FAKE_TABLES["D_LABITEMS"].to_csv(tmp_path / "d_labitems.csv", index=False)
    df = load_table(tmp_path, "D_LABITEMS")
    assert len(df) == 1


def test_missing_file_raises_clear_error(fake_mimic_dir):
    (fake_mimic_dir / "PATIENTS.csv").unlink()
    with pytest.raises(MimicDataError, match="Missing required MIMIC-III file for table 'PATIENTS'"):
        load_mimic_tables(fake_mimic_dir)


def test_missing_directory_raises_clear_error(tmp_path):
    missing_dir = tmp_path / "does_not_exist"
    with pytest.raises(MimicDataError, match="data directory does not exist"):
        load_mimic_tables(missing_dir)


def test_missing_column_raises_clear_error(tmp_path):
    bad_admissions = FAKE_TABLES["ADMISSIONS"].drop(columns=["DISCHTIME"])
    bad_admissions.to_csv(tmp_path / "ADMISSIONS.csv", index=False)
    for name, df in FAKE_TABLES.items():
        if name != "ADMISSIONS":
            df.to_csv(tmp_path / f"{name}.csv", index=False)

    with pytest.raises(MimicDataError, match="missing required columns.*DISCHTIME"):
        load_mimic_tables(tmp_path)


def test_unknown_table_name_raises_clear_error(fake_mimic_dir):
    with pytest.raises(MimicDataError, match="Unknown table 'NOT_A_TABLE'"):
        load_table(fake_mimic_dir, "NOT_A_TABLE")
