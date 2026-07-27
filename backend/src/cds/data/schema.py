"""Expected column sets and date columns for the eight MIMIC-III tables this
project uses. Column lists match the official MIMIC-III schema."""
from typing import NamedTuple


class TableSpec(NamedTuple):
    required_columns: list[str]
    date_columns: list[str]


TABLE_SPECS: dict[str, TableSpec] = {
    "PATIENTS": TableSpec(
        required_columns=["ROW_ID", "SUBJECT_ID", "GENDER", "DOB", "DOD", "DOD_HOSP", "DOD_SSN", "EXPIRE_FLAG"],
        date_columns=["DOB", "DOD", "DOD_HOSP", "DOD_SSN"],
    ),
    "ADMISSIONS": TableSpec(
        required_columns=[
            "ROW_ID", "SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME", "DEATHTIME",
            "ADMISSION_TYPE", "ADMISSION_LOCATION", "DISCHARGE_LOCATION", "INSURANCE",
            "LANGUAGE", "RELIGION", "MARITAL_STATUS", "ETHNICITY", "EDREGTIME", "EDOUTTIME",
            "DIAGNOSIS", "HOSPITAL_EXPIRE_FLAG", "HAS_CHARTEVENTS_DATA",
        ],
        date_columns=["ADMITTIME", "DISCHTIME", "DEATHTIME", "EDREGTIME", "EDOUTTIME"],
    ),
    "DIAGNOSES_ICD": TableSpec(
        required_columns=["ROW_ID", "SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"],
        date_columns=[],
    ),
    "PRESCRIPTIONS": TableSpec(
        required_columns=[
            "ROW_ID", "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "STARTDATE", "ENDDATE",
            "DRUG_TYPE", "DRUG", "DRUG_NAME_POE", "DRUG_NAME_GENERIC", "FORMULARY_DRUG_CD",
            "GSN", "NDC", "PROD_STRENGTH", "DOSE_VAL_RX", "DOSE_UNIT_RX", "FORM_VAL_DISP",
            "FORM_UNIT_DISP", "ROUTE",
        ],
        date_columns=["STARTDATE", "ENDDATE"],
    ),
    "NOTEEVENTS": TableSpec(
        required_columns=[
            "ROW_ID", "SUBJECT_ID", "HADM_ID", "CHARTDATE", "CHARTTIME", "STORETIME",
            "CATEGORY", "DESCRIPTION", "CGID", "ISERROR", "TEXT",
        ],
        date_columns=["CHARTDATE", "CHARTTIME", "STORETIME"],
    ),
    "LABEVENTS": TableSpec(
        required_columns=["ROW_ID", "SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "VALUE", "VALUENUM", "VALUEUOM", "FLAG"],
        date_columns=["CHARTTIME"],
    ),
    "D_LABITEMS": TableSpec(
        required_columns=["ROW_ID", "ITEMID", "LABEL", "FLUID", "CATEGORY", "LOINC_CODE"],
        date_columns=[],
    ),
    "D_ICD_DIAGNOSES": TableSpec(
        required_columns=["ROW_ID", "ICD9_CODE", "SHORT_TITLE", "LONG_TITLE"],
        date_columns=[],
    ),
}
