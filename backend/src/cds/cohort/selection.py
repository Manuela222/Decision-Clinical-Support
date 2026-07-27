"""Phase 3: cohort selection, productionizing the filters validated in the
Phase 0 feasibility audit as a single reusable, deterministic function."""
from typing import List

import pandas as pd

from .elixhauser import map_icd9_to_elixhauser

# MIMIC-III shifts DOB for patients aged >89 at first admission so that
# computed age comes out to roughly 300 years (a deliberate deidentification
# artifact documented by the MIMIC-III team, not a data error). Any raw
# computed age above this threshold is treated as a real adult and floored to
# a nominal reporting age, exactly as validated in the Phase 0 audit.
_DEIDENTIFICATION_AGE_THRESHOLD = 89
_DEIDENTIFIED_ELDERLY_REPORTED_AGE = 90


def select_cohort(
    patients: pd.DataFrame,
    admissions: pd.DataFrame,
    diagnoses_icd: pd.DataFrame,
    prescriptions: pd.DataFrame,
    min_age: int = 18,
    required_condition_groups: List[str] = ["Hypertension"],
    n_condition_groups: int = 2,
) -> pd.DataFrame:
    """Select admissions matching the project's cohort definition.

    Filters, applied cumulatively:
      1. Age >= min_age at admission (deidentification-corrected).
      2. The admission's diagnoses touch exactly `n_condition_groups` distinct
         Elixhauser comorbidity groups, and all of `required_condition_groups`
         are among them.
      3. The admission has at least one row in `prescriptions`.

    Returns a DataFrame with one row per qualifying admission:
    subject_id, hadm_id, age, gender, admittime, dischtime, condition_groups
    (the full set of groups touched, as a sorted list) and
    other_condition_groups (condition_groups minus required_condition_groups).
    """
    merged = admissions.merge(
        patients[["SUBJECT_ID", "DOB", "GENDER"]], on="SUBJECT_ID", how="inner"
    )

    # Cast to microsecond (not nanosecond) resolution before subtracting:
    # MIMIC-III's deidentification shift produces DOBs as early as ~1800 and
    # admissions as late as ~2200+, and datetime64[ns] subtraction overflows
    # well within that range (see loader.py for the same fix).
    admittime = pd.to_datetime(merged["ADMITTIME"]).astype("datetime64[us]")
    dob = pd.to_datetime(merged["DOB"]).astype("datetime64[us]")
    raw_age_years = (admittime - dob).dt.days / 365.2425
    is_deidentified_elderly = raw_age_years > _DEIDENTIFICATION_AGE_THRESHOLD
    age_years = raw_age_years.where(~is_deidentified_elderly, _DEIDENTIFIED_ELDERLY_REPORTED_AGE)

    merged = merged.assign(age=age_years, ADMITTIME=admittime)
    age_filtered = merged[merged["age"] >= min_age].copy()

    diag = diagnoses_icd[diagnoses_icd["HADM_ID"].isin(age_filtered["HADM_ID"])].copy()
    diag["condition_group"] = diag["ICD9_CODE"].apply(map_icd9_to_elixhauser)
    exploded = diag.explode("condition_group").dropna(subset=["condition_group"])

    groups_per_hadm = exploded.groupby("HADM_ID")["condition_group"].agg(lambda s: sorted(set(s)))
    required_set = set(required_condition_groups)
    matches_condition_filter = groups_per_hadm.apply(
        lambda groups: len(groups) == n_condition_groups and required_set.issubset(groups)
    )
    qualifying_hadm_ids = set(groups_per_hadm[matches_condition_filter].index)

    condition_filtered = age_filtered[age_filtered["HADM_ID"].isin(qualifying_hadm_ids)].copy()

    prescribed_hadm_ids = set(prescriptions["HADM_ID"].unique())
    final = condition_filtered[condition_filtered["HADM_ID"].isin(prescribed_hadm_ids)].copy()

    final["condition_groups"] = final["HADM_ID"].map(groups_per_hadm)
    final["other_condition_groups"] = final["condition_groups"].apply(
        lambda groups: sorted(set(groups) - required_set)
    )

    result = final.rename(
        columns={
            "SUBJECT_ID": "subject_id",
            "HADM_ID": "hadm_id",
            "GENDER": "gender",
            "ADMITTIME": "admittime",
            "DISCHTIME": "dischtime",
        }
    )[
        [
            "subject_id",
            "hadm_id",
            "age",
            "gender",
            "admittime",
            "dischtime",
            "condition_groups",
            "other_condition_groups",
        ]
    ]
    result["dischtime"] = pd.to_datetime(result["dischtime"]).astype("datetime64[us]")
    return result.reset_index(drop=True)
