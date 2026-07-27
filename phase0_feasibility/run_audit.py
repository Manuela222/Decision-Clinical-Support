"""
Phase 0 feasibility audit.

Runs entirely against the pre-built parquet extract at
MIMIC_III_10k/work/parquet/ (a ~10,000-subject subset of MIMIC-III, not the
full ~46,500-subject database). This is a load-bearing caveat for the
feasibility report: all counts below are subset counts and would scale up
(roughly 4-5x) on the full MIMIC-III database.

Outputs JSON results per audit step to phase0_feasibility/results/ so the
report generator can consume them without recomputation.
"""
import json
from pathlib import Path

import duckdb
import pandas as pd

from elixhauser import map_icd9_to_elixhauser, ALL_CATEGORIES

BASE = Path(__file__).parent
DATA = BASE.parent / "MIMIC_III_10k" / "work" / "parquet"
RESULTS = BASE / "results"
RESULTS.parent.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

con = duckdb.connect()


def tbl(name: str) -> str:
    return f"read_parquet('{(DATA / (name + '.parquet')).as_posix()}')"


def save_json(name: str, obj) -> None:
    with open(RESULTS / f"{name}.json", "w") as f:
        json.dump(obj, f, indent=2, default=str)


results = {}

# ---------------------------------------------------------------------------
# Step 0: raw table sizes
# ---------------------------------------------------------------------------
raw_counts = {}
for t, key in [
    ("patients", "n_patients"),
    ("admissions", "n_admissions"),
    ("diagnoses_icd", "n_diagnosis_rows"),
    ("prescriptions", "n_prescription_rows"),
    ("labevents", "n_labevent_rows"),
    ("noteevents", "n_noteevent_rows"),
]:
    n = con.execute(f"select count(*) from {tbl(t)}").fetchone()[0]
    raw_counts[key] = n
results["raw_table_counts"] = raw_counts
print("raw_table_counts:", raw_counts)

# ---------------------------------------------------------------------------
# Step 1: cohort filter audit (cumulative funnel)
# ---------------------------------------------------------------------------
funnel = {}

# 1a. Start: all admissions
adm = con.execute(f"""
    select a.SUBJECT_ID, a.HADM_ID, a.ADMITTIME, a.DISCHTIME,
           p.DOB, p.GENDER
    from {tbl('admissions')} a
    join {tbl('patients')} p using (SUBJECT_ID)
""").df()
funnel["0_all_admissions"] = len(adm)
funnel["0_all_patients"] = adm["SUBJECT_ID"].nunique()

# 1b. Age >= 18 at admission, correcting for MIMIC-III >89 date-shift
adm["ADMITTIME"] = pd.to_datetime(adm["ADMITTIME"])
adm["DOB"] = pd.to_datetime(adm["DOB"])
raw_age_days = (adm["ADMITTIME"] - adm["DOB"]).dt.days
adm["raw_age_years"] = raw_age_days / 365.2425
# MIMIC-III shifts DOB for patients >89 so raw computed age comes out ~300.
# Per MIMIC-III documentation these are real adults whose true age is
# deliberately obscured (deidentification); treat any raw age > 89 as a
# real patient with true age >= 89 (adult), not as a data error.
adm["is_deidentified_elderly"] = adm["raw_age_years"] > 89
adm["age_years"] = adm["raw_age_years"].where(~adm["is_deidentified_elderly"], 90)
age_filtered = adm[adm["age_years"] >= 18].copy()
funnel["1_age_ge_18_admissions"] = len(age_filtered)
funnel["1_age_ge_18_patients"] = age_filtered["SUBJECT_ID"].nunique()
funnel["1_deidentified_elderly_admissions"] = int(adm["is_deidentified_elderly"].sum())
funnel["1_excluded_under_18_admissions"] = len(adm) - len(age_filtered)

# 1c. Diagnoses -> Elixhauser groups per admission
diag = con.execute(f"""
    select SUBJECT_ID, HADM_ID, ICD9_CODE
    from {tbl('diagnoses_icd')}
    where HADM_ID in (select HADM_ID from age_filtered_view)
""") if False else None  # placeholder, using pandas join below instead

diag_all = con.execute(f"select SUBJECT_ID, HADM_ID, ICD9_CODE from {tbl('diagnoses_icd')}").df()
diag_all = diag_all[diag_all["HADM_ID"].isin(age_filtered["HADM_ID"])]

# map each code to its category set, explode
diag_all["categories"] = diag_all["ICD9_CODE"].apply(lambda c: sorted(map_icd9_to_elixhauser(c)))
exploded = diag_all.explode("categories").dropna(subset=["categories"])

groups_per_hadm = exploded.groupby("HADM_ID")["categories"].agg(lambda s: sorted(set(s)))
n_groups = groups_per_hadm.apply(len)
has_htn = groups_per_hadm.apply(lambda g: "Hypertension" in g)

two_group_htn_hadm_ids = groups_per_hadm[(n_groups == 2) & has_htn].index
funnel["2_exactly_2_groups_incl_htn_admissions"] = len(two_group_htn_hadm_ids)
cohort_step2 = age_filtered[age_filtered["HADM_ID"].isin(two_group_htn_hadm_ids)].copy()
funnel["2_exactly_2_groups_incl_htn_patients"] = cohort_step2["SUBJECT_ID"].nunique()

# also report the *other* group's distribution (i.e. the admission condition)
other_group = {}
for hadm_id, groups in groups_per_hadm[(n_groups == 2) & has_htn].items():
    other = [g for g in groups if g != "Hypertension"][0]
    other_group[hadm_id] = other
cohort_step2["admission_condition_group"] = cohort_step2["HADM_ID"].map(other_group)
admission_condition_counts = cohort_step2["admission_condition_group"].value_counts().to_dict()
results["admission_condition_group_counts_after_step2"] = admission_condition_counts

# distribution diagnostics: how many admissions have 1 group / 2 groups (any) / >=3 groups, for context
dist = n_groups.value_counts().sort_index().to_dict()
results["n_elixhauser_groups_distribution_among_age_filtered"] = {str(k): int(v) for k, v in dist.items()}
n_with_htn_anywhere = groups_per_hadm.apply(lambda g: "Hypertension" in g).sum()
results["admissions_with_htn_anywhere_among_age_filtered"] = int(n_with_htn_anywhere)

# 1d. current admission has >=1 row in PRESCRIPTIONS
presc_hadm_ids = set(con.execute(f"select distinct HADM_ID from {tbl('prescriptions')}").df()["HADM_ID"])
cohort_step3 = cohort_step2[cohort_step2["HADM_ID"].isin(presc_hadm_ids)].copy()
funnel["3_has_discharge_prescriptions_admissions"] = len(cohort_step3)
funnel["3_has_discharge_prescriptions_patients"] = cohort_step3["SUBJECT_ID"].nunique()

results["cohort_filter_funnel"] = funnel
print("funnel:", funnel)

cohort_step3.to_parquet(RESULTS / "cohort_final.parquet", index=False)
save_json("step1_cohort_filter", results)

print("Step 1 (cohort filter) complete. Final cohort admissions:", len(cohort_step3),
      "patients:", cohort_step3["SUBJECT_ID"].nunique())
