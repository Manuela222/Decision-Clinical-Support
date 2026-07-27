"""Phase 0, audit task 2: patient-level timeline completeness."""
import json
from pathlib import Path

import duckdb
import pandas as pd

BASE = Path(__file__).parent
DATA = BASE.parent / "MIMIC_III_10k" / "work" / "parquet"
RESULTS = BASE / "results"

con = duckdb.connect()


def tbl(name: str) -> str:
    return f"read_parquet('{(DATA / (name + '.parquet')).as_posix()}')"


cohort = pd.read_parquet(RESULTS / "cohort_final.parquet")
hadm_ids = cohort["HADM_ID"].tolist()

results = {}

# discharge summary present
notes = con.execute(f"select hadm_id, CATEGORY from {tbl('noteevents')}").df()
notes_cohort = notes[notes["hadm_id"].isin(hadm_ids)]
has_discharge_summary = set(notes_cohort[notes_cohort["CATEGORY"].str.strip() == "Discharge summary"]["hadm_id"])
has_any_note = set(notes_cohort["hadm_id"])

# labs present
labs = con.execute(f"select distinct HADM_ID from {tbl('labevents')}").df()
has_labs = set(labs[labs["HADM_ID"].isin(hadm_ids)]["HADM_ID"])

# prescriptions present (already guaranteed by cohort filter, sanity check)
presc = con.execute(f"select distinct HADM_ID from {tbl('prescriptions')}").df()
has_presc = set(presc[presc["HADM_ID"].isin(hadm_ids)]["HADM_ID"])

# prior admissions: does this patient have an earlier admission (by ADMITTIME) than the current one
all_adm = con.execute(f"select SUBJECT_ID, HADM_ID, ADMITTIME from {tbl('admissions')}").df()
all_adm["ADMITTIME"] = pd.to_datetime(all_adm["ADMITTIME"])
cohort_adm_times = cohort.set_index("HADM_ID")["ADMITTIME"]
has_prior_admission = set()
adm_by_subject = all_adm.groupby("SUBJECT_ID")
for _, row in cohort.iterrows():
    subj_admits = all_adm[all_adm["SUBJECT_ID"] == row["SUBJECT_ID"]]
    if (subj_admits["ADMITTIME"] < pd.to_datetime(row["ADMITTIME"])).any():
        has_prior_admission.add(row["HADM_ID"])

n = len(hadm_ids)
completeness = {
    "n_cohort_admissions": n,
    "has_discharge_summary": len(has_discharge_summary),
    "missing_discharge_summary": n - len(has_discharge_summary),
    "has_any_note": len(has_any_note),
    "missing_any_note": n - len(has_any_note),
    "has_labs": len(has_labs),
    "missing_labs": n - len(has_labs),
    "has_prescriptions": len(has_presc),
    "missing_prescriptions": n - len(has_presc),
    "has_prior_admission": len(has_prior_admission),
    "no_prior_admission_first_visit": n - len(has_prior_admission),
}

# fully complete = discharge summary + labs + prescriptions (prior admission is optional/contextual, not required)
fully_complete = has_discharge_summary & has_labs & has_presc
completeness["fully_complete_core_fields"] = len(fully_complete)
completeness["partial_missing_something_core"] = n - len(fully_complete)

with open(RESULTS / "step2_timeline_completeness.json", "w") as f:
    json.dump(completeness, f, indent=2)

print(json.dumps(completeness, indent=2))
