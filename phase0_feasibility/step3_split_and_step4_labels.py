"""Phase 0, audit tasks 3 (80/20 split) and 4 (medication label feasibility)."""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from med_classes_exploratory import (
    normalize_medication_name,
    map_to_medication_class,
    is_antihypertensive_class,
)

BASE = Path(__file__).parent
DATA = BASE.parent / "MIMIC_III_10k" / "work" / "parquet"
RESULTS = BASE / "results"

con = duckdb.connect()


def tbl(name: str) -> str:
    return f"read_parquet('{(DATA / (name + '.parquet')).as_posix()}')"


cohort = pd.read_parquet(RESULTS / "cohort_final.parquet")
hadm_ids = cohort["HADM_ID"].tolist()
dischtime_by_hadm = cohort.set_index("HADM_ID")["DISCHTIME"]

presc = con.execute(f"select * from read_parquet('{(DATA/'prescriptions.parquet').as_posix()}')").df()
presc = presc[presc["HADM_ID"].isin(hadm_ids)].copy()
presc["STARTDATE"] = pd.to_datetime(presc["STARTDATE"])
presc["ENDDATE"] = pd.to_datetime(presc["ENDDATE"])
presc["DISCHTIME"] = presc["HADM_ID"].map(dischtime_by_hadm)
presc["DISCHTIME"] = pd.to_datetime(presc["DISCHTIME"])

# discharge medication heuristic (see docstring in med_classes_exploratory.py)
is_discharge_med = (
    (presc["STARTDATE"].isna() | (presc["STARTDATE"] <= presc["DISCHTIME"]))
    & (presc["ENDDATE"].isna() | (presc["ENDDATE"].dt.normalize() >= presc["DISCHTIME"].dt.normalize()))
)
disch_presc = presc[is_discharge_med].copy()

disch_presc["normalized"] = disch_presc["DRUG"].apply(normalize_medication_name)
disch_presc["med_class"] = disch_presc["normalized"].apply(map_to_medication_class)
disch_presc = disch_presc[disch_presc["med_class"].notna() & (disch_presc["med_class"] != "__EXCLUDE__")]

# tag antihypertensive vs label-eligible
disch_presc["is_antihtn"] = disch_presc["med_class"].apply(is_antihypertensive_class)
label_presc = disch_presc[~disch_presc["is_antihtn"]]

# per-hadm_id label set (multi-label)
labels_per_hadm = label_presc.groupby("HADM_ID")["med_class"].agg(lambda s: sorted(set(s)))
cohort = cohort.set_index("HADM_ID")
cohort["label_classes"] = labels_per_hadm
cohort["label_classes"] = cohort["label_classes"].apply(lambda x: x if isinstance(x, list) else [])
cohort["n_label_classes"] = cohort["label_classes"].apply(len)
cohort = cohort.reset_index()

med_feasibility = {}
class_counts_all = label_presc["med_class"].value_counts()
med_feasibility["n_discharge_prescription_rows_raw"] = int(len(disch_presc)) + int((disch_presc["is_antihtn"]).sum()) * 0  # placeholder
med_feasibility["n_discharge_prescription_rows_after_exclude"] = int(len(disch_presc))
med_feasibility["n_antihypertensive_rows_excluded_from_label"] = int(disch_presc["is_antihtn"].sum())
med_feasibility["n_label_eligible_rows"] = int(len(label_presc))
med_feasibility["n_distinct_label_classes"] = int(class_counts_all.shape[0])
med_feasibility["n_other_unmapped_rows"] = int((label_presc["med_class"] == "Other/Unmapped").sum())
med_feasibility["pct_rows_other_unmapped"] = round(
    100 * med_feasibility["n_other_unmapped_rows"] / max(1, med_feasibility["n_label_eligible_rows"]), 1
)
med_feasibility["admissions_with_zero_label_classes"] = int((cohort["n_label_classes"] == 0).sum())
med_feasibility["class_admission_counts"] = {
    cls: int((label_presc.groupby("HADM_ID")["med_class"].apply(lambda s: cls in set(s))).sum())
    for cls in class_counts_all.index
}
med_feasibility["singleton_classes_lt5_admissions"] = sorted(
    [c for c, n in med_feasibility["class_admission_counts"].items() if n < 5]
)
med_feasibility["n_singleton_classes_lt5_admissions"] = len(med_feasibility["singleton_classes_lt5_admissions"])

with open(RESULTS / "step4_medication_label_feasibility.json", "w") as f:
    json.dump(med_feasibility, f, indent=2)
print("=== Medication label feasibility ===")
print(json.dumps({k: v for k, v in med_feasibility.items() if k != "class_admission_counts"}, indent=2))

# ---------------------------------------------------------------------------
# Step 3: 80/20 patient-level split, seeded
# ---------------------------------------------------------------------------
rng = np.random.RandomState(42)
patients = cohort["SUBJECT_ID"].unique()
rng.shuffle(patients)
n_train = int(round(0.8 * len(patients)))
train_patients = set(patients[:n_train])
test_patients = set(patients[n_train:])

cohort["split"] = cohort["SUBJECT_ID"].apply(lambda s: "train" if s in train_patients else "test")

split_summary = {
    "n_patients_total": len(patients),
    "n_patients_train": len(train_patients),
    "n_patients_test": len(test_patients),
    "n_admissions_train": int((cohort["split"] == "train").sum()),
    "n_admissions_test": int((cohort["split"] == "test").sum()),
}

train_df = cohort[cohort["split"] == "train"]
test_df = cohort[cohort["split"] == "test"]

# class balance across splits
train_label_presc = label_presc[label_presc["HADM_ID"].isin(train_df["HADM_ID"])]
test_label_presc = label_presc[label_presc["HADM_ID"].isin(test_df["HADM_ID"])]

train_class_counts = {
    cls: int((train_label_presc.groupby("HADM_ID")["med_class"].apply(lambda s: cls in set(s))).sum())
    for cls in class_counts_all.index
}
test_class_counts = {
    cls: int((test_label_presc.groupby("HADM_ID")["med_class"].apply(lambda s: cls in set(s))).sum())
    for cls in class_counts_all.index
}

split_summary["train_class_counts"] = train_class_counts
split_summary["test_class_counts"] = test_class_counts
split_summary["n_classes_with_lt5_train_admissions"] = sum(1 for v in train_class_counts.values() if v < 5)
split_summary["n_classes_with_zero_test_admissions"] = sum(1 for v in test_class_counts.values() if v == 0)

# admission-condition-group counts per split (for baseline / RAG support-per-group check)
cond_train = train_df["admission_condition_group"].value_counts().to_dict()
cond_test = test_df["admission_condition_group"].value_counts().to_dict()
split_summary["admission_condition_counts_train"] = cond_train
split_summary["admission_condition_counts_test"] = cond_test
split_summary["n_condition_groups_with_lt5_train_admissions"] = sum(1 for v in cond_train.values() if v < 5)
split_summary["n_condition_groups_with_zero_test_admissions"] = sum(
    1 for g in cond_train if cond_test.get(g, 0) == 0
)

with open(RESULTS / "step3_split_audit.json", "w") as f:
    json.dump(split_summary, f, indent=2)

# persist the actual split for reuse by later phases
cohort.to_parquet(RESULTS / "cohort_with_split_and_labels.parquet", index=False)

print("\n=== Split summary ===")
print(json.dumps(
    {k: v for k, v in split_summary.items() if not k.endswith("_counts") and "condition_counts" not in k},
    indent=2,
))
print("\nAdmission condition groups (train) with <5 admissions:", split_summary["n_condition_groups_with_lt5_train_admissions"])
print("Admission condition groups present in train but absent from test:", split_summary["n_condition_groups_with_zero_test_admissions"])
print("Medication classes with <5 train admissions:", split_summary["n_classes_with_lt5_train_admissions"], "/", len(train_class_counts))
