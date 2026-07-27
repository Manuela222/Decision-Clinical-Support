"""Builds a real `AppState` from the actual MIMIC_III_10k extract.

Same pipeline as Phase 18's `backend/scripts/run_integration.py` (cohort
selection, splitting, ground-truth extraction, timeline/clinical-state
construction, baseline stats, model training, RAG index) — reused here
instead of duplicated, since both need the identical real-data path.
Loads via duckdb over the pre-built parquet files, for the same reason
`run_integration.py` does: this repo's MIMIC_III_10k directory uses a
nonstandard per-table layout that Phase 2's loader does not target.
"""
from pathlib import Path
from typing import Callable, Optional

import duckdb
import pandas as pd

from cds.agent import LLMProvider, OpenAIProvider
from cds.baseline import compute_diagnosis_medication_stats
from cds.cohort import select_cohort
from cds.medications import (
    EXCLUDED_CLASS,
    is_antihypertensive_class,
    map_to_medication_class,
    normalize_medication_name,
)
from cds.model import train_model
from cds.retrieval import build_patient_profile_index
from cds.split import add_split_column
from cds.timeline import build_clinical_state, build_patient_timeline

from .app_state import AppState, CohortEntry

REPO_ROOT = Path(__file__).resolve().parents[3]
PARQUET_DIR = REPO_ROOT / "MIMIC_III_10k" / "work" / "parquet"


def _load_table(con: duckdb.DuckDBPyConnection, name: str) -> pd.DataFrame:
    return con.execute(f"select * from read_parquet('{(PARQUET_DIR / (name + '.parquet')).as_posix()}')").df()


def build_real_app_state(llm_provider_factory: Optional[Callable[[], LLMProvider]] = None) -> AppState:
    con = duckdb.connect()

    patients = _load_table(con, "patients")
    admissions = _load_table(con, "admissions")
    diagnoses_icd = _load_table(con, "diagnoses_icd")
    prescriptions = _load_table(con, "prescriptions")
    labevents = _load_table(con, "labevents")
    d_labitems = _load_table(con, "d_labitems")
    d_icd_diagnoses = _load_table(con, "d_icd_diagnoses")
    noteevents = _load_table(con, "noteevents_clean").rename(
        columns={"row_id": "ROW_ID", "subject_id": "SUBJECT_ID", "hadm_id": "HADM_ID",
                 "cgid": "CGID", "iserror_flag": "ISERROR"}
    )

    cohort = select_cohort(patients, admissions, diagnoses_icd, prescriptions)
    cohort["admission_reason"] = cohort["other_condition_groups"].apply(
        lambda groups: groups[0] if len(groups) == 1 else " & ".join(groups) if groups else "Unknown"
    )
    cohort = add_split_column(cohort, seed=42)

    cohort_hadm_ids = set(cohort["hadm_id"])
    diagnoses_icd_f = diagnoses_icd[diagnoses_icd["HADM_ID"].isin(cohort_hadm_ids)]
    prescriptions_f = prescriptions[prescriptions["HADM_ID"].isin(cohort_hadm_ids)].copy()
    labevents_f = labevents[labevents["HADM_ID"].isin(cohort_hadm_ids)]
    noteevents_f = noteevents[noteevents["HADM_ID"].isin(cohort_hadm_ids)]

    dischtime_by_hadm = cohort.set_index("hadm_id")["dischtime"]
    prescriptions_f["DISCHTIME"] = prescriptions_f["HADM_ID"].map(dischtime_by_hadm)
    is_discharge_med = (
        (prescriptions_f["STARTDATE"].isna() | (prescriptions_f["STARTDATE"] <= prescriptions_f["DISCHTIME"]))
        & (prescriptions_f["ENDDATE"].isna() | (prescriptions_f["ENDDATE"].dt.normalize() >= prescriptions_f["DISCHTIME"].dt.normalize()))
    )
    disch_presc = prescriptions_f[is_discharge_med].copy()
    disch_presc["normalized"] = disch_presc["DRUG"].apply(normalize_medication_name)
    disch_presc["med_class"] = disch_presc["normalized"].apply(map_to_medication_class)
    disch_presc = disch_presc[disch_presc["med_class"].notna() & (disch_presc["med_class"] != EXCLUDED_CLASS)]
    disch_presc = disch_presc[~disch_presc["med_class"].apply(is_antihypertensive_class)]
    ground_truth_by_hadm = disch_presc.groupby("HADM_ID")["med_class"].apply(lambda s: sorted(set(s))).to_dict()

    timelines, clinical_states, cohort_entries = {}, {}, []
    for _, row in cohort.iterrows():
        key = (row["subject_id"], row["hadm_id"])
        timeline = build_patient_timeline(
            subject_id=row["subject_id"], hadm_id=row["hadm_id"],
            admissions=admissions, diagnoses_icd=diagnoses_icd_f, prescriptions=prescriptions_f,
            noteevents=noteevents_f, labevents=labevents_f, d_labitems=d_labitems, d_icd_diagnoses=d_icd_diagnoses,
        )
        clinical_state = build_clinical_state(
            timeline, age=int(row["age"]), gender=row["gender"],
            condition_groups=row["condition_groups"], admission_reason=row["admission_reason"],
        )
        timelines[key] = timeline
        clinical_states[key] = clinical_state
        cohort_entries.append(
            CohortEntry(
                subject_id=row["subject_id"], hadm_id=row["hadm_id"], age=int(row["age"]), gender=row["gender"],
                admission_reason=row["admission_reason"], hypertension_status=clinical_state.hypertension_status.value,
                split=row["split"],
            )
        )

    ground_truth_classes = {
        (row["subject_id"], row["hadm_id"]): ground_truth_by_hadm.get(row["hadm_id"], [])
        for _, row in cohort.iterrows()
    }

    train_cohort = cohort[cohort["split"] == "train"]
    train_states = [clinical_states[(row["subject_id"], row["hadm_id"])] for _, row in train_cohort.iterrows()]
    train_labels = [ground_truth_by_hadm.get(row["hadm_id"], []) for _, row in train_cohort.iterrows()]

    diagnosis_medication_stats = compute_diagnosis_medication_stats(
        list(train_cohort["admission_reason"]), train_labels
    )
    trained_model_artifact = train_model(train_states, train_labels, seed=42)
    patient_profile_index = build_patient_profile_index(train_states, train_labels)

    return AppState(
        cohort=cohort_entries,
        timelines=timelines,
        clinical_states=clinical_states,
        ground_truth_classes=ground_truth_classes,
        diagnosis_medication_stats=diagnosis_medication_stats,
        trained_model_artifact=trained_model_artifact,
        patient_profile_index=patient_profile_index,
        llm_provider_factory=llm_provider_factory or OpenAIProvider,
    )
