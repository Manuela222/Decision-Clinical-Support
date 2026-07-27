"""One-off integration run for Phase 18's final report: wires the real
`cds` pipeline (Phases 3-11, 14) against the actual MIMIC_III_10k extract
and reports real baseline-vs-trained-model numbers.

Not part of the `cds` package and not covered by pytest -- same status as
`phase0_feasibility/`'s scripts: a reporting/analysis script, not library
code. Loads data via duckdb over the pre-built parquet files (as Phase 0's
audit did) rather than Phase 2's standard-MIMIC-CSV-layout loader, because
this repo's actual MIMIC_III_10k directory uses a nonstandard per-table
folder layout (see Phase 2 completion notes) -- this script is the
concrete resolution of that previously-flagged wiring gap, scoped to what
Phase 18 needs (it does not retrofit Phase 2's loader itself).

Agent evaluation is NOT run here: it requires a live OPENAI_API_KEY, which
this environment does not have. Baseline and trained-model numbers below
are real; agent behavior is demonstrated only via the mocked scenarios in
the test suite and the Phase 17 browser smoke test.
"""
import json
import time
from pathlib import Path

import duckdb
import pandas as pd

from cds.baseline import compute_diagnosis_medication_stats, recommend_baseline
from cds.cohort import select_cohort
from cds.evaluation import evaluate_medication_recommendations
from cds.medications import (
    EXCLUDED_CLASS,
    is_antihypertensive_class,
    map_to_medication_class,
    normalize_medication_name,
)
from cds.model import recommend_trained_model, train_model
from cds.retrieval import build_patient_profile_index
from cds.schemas import RecommendationMethod
from cds.split import add_split_column
from cds.timeline import build_clinical_state, build_patient_timeline

REPO_ROOT = Path(__file__).resolve().parents[2]
PARQUET_DIR = REPO_ROOT / "MIMIC_III_10k" / "work" / "parquet"
RESULTS_DIR = REPO_ROOT / "integration_results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_table(con: duckdb.DuckDBPyConnection, name: str) -> pd.DataFrame:
    return con.execute(f"select * from read_parquet('{(PARQUET_DIR / (name + '.parquet')).as_posix()}')").df()


def main() -> None:
    t0 = time.time()
    con = duckdb.connect()

    print("Loading real MIMIC_III_10k tables via duckdb...")
    patients = load_table(con, "patients")
    admissions = load_table(con, "admissions")
    diagnoses_icd = load_table(con, "diagnoses_icd")
    prescriptions = load_table(con, "prescriptions")
    labevents = load_table(con, "labevents")
    d_labitems = load_table(con, "d_labitems")
    d_icd_diagnoses = load_table(con, "d_icd_diagnoses")
    noteevents = load_table(con, "noteevents_clean").rename(
        columns={"row_id": "ROW_ID", "subject_id": "SUBJECT_ID", "hadm_id": "HADM_ID",
                 "cgid": "CGID", "iserror_flag": "ISERROR"}
    )
    print(f"  loaded in {time.time() - t0:.1f}s")

    print("Selecting cohort (Phase 3)...")
    cohort = select_cohort(patients, admissions, diagnoses_icd, prescriptions)
    cohort["admission_reason"] = cohort["other_condition_groups"].apply(
        lambda groups: groups[0] if len(groups) == 1 else " & ".join(groups) if groups else "Unknown"
    )
    print(f"  cohort: {len(cohort)} admissions, {cohort['subject_id'].nunique()} patients")

    print("Splitting 80/20 patient-level (Phase 6, seed=42)...")
    cohort = add_split_column(cohort, seed=42)
    print(f"  train: {(cohort['split'] == 'train').sum()}  test: {(cohort['split'] == 'test').sum()}")

    cohort_hadm_ids = set(cohort["hadm_id"])
    diagnoses_icd_f = diagnoses_icd[diagnoses_icd["HADM_ID"].isin(cohort_hadm_ids)]
    prescriptions_f = prescriptions[prescriptions["HADM_ID"].isin(cohort_hadm_ids)].copy()
    labevents_f = labevents[labevents["HADM_ID"].isin(cohort_hadm_ids)]
    noteevents_f = noteevents[noteevents["HADM_ID"].isin(cohort_hadm_ids)]

    print("Computing ground-truth discharge medication classes (Phase 0/4 heuristic)...")
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

    print("Building PatientTimeline + ClinicalState for every cohort admission...")
    clinical_states_by_hadm = {}
    for _, row in cohort.iterrows():
        timeline = build_patient_timeline(
            subject_id=row["subject_id"], hadm_id=row["hadm_id"],
            admissions=admissions, diagnoses_icd=diagnoses_icd_f, prescriptions=prescriptions_f,
            noteevents=noteevents_f, labevents=labevents_f, d_labitems=d_labitems, d_icd_diagnoses=d_icd_diagnoses,
        )
        clinical_states_by_hadm[row["hadm_id"]] = build_clinical_state(
            timeline, age=int(row["age"]), gender=row["gender"],
            condition_groups=row["condition_groups"], admission_reason=row["admission_reason"],
        )
    print(f"  built {len(clinical_states_by_hadm)} ClinicalStates ({time.time() - t0:.1f}s elapsed)")

    train_cohort = cohort[cohort["split"] == "train"]
    test_cohort = cohort[cohort["split"] == "test"]
    train_states = [clinical_states_by_hadm[h] for h in train_cohort["hadm_id"]]
    train_labels = [ground_truth_by_hadm.get(h, []) for h in train_cohort["hadm_id"]]
    test_states = [clinical_states_by_hadm[h] for h in test_cohort["hadm_id"]]
    test_labels = [ground_truth_by_hadm.get(h, []) for h in test_cohort["hadm_id"]]

    print("Building baseline stats (Phase 8) and RAG index (Phase 11) from TRAIN split only...")
    diagnosis_medication_stats = compute_diagnosis_medication_stats(
        list(train_cohort["admission_reason"]), train_labels
    )
    patient_profile_index = build_patient_profile_index(train_states, train_labels)

    print("Training the model (Phase 9) on TRAIN split only...")
    artifact = train_model(train_states, train_labels, seed=42)
    print(f"  model card validation metrics: {artifact.model_card.validation_metrics}")

    print("Running baseline + trained model on the TEST split...")
    baseline_predicted, model_predicted = [], []
    for state in test_states:
        baseline_result = recommend_baseline(state, diagnosis_medication_stats)
        model_result = recommend_trained_model(state, artifact)
        baseline_predicted.append([r.medication_class for r in baseline_result.recommended_medications])
        model_predicted.append([r.medication_class for r in model_result.recommended_medications])

    baseline_eval = evaluate_medication_recommendations(baseline_predicted, test_labels, RecommendationMethod.BASELINE)
    model_eval = evaluate_medication_recommendations(model_predicted, test_labels, RecommendationMethod.TRAINED_MODEL)

    comparison = pd.DataFrame(
        [
            {"method": "baseline", "n_admissions_evaluated": baseline_eval.n_admissions_evaluated,
             "micro_precision": baseline_eval.micro_precision, "micro_recall": baseline_eval.micro_recall, "micro_f1": baseline_eval.micro_f1,
             "macro_precision": baseline_eval.macro_precision, "macro_recall": baseline_eval.macro_recall, "macro_f1": baseline_eval.macro_f1},
            {"method": "trained_model", "n_admissions_evaluated": model_eval.n_admissions_evaluated,
             "micro_precision": model_eval.micro_precision, "micro_recall": model_eval.micro_recall, "micro_f1": model_eval.micro_f1,
             "macro_precision": model_eval.macro_precision, "macro_recall": model_eval.macro_recall, "macro_f1": model_eval.macro_f1},
        ]
    )
    print("\n=== Comparison (real test split, n={} ) ===".format(len(test_states)))
    print(comparison.to_string(index=False))

    comparison.to_csv(RESULTS_DIR / "baseline_vs_model_comparison.csv", index=False)
    (RESULTS_DIR / "model_card.json").write_text(artifact.model_card.model_dump_json(indent=2))
    (RESULTS_DIR / "cohort_summary.json").write_text(
        json.dumps(
            {
                "n_admissions": len(cohort),
                "n_patients": int(cohort["subject_id"].nunique()),
                "n_train_admissions": len(train_cohort),
                "n_test_admissions": len(test_cohort),
                "n_distinct_label_classes_train": len({c for labels in train_labels for c in labels}),
                "n_distinct_label_classes_test": len({c for labels in test_labels for c in labels}),
            },
            indent=2,
        )
    )
    print(f"\nSaved results to {RESULTS_DIR} ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    main()
