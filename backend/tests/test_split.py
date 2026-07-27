"""Phase 6: patient-level train/test split tests."""
import pandas as pd
import pytest

from cds.split import SPLIT_COLUMN, add_split_column, load_cohort_split, save_cohort_split, split_cohort


def _fake_cohort(n_patients=20, admissions_per_patient=1):
    rows = []
    hadm_id = 1000
    for subject_id in range(1, n_patients + 1):
        for _ in range(admissions_per_patient):
            rows.append({"subject_id": subject_id, "hadm_id": hadm_id, "age": 50, "gender": "F"})
            hadm_id += 1
    return pd.DataFrame(rows)


def test_split_is_deterministic_given_same_seed():
    cohort = _fake_cohort()
    train_a, test_a = split_cohort(cohort, seed=42)
    train_b, test_b = split_cohort(cohort, seed=42)
    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(test_a, test_b)


def test_split_differs_with_different_seed():
    cohort = _fake_cohort()
    _, test_a = split_cohort(cohort, seed=1)
    _, test_b = split_cohort(cohort, seed=2)
    assert set(test_a["subject_id"]) != set(test_b["subject_id"])


def test_split_proportions_approximately_80_20():
    cohort = _fake_cohort(n_patients=100)
    train_df, test_df = split_cohort(cohort, test_size=0.2, seed=42)
    assert len(test_df) == 20
    assert len(train_df) == 80


def test_split_no_patient_overlap_between_train_and_test():
    cohort = _fake_cohort(n_patients=50)
    train_df, test_df = split_cohort(cohort, seed=7)
    assert set(train_df["subject_id"]).isdisjoint(set(test_df["subject_id"]))


def test_split_keeps_all_of_a_patients_admissions_together():
    cohort = _fake_cohort(n_patients=30, admissions_per_patient=3)
    train_df, test_df = split_cohort(cohort, seed=42)
    train_subjects = set(train_df["subject_id"])
    test_subjects = set(test_df["subject_id"])
    # every subject appears exactly 3 times, entirely on one side
    for subject_id in train_subjects:
        assert (train_df["subject_id"] == subject_id).sum() == 3
    for subject_id in test_subjects:
        assert (test_df["subject_id"] == subject_id).sum() == 3
    assert train_subjects.isdisjoint(test_subjects)


def test_split_covers_every_row_exactly_once():
    cohort = _fake_cohort(n_patients=40, admissions_per_patient=2)
    train_df, test_df = split_cohort(cohort, seed=42)
    assert len(train_df) + len(test_df) == len(cohort)
    assert set(train_df["hadm_id"]) | set(test_df["hadm_id"]) == set(cohort["hadm_id"])
    assert set(train_df["hadm_id"]).isdisjoint(set(test_df["hadm_id"]))


def test_invalid_test_size_raises():
    cohort = _fake_cohort()
    with pytest.raises(ValueError, match="test_size"):
        split_cohort(cohort, test_size=1.5)
    with pytest.raises(ValueError, match="test_size"):
        split_cohort(cohort, test_size=0)


def test_add_split_column_adds_expected_labels():
    cohort = _fake_cohort(n_patients=10)
    result = add_split_column(cohort, seed=42)
    assert set(result[SPLIT_COLUMN]) == {"train", "test"}
    assert len(result) == len(cohort)


def test_save_and_load_cohort_split_round_trips(tmp_path):
    cohort = _fake_cohort(n_patients=10)
    with_split = add_split_column(cohort, seed=42)
    path = tmp_path / "cohort_split.parquet"
    save_cohort_split(with_split, path)

    loaded = load_cohort_split(path)
    pd.testing.assert_frame_equal(loaded, with_split)


def test_save_cohort_split_requires_split_column(tmp_path):
    cohort = _fake_cohort(n_patients=5)
    with pytest.raises(ValueError, match="split"):
        save_cohort_split(cohort, tmp_path / "bad.parquet")


def test_load_cohort_split_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No persisted cohort split found"):
        load_cohort_split(tmp_path / "does_not_exist.parquet")
