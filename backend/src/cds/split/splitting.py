"""Phase 6: patient-level 80/20 train/test split, seeded and deterministic.

Split at SUBJECT_ID (not HADM_ID) so a patient's admissions never straddle
both sides — the same leakage concern validated in the Phase 0 audit.
Persisted to disk so baseline (Phase 8), trained model (Phase 9), and agent
(Phase 13) are all scored against the exact same test set in Phase 14.
"""
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

SPLIT_COLUMN = "split"
TRAIN = "train"
TEST = "test"


def add_split_column(
    cohort: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    subject_id_column: str = "subject_id",
) -> pd.DataFrame:
    """Return a copy of `cohort` with a `split` column ('train'/'test'),
    assigned per-patient so all of a patient's admissions land on one side."""
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

    subjects = np.sort(cohort[subject_id_column].unique())
    rng = np.random.RandomState(seed)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)

    n_test = max(1, round(len(shuffled) * test_size)) if len(shuffled) > 1 else 0
    test_subjects = set(shuffled[:n_test])

    result = cohort.copy()
    result[SPLIT_COLUMN] = result[subject_id_column].apply(lambda s: TEST if s in test_subjects else TRAIN)
    return result


def split_cohort(
    cohort: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    subject_id_column: str = "subject_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper around `add_split_column`: returns (train_df, test_df)."""
    with_split = add_split_column(cohort, test_size=test_size, seed=seed, subject_id_column=subject_id_column)
    train_df = with_split[with_split[SPLIT_COLUMN] == TRAIN].drop(columns=[SPLIT_COLUMN]).reset_index(drop=True)
    test_df = with_split[with_split[SPLIT_COLUMN] == TEST].drop(columns=[SPLIT_COLUMN]).reset_index(drop=True)
    return train_df, test_df


def save_cohort_split(cohort_with_split: pd.DataFrame, path: "Path | str") -> None:
    """Persist a cohort DataFrame that already has a `split` column."""
    if SPLIT_COLUMN not in cohort_with_split.columns:
        raise ValueError(f"cohort_with_split must have a '{SPLIT_COLUMN}' column; call add_split_column() first.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cohort_with_split.to_parquet(path, index=False)


def load_cohort_split(path: "Path | str") -> pd.DataFrame:
    """Load a previously persisted cohort+split DataFrame."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No persisted cohort split found at '{path}'.")
    return pd.read_parquet(path)
