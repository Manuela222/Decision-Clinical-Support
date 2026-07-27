"""Load the eight MIMIC-III tables this project uses into validated pandas
DataFrames (Phase 2). "Validated" here means: the file exists, and it has
every required column for that table — clear, specific errors otherwise.

No cohort filtering, normalization, or business logic here — that belongs to
later phases (3+). This module only gets raw tables safely into memory.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .schema import TABLE_SPECS


class MimicDataError(Exception):
    """Raised when a required MIMIC-III table is missing or malformed."""


def _find_table_file(data_dir: Path, table_name: str) -> Path:
    candidates = [
        data_dir / f"{table_name}.csv",
        data_dir / f"{table_name}.csv.gz",
        data_dir / f"{table_name.lower()}.csv",
        data_dir / f"{table_name.lower()}.csv.gz",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise MimicDataError(
        f"Missing required MIMIC-III file for table '{table_name}': expected one of "
        f"{[c.name for c in candidates]} in directory '{data_dir}'."
    )


def load_table(data_dir: Path | str, table_name: str) -> pd.DataFrame:
    """Load one named MIMIC-III table, validating it against `TABLE_SPECS`."""
    if table_name not in TABLE_SPECS:
        raise MimicDataError(
            f"Unknown table '{table_name}'. Supported tables: {sorted(TABLE_SPECS.keys())}."
        )
    spec = TABLE_SPECS[table_name]
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise MimicDataError(f"MIMIC-III data directory does not exist: '{data_dir}'.")

    file_path = _find_table_file(data_dir, table_name)
    df = pd.read_csv(file_path, low_memory=False)

    missing_columns = [c for c in spec.required_columns if c not in df.columns]
    if missing_columns:
        raise MimicDataError(
            f"Table '{table_name}' loaded from '{file_path}' is missing required columns: "
            f"{missing_columns}. Columns found: {list(df.columns)}."
        )

    for date_col in spec.date_columns:
        # Cast to microsecond (not the pandas default nanosecond) resolution:
        # MIMIC-III's deidentification shifts dates across a very wide range
        # (real DOBs as early as ~1800, admissions shifted as late as ~2200+),
        # and datetime64[ns] arithmetic overflows well within that range.
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce").astype("datetime64[us]")

    return df


@dataclass
class MimicTables:
    patients: pd.DataFrame
    admissions: pd.DataFrame
    diagnoses_icd: pd.DataFrame
    prescriptions: pd.DataFrame
    noteevents: pd.DataFrame
    labevents: pd.DataFrame
    d_labitems: pd.DataFrame
    d_icd_diagnoses: pd.DataFrame


def load_mimic_tables(data_dir: Path | str) -> MimicTables:
    """Load all eight required MIMIC-III tables from `data_dir`.

    Raises MimicDataError (naming the specific table) on the first missing
    file or missing column, rather than partially loading.
    """
    return MimicTables(
        patients=load_table(data_dir, "PATIENTS"),
        admissions=load_table(data_dir, "ADMISSIONS"),
        diagnoses_icd=load_table(data_dir, "DIAGNOSES_ICD"),
        prescriptions=load_table(data_dir, "PRESCRIPTIONS"),
        noteevents=load_table(data_dir, "NOTEEVENTS"),
        labevents=load_table(data_dir, "LABEVENTS"),
        d_labitems=load_table(data_dir, "D_LABITEMS"),
        d_icd_diagnoses=load_table(data_dir, "D_ICD_DIAGNOSES"),
    )
