"""Phase 5: build a chronological PatientTimeline from raw MIMIC-III tables.

No summarization, no recommendations, no LLM calls — every event carries its
full raw text/value and an EvidenceItem citing the source row. Compacting
this into a clinician-facing summary is ClinicalState's job (state_builder.py).
"""
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from ..schemas import EvidenceItem, PatientTimeline, SourceTable, TimelineEvent, TimelineEventType


def _evidence(
    source_table: SourceTable,
    row_id,
    subject_id: int,
    hadm_id: Optional[int],
    description: str,
    value: Optional[str] = None,
    timestamp=None,
    text_excerpt: Optional[str] = None,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=f"{source_table.value}-{row_id}",
        source_table=source_table,
        source_row_id=str(row_id),
        subject_id=subject_id,
        hadm_id=hadm_id,
        description=description,
        value=value,
        timestamp=_none_if_nat(timestamp),
        text_excerpt=text_excerpt,
    )


def _none_if_nat(value):
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return None
    return value


def build_patient_timeline(
    subject_id: int,
    hadm_id: int,
    admissions: pd.DataFrame,
    diagnoses_icd: pd.DataFrame,
    prescriptions: pd.DataFrame,
    noteevents: pd.DataFrame,
    labevents: pd.DataFrame,
    d_labitems: pd.DataFrame,
    d_icd_diagnoses: pd.DataFrame,
    generated_at: Optional[datetime] = None,
) -> PatientTimeline:
    """Assemble every recorded fact about one admission into a single,
    chronologically sorted timeline. Events with no usable timestamp are
    placed last (not dropped)."""
    adm_match = admissions[(admissions["SUBJECT_ID"] == subject_id) & (admissions["HADM_ID"] == hadm_id)]
    if adm_match.empty:
        raise ValueError(f"No ADMISSIONS row found for subject_id={subject_id}, hadm_id={hadm_id}")
    adm_row = adm_match.iloc[0]
    admittime = _none_if_nat(adm_row["ADMITTIME"])
    dischtime = _none_if_nat(adm_row["DISCHTIME"])

    events: list[TimelineEvent] = []

    events.append(
        TimelineEvent(
            event_id=f"admission-{hadm_id}",
            subject_id=subject_id,
            hadm_id=hadm_id,
            event_type=TimelineEventType.ADMISSION,
            timestamp=admittime,
            description=f"Admitted ({adm_row.get('ADMISSION_TYPE', 'unknown type')})",
            details={
                "admission_type": str(adm_row.get("ADMISSION_TYPE", "")),
                "admission_location": str(adm_row.get("ADMISSION_LOCATION", "")),
                "diagnosis_text": str(adm_row.get("DIAGNOSIS", "")),
            },
            evidence=_evidence(
                SourceTable.ADMISSIONS, adm_row["ROW_ID"], subject_id, hadm_id,
                "Admission record", timestamp=admittime,
            ),
        )
    )
    events.append(
        TimelineEvent(
            event_id=f"discharge-{hadm_id}",
            subject_id=subject_id,
            hadm_id=hadm_id,
            event_type=TimelineEventType.DISCHARGE,
            timestamp=dischtime,
            description=f"Discharged to {adm_row.get('DISCHARGE_LOCATION', 'unknown location')}",
            details={"discharge_location": str(adm_row.get("DISCHARGE_LOCATION", ""))},
            evidence=_evidence(
                SourceTable.ADMISSIONS, adm_row["ROW_ID"], subject_id, hadm_id,
                "Discharge record", timestamp=dischtime,
            ),
        )
    )

    diag = diagnoses_icd[diagnoses_icd["HADM_ID"] == hadm_id].merge(
        d_icd_diagnoses[["ICD9_CODE", "SHORT_TITLE"]], on="ICD9_CODE", how="left"
    )
    for _, row in diag.iterrows():
        title = row["SHORT_TITLE"] if pd.notna(row["SHORT_TITLE"]) else row["ICD9_CODE"]
        events.append(
            TimelineEvent(
                event_id=f"diagnosis-{row['ROW_ID']}",
                subject_id=subject_id,
                hadm_id=hadm_id,
                event_type=TimelineEventType.DIAGNOSIS,
                timestamp=admittime,  # DIAGNOSES_ICD has no per-code timestamp in MIMIC-III
                description=f"Diagnosis: {title} ({row['ICD9_CODE']})",
                details={"icd9_code": str(row["ICD9_CODE"]), "seq_num": int(row["SEQ_NUM"])},
                evidence=_evidence(
                    SourceTable.DIAGNOSES_ICD, row["ROW_ID"], subject_id, hadm_id,
                    f"Diagnosis code {row['ICD9_CODE']} ({title})", value=str(row["ICD9_CODE"]), timestamp=admittime,
                ),
            )
        )

    presc = prescriptions[prescriptions["HADM_ID"] == hadm_id]
    for _, row in presc.iterrows():
        start = _none_if_nat(row["STARTDATE"])
        events.append(
            TimelineEvent(
                event_id=f"medication-{row['ROW_ID']}",
                subject_id=subject_id,
                hadm_id=hadm_id,
                event_type=TimelineEventType.MEDICATION_ORDER,
                timestamp=start,
                description=f"Medication ordered: {row['DRUG']}",
                details={
                    "drug": str(row["DRUG"]),
                    "route": str(row.get("ROUTE", "")),
                    "startdate": str(row["STARTDATE"]) if pd.notna(row["STARTDATE"]) else None,
                    "enddate": str(row["ENDDATE"]) if pd.notna(row["ENDDATE"]) else None,
                },
                evidence=_evidence(
                    SourceTable.PRESCRIPTIONS, row["ROW_ID"], subject_id, hadm_id,
                    f"Prescription order: {row['DRUG']}", value=str(row["DRUG"]), timestamp=start,
                ),
            )
        )

    labs = labevents[labevents["HADM_ID"] == hadm_id].merge(
        d_labitems[["ITEMID", "LABEL"]], on="ITEMID", how="left"
    )
    for _, row in labs.iterrows():
        charttime = _none_if_nat(row["CHARTTIME"])
        label = row["LABEL"] if pd.notna(row["LABEL"]) else f"itemid {row['ITEMID']}"
        events.append(
            TimelineEvent(
                event_id=f"lab-{row['ROW_ID']}",
                subject_id=subject_id,
                hadm_id=hadm_id,
                event_type=TimelineEventType.LAB_RESULT,
                timestamp=charttime,
                description=f"Lab result: {label} = {row.get('VALUE', '')}",
                details={
                    "itemid": int(row["ITEMID"]),
                    "label": str(label),
                    "value": str(row.get("VALUE", "")),
                    "valuenum": float(row["VALUENUM"]) if pd.notna(row.get("VALUENUM")) else None,
                    "valueuom": str(row.get("VALUEUOM", "")),
                    "flag": str(row.get("FLAG", "")) if pd.notna(row.get("FLAG")) else None,
                },
                evidence=_evidence(
                    SourceTable.LABEVENTS, row["ROW_ID"], subject_id, hadm_id,
                    f"Lab result: {label}", value=str(row.get("VALUE", "")), timestamp=charttime,
                ),
            )
        )

    notes = noteevents[noteevents["HADM_ID"] == hadm_id]
    for _, row in notes.iterrows():
        note_time = _none_if_nat(row["CHARTTIME"]) or _none_if_nat(row["CHARTDATE"])
        text = str(row.get("TEXT", ""))
        events.append(
            TimelineEvent(
                event_id=f"note-{row['ROW_ID']}",
                subject_id=subject_id,
                hadm_id=hadm_id,
                event_type=TimelineEventType.NOTE,
                timestamp=note_time,
                description=f"Note: {row.get('CATEGORY', 'unknown category')} — {row.get('DESCRIPTION', '')}",
                details={"category": str(row.get("CATEGORY", "")), "text": text},
                evidence=_evidence(
                    SourceTable.NOTEEVENTS, row["ROW_ID"], subject_id, hadm_id,
                    f"{row.get('CATEGORY', 'Note')}: {row.get('DESCRIPTION', '')}",
                    timestamp=note_time, text_excerpt=text[:500],
                ),
            )
        )

    events.sort(key=lambda e: (e.timestamp is None, e.timestamp))

    return PatientTimeline(
        subject_id=subject_id,
        hadm_id=hadm_id,
        events=events,
        generated_at=generated_at or datetime.now(timezone.utc),
    )
