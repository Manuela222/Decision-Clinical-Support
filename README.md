# Clinical Decision Support — Discharge Medication Recommendation for Multimorbid (Hypertension + Admission Condition) Patients

Prototype clinical decision support system using MIMIC-III. Recommends
discharge medication classes for patients admitted for one primary
morbidity who also carry hypertension as a chronic comorbidity.
Hypertension is a conditioning factor and safety constraint — **never the
prediction target**.

Three methods, always compared against the same held-out test set: a
deterministic frequency baseline, a supervised multi-label model, and an
LLM agent (OpenAI, tool use + retrieval). See `FINAL_REPORT.md` for full
methodology, real results, and known limitations — `FEASIBILITY_REPORT.md`
for the Phase 0 go/no-go audit everything else builds on.

**Status: prototype. Not validated for clinical use.**

## Project layout

- `phase0_feasibility/` — Phase 0 audit scripts and results (not
  production code; kept for provenance of the feasibility numbers).
- `backend/` — Python package `cds` (business logic, schemas, pipeline)
  and `cds_api` (FastAPI backend), plus `pytest` tests.
- `frontend/` — Node.js/Express app serving the browser UI.
- `MIMIC_III_10k/` — **not included in this repository** (multi-GB, not
  redistributable). Supply your own local MIMIC-III extract at this path
  to run against real data — see "Getting MIMIC-III data" below.
- `integration_results/` — real baseline-vs-model numbers from the last
  integration run against actual MIMIC-III data (small, checked in).

## Getting MIMIC-III data

This repo does not ship any MIMIC-III data (`.gitignore`'d — the local
extract used during development is ~7 GB). To run against real data:

1. Get credentialed access to MIMIC-III via [PhysioNet](https://physionet.org/content/mimiciii/)
   (requires completing their data-use training).
2. Place `PATIENTS`, `ADMISSIONS`, `DIAGNOSES_ICD`, `PRESCRIPTIONS`,
   `NOTEEVENTS`, `LABEVENTS`, `D_LABITEMS`, `D_ICD_DIAGNOSES` under
   `MIMIC_III_10k/work/parquet/` as parquet files (see
   `backend/src/cds_api/real_data.py` for the exact table names/paths
   expected, and `phase0_feasibility/run_audit.py` for how the original
   extract was prepared).

Without real data, everything still runs against fake/synthetic fixtures —
the full test suite, `cds_api.dev_server` (a demo backend with a small
built-in synthetic cohort), and the frontend's "New Patient" page all work
with **zero MIMIC-III data required**.

## Setup

Requires Python ≥3.10 and Node.js ≥18.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
pip install -e ./backend
cd frontend && npm install && cd ..
```

(`backend/requirements.txt` is a full pinned freeze for exact
reproducibility; `backend/pyproject.toml` is the canonical, looser
dependency declaration if you prefer `pip install -e "./backend[dev,serve]"`.)

## Running the tests

```bash
source .venv/bin/activate
cd backend && python -m pytest
```

200 tests, all against fake or synthetic fixtures — no MIMIC-III data or
live network/API calls required (see `FINAL_REPORT.md` Section 2.18).

## Using the OpenAI agent

The agent method (`cds.agent`) calls the real OpenAI API and needs an API
key. It is never hardcoded — set it as an environment variable before
starting the backend:

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o-mini"   # optional, this is the default
```

Without a key, everything else still works — the baseline and trained
model don't need one, and `cds.agent.MockLLMProvider` (a scripted,
deterministic stand-in) covers the entire test suite.

## Running the interface

Two things need to run at once: the FastAPI backend (port 8000) and the
Node.js frontend (port 3000). The frontend only ever talks to the backend
over HTTP — never to OpenAI or any data file directly.

**Backend — pick one:**

```bash
# Demo mode: a small built-in synthetic cohort, no MIMIC-III data needed.
# Agent calls still go to real OpenAI if OPENAI_API_KEY is set.
cd backend && python -m cds_api.dev_server

# Real mode: builds the cohort from your local MIMIC_III_10k extract
# (cohort selection, timelines, model training, RAG index — takes ~15-20s
# to start). Requires the data described above.
cd backend && python -m cds_api.real_server
```

Both serve the same API on `http://127.0.0.1:8000`.

**Frontend (in a second terminal):**

```bash
cd frontend && npm start
```

Then open **http://localhost:3000**. "Existing Patient" compares all
three methods against a held-out test admission's true discharge
medications; "New Patient" runs the agent only, against a manually entered
or auto-filled synthetic patient. See `FINAL_REPORT.md` Section 5 for how
each page works and how to read the results.

## Comparing methods / evaluation

`POST /evaluate` (body: `{"include_agent": true|false}`) runs baseline,
trained model, and — if requested — the agent against the full test split
and returns micro/macro precision/recall/F1 for each, using the identical
metric function for all three (`cds.evaluation.compute_multilabel_metrics`,
see `FINAL_REPORT.md` Section 5.4 for how to read it). Including the agent
makes one real, billed OpenAI call per tool step per test admission — be
aware of the cost before setting `include_agent: true` on the full test
split.

`backend/scripts/run_integration.py` is the one-off script that produced
`integration_results/` — baseline vs. trained model on the real
128-admission test split (no agent, no API cost).

## Synthetic data — read before using any "New Patient" / demo data

`backend/src/cds/synthetic/` generates **fully synthetic** patient records
(`generate_synthetic_clinical_state`, `generate_synthetic_cohort`) for use
in automated tests, `cds_api.dev_server`, and the frontend's "New Patient"
page.

**This data is not real. It is not derived from, sampled from, or a
subset of any real MIMIC-III patient.** Every value comes from fixed lists
and random ranges hardcoded in `cds/synthetic/generator.py`. Subject and
admission IDs are always negative integers specifically so they can never
be confused with a real MIMIC-III `SUBJECT_ID`/`HADM_ID` (both always
positive in the real data).

Do not use output from this generator to make claims about real patients,
and do not present it as clinical evidence of anything — its only purpose
is exercising the rest of the system without touching real data.

## Known data limitations (carried from Phase 0/5)

- Blood pressure readings live in MIMIC-III's `CHARTEVENTS` table, which
  this project does not load. Hypertension status is inferred from
  diagnosis codes and antihypertensive-medication presence only — never
  from an actual BP reading. See `cds/timeline/state_builder.py`.
- The Elixhauser comorbidity mapping (`cds/cohort/elixhauser.py`) was
  reconstructed from the published Quan et al. 2005 algorithm by hand and
  has not been cross-validated against a reference implementation (e.g.
  the R `comorbidity` package). Treat comorbidity groupings as
  approximate.
- The medication synonym/class dictionary
  (`cds/medications/data/default_dictionary.yaml`) covers the
  highest-volume drugs seen in the Phase 0 cohort audit, not the full drug
  vocabulary — anything not listed falls back to `Other/Unmapped` rather
  than being guessed at.

See `FINAL_REPORT.md` Section 7 for the full list of known gaps and
recommended next steps.
