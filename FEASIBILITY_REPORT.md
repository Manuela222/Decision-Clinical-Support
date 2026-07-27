# Phase 0 — Feasibility Audit Report

**Date:** 2026-07-26
**Data source:** `MIMIC_III_10k/` — a pre-extracted **~10,000-subject subset** of MIMIC-III
(NOT the full MIMIC-III database, which has ~46,520 subjects / ~58,976 admissions).
All counts in this report are on that subset. See "Scaling to full MIMIC-III" below.

**Recommendation: GO**, with caveats. The filtered cohort is small (643 admissions /
622 patients on the 10k subset) but clears the project's own stated minimum bar
("cohort under a few hundred patients" is the no-go threshold; we have >600), core
timeline data is >95% complete, and the multi-label medication-class space is not
dominated by singletons. The weak points are (a) an exploratory, non-clinically-validated
comorbidity grouping and medication-class mapping that both need real implementation
in Phases 3/4 rather than reuse of this audit code as-is, and (b) several admission-condition
groups and 2 medication classes with too few examples (<5) to model individually —
these should be merged into "Other" buckets or excluded rather than driving false confidence.

---

## 1. Cohort filter audit

Filters applied cumulatively, exactly as specified in the project brief:

| Step | Filter | Admissions | Patients |
|---|---|---:|---:|
| 0 | All admissions in the 10k subset | 12,911 | 10,000 |
| 1 | Age ≥ 18 at admission (deidentification-corrected) | 10,134 | 7,309 |
| 2 | Exactly 2 Elixhauser comorbidity groups, one = Hypertension | 809 | 780 |
| 3 | Admission has ≥1 row in PRESCRIPTIONS | **643** | **622** |

**Age correction detail:** MIMIC-III shifts date-of-birth for patients aged >89 at
first admission so that computed age comes out to ~300 years (a deliberate
deidentification artifact, documented by the MIMIC-III team). We detected 494
admissions with raw computed age >89 and treated them as real adults (age floored
to 90 for reporting) rather than excluding them as bad data. Excluded strictly-pediatric
admissions (age <18, uncorrected): 2,777.

**Comorbidity grouping scheme used:** Quan et al. (2005) "Elixhauser" ICD-9-CM
coding algorithm, re-implemented by hand in `phase0_feasibility/elixhauser.py`
against ICD-9 code-prefix rules reconstructed from the published literature (not
copied from a validated package such as R's `comorbidity` or AHRQ's SAS/R software).

> **ASSUMPTION FLAG:** This hand-built Elixhauser mapping has *not* been
> cross-validated row-for-row against a reference implementation. It is adequate
> for a go/no-go check but **must be revalidated** (e.g. diffed against the
> `comorbidity` R package or AHRQ's official software on the same extract, or
> replaced with a maintained Python package if one is adopted) before Phase 3
> treats it as ground truth.

> **ASSUMPTION FLAG:** Quan's algorithm defines separate "Hypertension,
> uncomplicated" and "Hypertension, complicated" categories. We merged them into
> a single "Hypertension" group (matching the original 1998 Elixhauser formulation
> and the project's framing of hypertension as one condition). Some codes in the
> "complicated" bucket are shared with Congestive Heart Failure / Renal Failure by
> design of the source algorithm — an admission can therefore register in both
> Hypertension and CHF/Renal Failure simultaneously, which is intended, not a bug.

> **ASSUMPTION FLAG:** "Exactly 2 clinically grouped conditions per admission"
> was interpreted as: across **all** ICD-9 diagnosis codes recorded for the
> admission (not just the principal diagnosis, `SEQ_NUM=1`), the number of
> *distinct* Elixhauser categories touched is exactly 2, one being Hypertension.
> This means the "admission condition" (the other group) is not guaranteed to be
> the actual principal reason for admission — it is simply "the one other chronic
> condition category present alongside hypertension." This is a meaningful
> simplification worth revisiting in Phase 3 if SEQ_NUM-based principal-diagnosis
> logic is preferred instead.

**Context on filter selectivity:** among age-eligible admissions, the distribution
of *how many* distinct Elixhauser groups an admission touches is broad (median
~3, e.g. 1,352 admissions have exactly 1 group, 1,967 have exactly 2 of any kind,
660 have 6, etc.) — 5,043 admissions have hypertension present at all, but only
809 have it as one of *exactly two* groups. This two-condition constraint is by
far the most selective filter, which is expected given real patients are usually
multimorbid with more than 2 categories.

**Admission-condition-group spread (the non-HTN group), final cohort, n=643:**

| Condition group | Admissions |
|---|---:|
| Diabetes, uncomplicated | 120 |
| Cardiac arrhythmias | 114 |
| Fluid and electrolyte disorders | 48 |
| Peripheral vascular disorders | 46 |
| Chronic pulmonary disease | 42 |
| Valvular disease | 41 |
| Congestive heart failure | 27 |
| Other neurological disorders | 27 |
| Renal failure | 20 |
| Solid tumor without metastasis | 19 |
| Hypothyroidism | 18 |
| Alcohol abuse | 15 |
| Depression | 14 |
| Coagulopathy | 13 |
| Metastatic cancer / Diabetes, complicated / Obesity | 12 each |
| Liver disease | 9 |
| Pulmonary circulation disorders / Psychoses / Blood loss anemia | 5 each |
| Rheumatoid arthritis / Drug abuse | 4 each |
| Lymphoma / Deficiency anemia / Paralysis | 3 each |
| AIDS/HIV | 2 |

The top 6 groups (Diabetes-uncomplicated, Cardiac arrhythmias, Fluid/electrolyte,
PVD, COPD, Valvular disease) cover 66% of the cohort. The long tail (11 groups
with ≤5 admissions) is too sparse to model per-group and should be bucketed as
"Other admission condition" or excluded from group-specific baseline/RAG logic.

---

## 2. Patient-level timeline completeness audit (n=643 admissions)

| Field | Present | Missing |
|---|---:|---:|
| Discharge summary note | 616 (95.8%) | 27 |
| Any clinical note | 642 (99.8%) | 1 |
| Any lab event | 640 (99.5%) | 3 |
| Any prescription | 643 (100%) | 0 (guaranteed by cohort filter) |
| Prior admission (same patient, earlier ADMITTIME) | 105 (16.3%) | 538 (83.7% first-seen admission) |
| **All core fields (discharge summary + labs + prescriptions)** | **615 (95.6%)** | **28 (4.4%)** |

95.6% of the filtered cohort has a fully complete core timeline. The 28 partial
cases (missing discharge summary, and/or labs) should be handled by the
`missing_information` flags in `ClinicalState` (Phase 5) rather than dropped, since
they're a small fraction and dropping them would bias the cohort toward
better-documented stays. Prior-admission history is available for only 16% of
patients — expected for a subset extract and fine, since prior admissions are
contextual/optional, not a required field.

---

## 3. 80/20 patient-level split audit

Seeded (seed=42), split at `SUBJECT_ID` level to avoid leakage, on the final
n=622-patient / 643-admission cohort:

| | Patients | Admissions |
|---|---:|---:|
| Train | 498 | 515 |
| Test | 124 | 128 |

**Admission-condition-group support:** of 26 groups, **8 have fewer than 5
admissions in train**, and **4 groups present in train have zero admissions in
test** (Pulmonary circulation disorders, Rheumatoid arthritis, Lymphoma,
Deficiency anemia). This means a strict "one baseline/RAG-index-per-condition-group"
design will have several groups with no train-side statistical support and no
test-side evaluation coverage. **Recommendation carried into Phase 3/8:** collapse
condition groups with <5 train admissions into an explicit "Other/rare admission
condition" bucket for baseline and RAG purposes, rather than trying to model each
of the 26 groups independently.

**Medication-class label support** (see Section 4 for how classes were derived):
of 32 non-antihypertensive classes, only **2 have fewer than 5 train admissions**
("anticholinergic": 2 train/0 test, "antibiotic - macrolide": 2 train/3 test —
note the latter's test count exceeding train count is exactly the kind of
small-n instability this flag exists to catch). The other 30 classes all have
≥9 train admissions, several with 200+. **This label space is workable for
multi-label classification** without being dominated by singletons, provided the
2 near-empty classes are merged into "Other" or dropped.

Full per-class and per-group train/test counts are in
`phase0_feasibility/results/step3_split_audit.json`.

---

## 4. Medication label feasibility (exploratory, not the Phase 4 deliverable)

A deliberately minimal normalization + class-mapping table
(`phase0_feasibility/med_classes_exploratory.py`) was built by inspecting the
actual highest-frequency `DRUG` strings in the filtered cohort (not a
comprehensive dictionary) — enough to answer the feasibility question, not to
serve as the real Phase 4 implementation.

> **ASSUMPTION FLAG — "discharge medication" definition:** PRESCRIPTIONS has no
> explicit discharge flag. We defined a discharge medication as: a prescription
> row whose `STARTDATE` is on/before `DISCHTIME` **and** whose `ENDDATE` is
> null (ongoing) or on/after the discharge date. This is a common heuristic in
> MIMIC-based discharge-medication literature but is an approximation — MIMIC-III
> PRESCRIPTIONS reflects inpatient order entry, not a reconciled discharge
> medication list. Phase 4 should treat this as a starting point, not ground truth.

> **ASSUMPTION FLAG — antihypertensive exclusion is class-based, not intent-based:**
> Per the project spec, entire classes (ACE inhibitors, ARBs, beta-blockers,
> CCBs, thiazide/loop diuretics, central alpha agonists, direct vasodilators) are
> excluded from the label space regardless of why an individual drug was
> ordered (e.g. metoprolol for rate control, furosemide for volume overload).
> This is a known over-exclusion the project spec already accepts as a
> simplification.

**Results, final cohort (n=643 admissions, 10,868 discharge-eligible prescription rows):**

- 1,485 rows (13.7%) excluded as antihypertensive-class (kept for safety checks, not labels).
- 9,383 label-eligible rows remain.
- **32 distinct medication classes** in the label space.
- **3,566 rows (38.0%) fell into "Other/Unmapped"** — i.e., this exploratory
  dictionary only covers ~62% of discharge-medication volume by row count. This
  is the single biggest piece of unfinished work the feasibility check surfaces:
  **Phase 4 needs meaningfully broader drug coverage** (a real synonym dictionary
  and class table, likely informed by RxNorm/ATC or a maintained drug-class
  mapping) before the label space can be trusted end-to-end. The 32-class count
  above should be read as a lower bound — a more complete mapping will likely
  add classes, not remove them.
- 0 admissions ended up with zero label classes (every admission has at least
  one non-antihypertensive discharge medication under this mapping).
- Only 1 class ("anticholinergic", 2 admissions total) is a near-singleton in
  the full cohort; the train/test split analysis in Section 3 found 2 such
  classes once split (the split can push a borderline class under 5 on one side).

**Verdict:** the label space size (~30 usable classes) and per-class support are
appropriate for multi-label classification — not dominated by singletons — but
the underlying normalization dictionary is exploratory and must be substantially
expanded in Phase 4 before results are meaningful (a 38%-unmapped rate would
otherwise silently bias the model toward whichever high-frequency drugs happen
to already be in the dictionary).

---

## 5. Scaling to full MIMIC-III

This audit ran against a ~10,000-subject subset (~21.5% of full MIMIC-III's
~46,520 subjects). If cohort yield scales roughly linearly with subject count,
the full database would be expected to yield on the order of **~2,900-3,000
qualifying admissions** and a **proportionally larger, better-balanced train
split** — meaningfully de-risking the sparse condition-group and medication-class
tails identified above. This audit does not attempt to acquire or process full
MIMIC-III data; it is a documented expectation only, not a measured result.

---

## 6. Go/no-go recommendation

**GO**, conditional on carrying the following forward explicitly into later phases
(not re-deriving them from scratch, and not treating this audit's exploratory code
as production-ready):

1. Revalidate the Elixhauser ICD-9 mapping against a reference implementation
   before Phase 3 locks in `select_cohort()`.
2. Decide explicitly in Phase 3 whether "condition group" should stay
   all-diagnoses-based (as audited here) or switch to principal-diagnosis
   (`SEQ_NUM=1`)-based, and document the choice.
3. Collapse admission-condition groups and medication classes with <5 train
   admissions into explicit "Other" buckets in Phases 3/4/8/9, rather than
   modeling every group/class independently.
4. Treat the discharge-medication heuristic (STARTDATE/ENDDATE around DISCHTIME)
   as provisional; Phase 4 should sanity-check it against a manual chart review
   of a handful of cases if time allows.
5. Expand medication-class coverage well past the 62%-mapped exploratory
   dictionary before trusting Phase 9/14 results.
6. Keep the ~600-patient cohort size in mind when writing Phase 18's limitations
   section — this is a small-n prototype, not a validated clinical tool,
   regardless of which of the three methods "wins" in Phase 14.

---

## Appendix: reproducing this audit

```
cd phase0_feasibility
source ../.venv/bin/activate   # venv created with duckdb, pandas, pyarrow, pyyaml, tabulate
python run_audit.py                        # Section 1
python step2_timeline_audit.py             # Section 2
python step3_split_and_step4_labels.py     # Sections 3 & 4
```

All intermediate JSON/parquet results are written to `phase0_feasibility/results/`.
`results/cohort_with_split_and_labels.parquet` carries the final cohort, the
train/test split, and the exploratory medication-class labels forward for
reference — but per the notes above, it should be **regenerated from the real
Phase 3/4/6 implementations**, not consumed directly by later phases as-is.
