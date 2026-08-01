# Final Report — Discharge Medication Recommendation for Multimorbid (Hypertension + Admission Condition) Patients

**Date:** 2026-07-26 (agent live-testing findings added 2026-07-27)
**Data source:** MIMIC-III, ~10,000-subject local extract (`MIMIC_III_10k/`)
**Status:** Prototype. Not validated for clinical use.

---

## 1. Executive summary

This project built and compared three methods for recommending discharge
medication classes for patients admitted for one primary condition who also
carry chronic hypertension: a deterministic frequency baseline, a
supervised multi-label classifier (structured features + note text), and
an LLM agent with tool access and retrieval over similar past patients.
Hypertension was treated throughout as a safety constraint and conditioning
variable, never as something to be treated by the recommendation itself.

The full pipeline — cohort selection, medication normalization, timeline
and clinical-state construction, the three recommenders, a shared safety
layer, retrieval, an MCP tool server, evaluation, explainability, a FastAPI
backend, and a Node.js frontend — was built phase by phase, each phase
gated on the previous one, with 203 automated tests. Phase 0's feasibility
audit and this report's own integration run were both executed against the
real MIMIC_III_10k extract (not just synthetic fixtures); the baseline and
trained-model numbers in Section 4 are real, measured results on a
held-out, patient-level test split — not illustrative placeholders.

**Headline result:** on the real 128-admission test split, the simple
frequency baseline **outperformed** the trained model on F1 (micro F1 0.548
vs. 0.405), because the model's higher recall came at a much larger
precision cost. This is a genuine, somewhat counter-intuitive finding
discussed in Section 4 — not the result a "the fancier method wins"
narrative would predict, and worth taking seriously rather than smoothing
over.

Agent evaluation could not initially be run against a live model in this
environment (no `OPENAI_API_KEY` / network access) — its behavior was
first verified through 10 orchestration-logic tests against a scripted
mock provider and a live browser walkthrough of the actual UI (Phase 17).
Live OpenAI connectivity was established afterward. Ad hoc testing
surfaced and led to fixing two real defects invisible to the mock — an
undersized tool-call budget and, more seriously, the agent proposing
medication-class names outside the fixed vocabulary in a way that could
have silently bypassed the hypertension safety check (Section 4.2). A
subsequent full evaluation run across the real 128-admission test split
(Section 4.3) surfaced **two more** real defects the same way (a
markdown-fenced JSON answer, and an out-of-enum action crashing an entire
admission's response) — both fixed and confirmed by a second run, which
succeeded on 125/128 admissions (97.7%). With the agent's row filled in:
it ties the trained model on micro F1 but has the **best macro F1 of all
three methods** (0.202), while the baseline still wins on micro F1. A
Section 4.3 confidence-calibration test also found that, unlike baseline
and the trained model, the agent's self-reported confidence barely
distinguishes correct from incorrect recommendations — a real,
quantified caveat for anyone tempted to read its confidence score at
face value.

---

## 2. Methodology

### 2.0 Feasibility audit (Phase 0)

Before any modeling code, the cohort filters were run against the real
extract to check the project was viable at all: age ≥18 (correcting for
MIMIC-III's >89 deidentification date-shift), exactly two Elixhauser
comorbidity groups with hypertension as one of them, and ≥1 discharge
prescription. Result: **643 qualifying admissions, 622 patients**, 95.6%
with a complete core timeline. Full detail, including every assumption
flagged at the time (the hand-built Elixhauser mapping, the
all-diagnoses-vs-principal-diagnosis interpretation of "condition group,"
the discharge-medication heuristic), is in `FEASIBILITY_REPORT.md`. The
go/no-go call was **GO**, conditional on carrying those flagged assumptions
forward rather than re-deciding them silently later — which is what
happened: every phase below that touches one of them says so explicitly.

### 2.1 Shared schemas (Phase 1)

Every structured object in the system — evidence citations, timeline
events, clinical state, recommendations, safety warnings, reasoning steps,
evaluation results, UI state — is a Pydantic v2 model, JSON-serializable
end to end. This is what makes the XAI requirement (Phase 15) and the API
boundary (Phase 16) straightforward: nothing crosses a module or HTTP
boundary as an ad hoc dict.

### 2.2 Data loading (Phase 2)

A loader for the eight required MIMIC-III tables (`PATIENTS`,
`ADMISSIONS`, `DIAGNOSES_ICD`, `PRESCRIPTIONS`, `NOTEEVENTS`, `LABEVENTS`,
`D_LABITEMS`, `D_ICD_DIAGNOSES`), validating file presence and required
columns with specific error messages rather than raw pandas tracebacks.
It targets the *standard* MIMIC-III flat-CSV layout; this repo's actual
`MIMIC_III_10k/` directory uses a nonstandard per-table-folder layout left
over from earlier exploratory work, so the real integration run
(Section 4) loads via the pre-built DuckDB/parquet cache instead — a
wiring detail flagged at the time and resolved concretely for this report,
not silently worked around.

### 2.3 Cohort selection (Phase 3)

`select_cohort()` productionizes the Phase 0 filters as a single,
deterministic function against a hand-implemented Elixhauser comorbidity
mapping (Quan et al. 2005 — see Section 3) — still not cross-validated
against a reference implementation, a limitation carried forward
explicitly rather than silently assumed fixed.

**A real bug was caught here, not just a test artifact:** subtracting
`DOB` from `ADMITTIME` using pandas' default nanosecond-resolution
datetimes overflows for genuine MIMIC-III dates (real DOBs run to ~1800;
deidentification shifts admission dates past 2100+). Fixed by using
microsecond resolution throughout the date arithmetic in both the loader
and cohort selection.

### 2.4 Medication normalization (Phase 4)

`normalize_medication_name()` strips dosage/route/salt noise and applies a
synonym dictionary (external YAML/JSON, hot-swappable);
`map_to_medication_class()` maps normalized names to a ~75-class fixed
vocabulary with antihypertensive classes tagged separately so they can be
excluded from the label space but retained for safety checks. Re-running
this against the real cohort's discharge prescriptions dropped the
unmapped rate from Phase 0's exploratory 38% to **8%**, and along the way
fixed a real bug where salt-suffix stripping ran before class lookup,
silently misclassifying things like "Fentanyl Citrate."

### 2.5 Timeline & clinical state (Phase 5)

`build_patient_timeline()` assembles every dated fact for an admission
(admission, discharge, diagnoses, medication orders, labs, notes) into one
chronologically sorted, fully evidence-cited record — no summarization.
`build_clinical_state()` reduces that into the compact object every
downstream method actually consumes. Two decisions here are load-bearing
for the whole project's validity:

- **Leakage guard on `current_medications`:** restricted to orders started
  within 48h of admission, since MIMIC's `PRESCRIPTIONS` has no "home
  medication" flag and including later orders would hand the discharge-
  medication answer to the trained model and agent as an input feature.
- **Note-text leakage guard:** MIMIC discharge summaries routinely contain
  a literal "Discharge Medications:" section; note excerpts are truncated
  before that heading so Phase 9's text features can't read off the label.

A structural data gap, not a bug: blood pressure lives in MIMIC-III's
`CHARTEVENTS` table, which this project does not load (per the fixed
8-table list). Hypertension status is therefore inferred from diagnosis
codes and antihypertensive-medication presence only, never a real BP
reading — `HYPERTENSION_LABS_MISSING` is always set as a result.

### 2.6 Train/test split (Phase 6)

Seeded, deterministic, **patient-level** 80/20 split (a patient's
admissions never straddle both sides). Persisted so all three methods are
compared on the identical test set.

### 2.7 Synthetic data generator (Phase 7)

Produces fully synthetic `ClinicalState` objects — nothing derived from or
resembling any real patient row — for tests, demos, and the frontend's
"new patient" page. Negative subject/hadm IDs so they can never be
confused with real MIMIC IDs. Deliberately does *not* fabricate blood
pressure, to keep behavior consistent with the real-patient pipeline's
same structural gap.

### 2.8 Baseline (Phase 8)

`recommend_baseline()`: most-common discharge medication class(es) for the
patient's admission condition in the training split, above a frequency
threshold, falling back to a cohort-wide statistic for unseen conditions.
Deliberately leaves `evidence_ids` empty (it has no per-patient evidence)
and runs every candidate through the hypertension-compatibility check —
which is exactly what let the baseline demonstrably recommend an NSAID for
a diabetes admission in testing and get correctly flagged unsafe, an
illustrative example of why Phase 10 exists.

### 2.9 Trained model (Phase 9)

A scikit-learn `RandomForestClassifier` (natively multi-label) over
structured features (age, gender, admission condition, hypertension
status, a handful of renal-relevant labs, current-medication counts) fused
with TF-IDF over note text, via a `ColumnTransformer` pipeline. Two
choices are documented and justified in code (`cds/model/train.py`):

- **TF-IDF over a pretrained clinical embedding model** — the training set
  is small (515 real admissions), a pretrained transformer adds a large
  new dependency for a prototype meant to run offline, and TF-IDF terms
  stay interpretable for Phase 15.
- **RandomForest over XGBoost** — native multi-label support with no
  wrapper, no new dependency (already needed for TF-IDF's scikit-learn
  pipeline), and direct `feature_importances_` for Phase 15, at the cost
  of likely weaker raw performance than a tuned XGBoost model.

### 2.10 Safety checker + query allowlist (Phase 10)

`check_recommendation_safety()`: duplicate classes, hypertension-unsafe
classes (reusing each recommendation's own compatibility verdict — one
source of truth, not two), renal-sensitive classes checked against
creatinine, missing indication, missing evidence. Separately,
`sanitize_query()` for the chat panel: **exact, non-fuzzy token matching**
against a vocabulary built from ICD-9 codes/titles plus a small safe-word
list — fuzzy matching was deliberately rejected because it would let
injected text ride along on near-matches. Measured false-reject rate on 8
sample legitimate clinical queries: **9.7%**.

### 2.11 Retrieval / RAG (Phase 11)

`build_patient_profile_index()` / `find_similar_patient_profiles()`:
cosine-similarity search over **train-split-only** `ClinicalState`
embeddings, returning similar patients with their actual discharge
medications — the core RAG signal for the agent. A leakage risk flagged
explicitly in code: the four raw-record retrieval helpers
(`search_notes`, `get_recent_labs`, `get_medications`, `get_diagnoses`)
return *every* matching event regardless of date, so they must only ever
be called against a similar *train-split* patient's timeline, never the
query patient's own.

### 2.12 MCP server (Phase 12)

Exactly 5 fixed tools (`search_similar_patient_profiles`,
`check_medication_compatibility`, `lookup_lab_abnormalities`,
`check_drug_interactions`, `get_evidence_citations`), business logic kept
separate from the FastMCP transport layer. `check_drug_interactions` uses
a small curated interaction table (NSAID+ACEi/ARB, ACEi+ARB,
anticoagulant+antiplatelet/NSAID, beta blocker+CCB). Building this phase
required migrating the whole project from Python 3.9 to 3.12 (the `mcp`
SDK requires ≥3.10) — done with the user's explicit sign-off, and all 145
pre-existing tests re-verified passing before continuing.

### 2.13 Agentic recommender (Phase 13)

OpenAI-API-backed (`OPENAI_MODEL` env var, `OPENAI_API_KEY` read by the
SDK itself, never hardcoded), with a `MockLLMProvider` covering every test.
The one design decision worth restating: **`hypertension_compatible` is
never taken from the LLM's own words.** The orchestrator deterministically
calls the compatibility tool for every candidate class the model proposes
— reusing the result if the model already called it, force-running it and
logging that explicitly (`HYPERTENSION_COMPATIBILITY_CHECK`, marked "run
automatically") if it didn't. A system prompt asking nicely is not
enforcement for a safety-critical field; this makes it structural.

### 2.14 Evaluation (Phase 14)

`evaluate_medication_recommendations()` / `evaluate_all_methods()`: the
same multi-label precision/recall/F1 (micro + macro) function scores all
three methods, with an explicit alignment check that the three result
lists actually refer to the same admissions in the same order before
scoring — a silent misalignment here would make any comparison
meaningless.

### 2.15 Explainability (Phase 15)

For any existing recommendation (never generates a new one):
patient profile, retrieved similar profiles (agent), feature importances
(trained model, mapped back through the `ColumnTransformer`'s output
names), medication candidates considered-and-rejected per method, safety
warnings, evidence citations, and comparison to ground truth.

### 2.16 FastAPI backend (Phase 16)

Nine endpoints, each a thin `Depends`-injected lookup calling exactly one
service function; a `NewPatientRequest` whose medication list is validated
server-side against the fixed vocabulary (rejecting anything else, not
coercing it).

### 2.17 Node.js frontend (Phase 17)

Two pages calling the FastAPI backend only. **Actually driven in a
headless browser** (not just written) against the live backend+frontend —
this caught a real CSS bug (the Rationale column clipping long text
instead of wrapping in the narrower side-by-side cards), fixed and
re-verified before moving on.

### 2.18 Test suite

203 `pytest` tests (`backend/tests/`, one file per phase), run entirely
against fake or synthetic fixtures — no test ever touches real MIMIC-III
data or makes a live network call. Two patterns make this possible for the
otherwise-hardest-to-test parts of the system: `MockLLMProvider` (a
scripted, deterministic `LLMProvider` returning fixed `LLMTurn`s) drives
every agent-orchestration test in `test_agent.py`, decoupling the tool-call
loop, the forced hypertension-compatibility guarantee, and error handling
from any specific model's language behavior; and `build_fake_app_state()`
builds a small synthetic `AppState` for every FastAPI route test in
`test_api.py`, injected via `app.dependency_overrides`. Four tests were
added directly in response to live-testing findings rather than written
speculatively: `test_predict_agent_exceeding_tool_call_budget_returns_502`
(Section 4.2 — an unhandled `AgentError` used to bubble up as an opaque
500 with no JSON body, which browsers report as an unhelpful "Failed to
fetch"); and three more from Section 4.3's full evaluation run —
`test_agent_rejects_medication_class_outside_fixed_vocabulary`,
`test_agent_strips_markdown_code_fence_from_final_answer`, and
`test_agent_rejects_invalid_action_without_crashing_whole_admission` — each
reproducing a real failure mode the live 128-admission run surfaced
against a scripted `MockLLMProvider`, so it stays caught by `pytest` going
forward rather than only by that one script.

---

## 3. State of the art

This section situates the project's design choices against real,
published work — not a generic literature summary.

**The two dominant published approaches to MIMIC-III-based medication
recommendation are graph/deep-learning models that explicitly model
drug-drug interactions.** GAMENet (Shang et al., 2018, arXiv:1809.01852)
combines graph convolutional networks with a memory-augmented network over
patient history and a drug-drug-interaction (DDI) graph, evaluated on
MIMIC-III. SafeDrug (Yang et al., *IJCAI* 2021) goes further, encoding
drug molecular structure directly (via DrugBank) alongside patient history
to jointly optimize accuracy and DDI-rate reduction. Both report
substantial gains from encoding *why* two drugs shouldn't co-occur, not
just *whether* they historically did — a materially more sophisticated
safety mechanism than this project's curated interaction-rule table
(Section 2.12). That table is a legitimate simplification for a scoped
prototype, not a claim of parity with DDI-graph methods; a real deployment
descended from this prototype should look at incorporating structured
DDI data (e.g. DrugBank) the way SafeDrug does, rather than only a
hand-curated list.

**Clinical decision support system reviews** (Armando et al., *BMJ Health
& Care Informatics* 2023, a scoping review of CDSS for drug prescription
and therapy optimization) frame a recurring split this project's own
three-method design mirrors: knowledge-based/rule-driven systems (this
project's baseline and safety-checker layer) versus non-knowledge-based,
AI-driven systems (the trained model and agent) — with the review's
central caution being that AI-driven accuracy gains are frequently
offset by opacity, which motivates the explicit Phase 15 explainability
requirement rather than treating it as optional polish.

**The agentic, tool-using, RAG-grounded architecture (Phase 11-13) follows
a pattern established in Almanac** (Zakka et al., Research Square preprint
2023, doi:10.21203/rs.3.rs-2883198/v1, later published in *NEJM AI*):
retrieval-augmented LLMs for clinical medicine, grounding generation in
retrieved evidence specifically to counter hallucination and improve
traceability — the same motivation behind this project's evidence-cited
`EvidenceItem` objects threaded through every layer, and behind forcing
(not merely prompting for) the hypertension-compatibility check.

**Automation bias** — the specific risk this project's ethics section
(Section 5) centers on — has direct empirical grounding in prescribing
contexts: Lyell et al. (*BMC Medical Informatics and Decision Making*,
2017) studied automation bias in electronic prescribing directly, finding
that clinicians using decision support can both under-detect errors the
system fails to flag and over-detect false problems it does, i.e.
automation bias cuts in both directions, not just toward blind trust.

**Comorbidity methodology:** the Elixhauser grouping this project's cohort
filter depends on throughout follows Quan et al. (*Medical Care*, 2005),
the standard ICD-9-CM/ICD-10 comorbidity coding algorithm reference —
correctly cited in code from the start, but, as flagged repeatedly above,
never independently re-validated against a maintained implementation.

**MIMIC-III itself** is documented in Johnson et al. (*Scientific Data*,
2016, doi:10.1038/sdata.2016.35), the standard dataset citation.

None of the citations above were taken from memory without verification —
each was located and its title/authors/venue/year confirmed via live web
search and source-page fetch while writing this report, specifically to
avoid the fabricated-citation failure mode this section was explicitly
scoped to prevent.

---

## 4. Results: baseline vs. trained model vs. agent

### 4.1 Baseline vs. trained model — real numbers, real test split

Run via `backend/scripts/run_integration.py` against the actual
MIMIC_III_10k extract: cohort selection, patient-level 80/20 split
(seed=42), baseline stats and model trained on the 515-admission train
split only, both evaluated on the identical 128-admission held-out test
split.

| Method | n evaluated | Micro P | Micro R | Micro F1 | Macro P | Macro R | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 128 | 0.481 | 0.638 | **0.548** | 0.182 | 0.231 | 0.185 |
| Trained model | 128 | 0.286 | 0.692 | 0.405 | 0.129 | 0.357 | 0.167 |

(Raw output: `integration_results/baseline_vs_model_comparison.csv`,
`integration_results/model_card.json`, `integration_results/cohort_summary.json`.)

**The baseline wins on F1, both micro and macro.** The trained model has
notably higher recall (it predicts more classes per admission) but at a
large precision cost — net worse F1. This is a genuine result, not
massaged: with only 515 training admissions spread across dozens of
medication classes (66 distinct classes in the training labels), a
RandomForest has very little per-class signal to learn from, and a
frequency-based baseline is a legitimately hard target to beat at this
scale. This matches the model card's own honestly-reported known
limitation ("trained on a small cohort... metrics are not statistically
robust") — the number confirms the caveat, not just a bare disclaimer.
Scaling to full MIMIC-III (Phase 0 extrapolated ~4-5x more qualifying
admissions) is the most direct lever to test whether this ordering holds.

### 4.2 Agent — live connectivity confirmed, two real defects found and fixed via ad hoc testing

At report-writing time, no `OPENAI_API_KEY` or network access was
available, so the agent could only be verified via:

- 10 tests exercising the full orchestration loop (tool-call sequencing,
  the forced hypertension-compatibility guarantee, malformed-JSON and
  budget-overrun error handling) against a scripted `MockLLMProvider` —
  the agent's control logic, decoupled from any specific model's language
  behavior.
- The Phase 17 browser walkthrough, which drove the actual UI against a
  running backend with a scripted mock response and confirmed the full
  request/response/rendering path end to end, including the reasoning
  trace display.

Live OpenAI connectivity was established afterward (a real
`OPENAI_API_KEY`, `OpenAIProvider`, and a second FastAPI entrypoint —
`cds_api.real_server` — building `AppState` from the actual MIMIC_III_10k
extract instead of Phase 16's fake one) and exercised with real API calls.
This surfaced two real defects the mock tests structurally could not have
caught, since both depend on how an actual model behaves rather than on
the orchestration logic:

**1. Tool-call budget too low for real model behavior.** `MAX_TOOL_CALLS`
was originally 10 — enough for every scripted mock scenario, which always
converges in 1-2 turns by construction. Against the live API, a patient
with a thin evidence trail (e.g. an "AIDS/HIV" admission reason with few
similar training-set patients) occasionally exceeded the budget without
producing a final answer, surfacing to the end user as an unhandled 500
error. Root-caused with a standalone repro script that logged every tool
call the live model made, turn by turn; confirmed non-deterministic (the
same synthetic patient succeeded in most repeated runs, failed in others).
Fixed by raising the budget to 25 (`cds/agent/agent.py`) — a real model
can legitimately need more back-and-forth than a scripted mock, especially
when it calls one tool per turn instead of batching several together.

**2. Medication-class hallucination silently bypassing the hypertension
safety check — the more serious finding.** Live testing showed the agent
proposing and tool-checking medication-class strings that do not exist in
the fixed 77-class vocabulary (`cds.medications.default_dictionary()`) —
e.g. `"anticoagulants"` (plural; only `"anticoagulant"` exists), `"pain
management"` (the real classes are `"analgesic - opioid"` /
`"analgesic - non-opioid"`), `"antineoplastic agent"`, and a bare
`"antibiotic"` (only subclasses like `"antibiotic - macrolide"` exist).
This is not merely a vocabulary-hygiene issue: `check_hypertension_
compatibility` (`cds/safety/hypertension_compatibility.py`) recognizes
exactly two unsafe classes (`nsaid`, `decongestant`) by **exact string
match** and defaults every unrecognized string to `compatible = True`. A
hallucinated or malformed class name therefore silently passed the safety
check with a fabricated-sounding "no known adverse interaction" reasoning
string, regardless of whether the real class it was meant to represent
would actually have been safe. Fixed with two layers in
`cds/agent/agent.py`: the fixed vocabulary (all 77 classes) is now listed
explicitly in the system prompt with an instruction to copy names exactly,
and — the actual backstop, since prompting alone is not enforcement —
every proposed `medication_class` is validated against the known
vocabulary after the model responds, with anything not in it rejected
outright (dropped from the recommendations and logged as a `REASONING`
step explaining why) rather than surfaced with an unvalidated safety
verdict. Re-tested against the live API afterward across multiple runs:
every proposed class matched the fixed vocabulary exactly, with no
rejections triggered.

**Update:** the batch run described as a future step above has since been
performed. Section 4.3 reports it in full — four separate tests, two more
real bugs it surfaced and their fixes, and the agent's completed row in
the three-way comparison.

---

### 4.3 Full agent evaluation: four tests against the real 128-admission test split

Run via `backend/scripts/run_agent_evaluation.py` (real `OPENAI_API_KEY`,
`gpt-4o-mini`, the same real MIMIC_III_10k cohort and 128-admission test
split as Section 4.1 — built with the identical `build_real_app_state()`
used by `cds_api.real_server`). Four tests, run in sequence, each feeding
the next:

**Test 1 — smoke test (8 admissions).** A small pre-flight batch, run
before committing to the full test split's API cost, specifically to fail
fast if something was structurally broken. Result: 8/8 succeeded. This
gated whether the full run below was worth running at all.

**Tests 2-4 — full run, 128 admissions, in one pass.** The script runs
baseline, trained model, and agent over every test-set admission in one
sweep, so Test 2 (accuracy), Test 3 (safety audit), and Test 4 (confidence
calibration) all come from the same run rather than three separate,
possibly-inconsistent samples. Total wall-clock time for the agent portion
across 128 admissions: ~22 minutes (dominated by sequential API round
trips, not computation) — inexpensive at `gpt-4o-mini` pricing, but not
instant, which is itself a real deployment-relevant data point.

#### Two more real bugs found by this run, fixed, and confirmed by a second run

The **first** full run succeeded on 116/128 admissions (12 failures,
9.4%). Breaking down those 12 failures surfaced two defects that had never
appeared in any prior mocked test or ad hoc single-patient live test:

- **9 of 12**: `max_tool_calls=25` exceeded — consistent with Section
  4.2's finding, now with a real base rate attached (~7% of admissions).
- **2 of 12**: `AgentError: Agent's final answer was not valid JSON`. The
  model had wrapped its JSON answer in a ```` ```json ... ``` ```` markdown
  code fence — despite the system prompt's explicit "respond with ONLY a
  JSON object, no other text" — which `json.loads` cannot parse as-is.
- **1 of 12**: `ValueError: 'consider' is not a valid MedicationAction`.
  The model proposed an action outside the fixed
  `start|continue|stop|adjust|avoid` enum the system prompt lists, and the
  code called `MedicationAction(raw.get("action"))` directly with no
  validation — an unhandled `ValueError` that crashed the *entire*
  admission's response, discarding every other, valid recommendation in
  the same answer.

Both are now fixed in `cds/agent/agent.py`, following the same principle
Section 4.2 already established for hallucinated medication classes —
validate the model's output before trusting it, and fail one recommendation
rather than one entire admission:

1. `_parse_final_answer` now strips a leading/trailing markdown code fence
   before calling `json.loads`.
2. Parsing `action` is now wrapped in a `try/except ValueError`; an invalid
   action rejects *that one* candidate recommendation (logged as a
   `REASONING` step, same pattern as the vocabulary rejection) instead of
   raising and losing the whole admission's answer.

Three new tests were added (`test_agent.py`, 203 tests total) reproducing
each failure mode against `MockLLMProvider` so they stay caught by
`pytest` going forward, not just by this one script.

The **second** full run, with both fixes applied, succeeded on **125/128
admissions (97.7%)** — all three remaining failures are budget-exceeded,
**zero** malformed-JSON or invalid-action failures. All results below are
from this second, post-fix run.

#### Test 2: the agent's completed row in the three-way comparison

Scored over the 125 admissions where all three methods have a result
(the 3 that still exceeded the tool-call budget are excluded from *all
three* columns here, not just the agent's, so the comparison stays
apples-to-apples — see `test2_full_comparison.csv`):

| Method | n evaluated | Micro P | Micro R | Micro F1 | Macro P | Macro R | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 125 | 0.478 | 0.638 | **0.546** | 0.181 | 0.231 | 0.185 |
| Trained model | 125 | 0.284 | 0.692 | 0.403 | 0.128 | 0.357 | 0.166 |
| Agent | 125 | 0.432 | 0.374 | 0.401 | **0.270** | 0.219 | **0.202** |

**Conclusion:** the baseline still wins on micro F1, matching Section
4.1's finding on this slightly reduced (125 vs. 128) admission set — this
was not an artifact of the agent being absent. The agent essentially ties
the trained model on micro F1 (0.401 vs. 0.403) but gets there completely
differently: far lower recall (0.374, the most conservative of the three
— it recommends fewer classes per admission) but the **best macro
precision and macro F1 of all three methods** (0.270 / 0.202). Read
together with the rare-class discussion in Section 6.4, this is a
genuinely interesting, non-obvious result: the agent is comparatively
*better* at the classes with little training signal — plausible, since it
reasons per-patient from retrieved similar cases rather than reproducing
whatever was statistically most common — while baseline and the trained
model both lean on frequency and therefore dominate the common classes
(hence their much higher recall). No method wins outright; each wins on a
different axis, which is a more honest and more useful takeaway than a
single ranked list.

#### Test 3: agent safety and consistency audit (same run)

- **Hypertension-compatibility check coverage: 100%** (125/125 admissions)
  — every single recommended medication class had a forced or
  agent-initiated compatibility check in its reasoning trace, no
  exceptions. The structural guarantee described in Section 2.13 held at
  scale, not just in the hand-picked cases tested earlier.
- **Tool calls per admission:** min 4, max 25 (the budget ceiling), mean
  10.1. **Hypertension checks per admission:** min 2, max 20, mean 7.7.
- **Vocabulary-hallucination backstop: triggered 8 times across the 128
  admissions (~6%), caught 8/8.** Real rejected classes included
  `digoxin`, `opioid` (3 separate admissions), `antidepresant` (a
  misspelling of a real class), `antinauseant` (2 admissions), and
  `theophylline` — plausible-sounding clinical terms, none of them exact
  matches to the fixed 77-class vocabulary. Every one was rejected before
  reaching the output, per `test3_safety_audit.json`; zero leaked through
  with a fabricated "compatible" verdict.
- **Residual budget failures: 3/128 (2.3%).** These fail *loud*: `AgentError`
  → the route's `except AgentError` handler (Section 2.18) → a clean 502,
  not a silently wrong or missing answer. The system's behavior when it
  cannot safely answer is to say so, not guess.

**Conclusion:** the two-layer defense from Section 4.2 (constrained prompt
+ hard backend validation) is not just a single-case fix — at real scale,
across 128 real admissions, it caught every hallucination attempt with no
exceptions. The remaining failure mode (budget) is a availability/UX
problem (the admission gets no answer), not a safety problem (it never
produces a *wrong* answer silently).

#### Test 4: confidence calibration — does "confidence" actually track correctness?

For each method, mean self-reported `confidence` split by whether the
recommendation was actually correct (matched the true discharge
medications) or not (extra) — see `test4_confidence_calibration.json`:

| Method | Mean confidence — matched | Mean confidence — extra | Gap |
|---|---:|---:|---:|
| Baseline | 0.596 | 0.489 | 0.107 |
| Trained model | 0.628 | 0.488 | 0.140 |
| Agent | 0.949 | 0.938 | 0.011 |

**Conclusion:** this is an empirical, quantified confirmation of Section
5.1's conceptual claim, not just a caveat. Baseline and trained-model
confidence both carry real discriminative signal — a correct
recommendation scores meaningfully higher, on average, than an incorrect
one (gaps of 0.107 and 0.140). The agent's self-reported confidence
carries almost none: 0.949 vs. 0.938 is barely distinguishable, and both
numbers sit near the ceiling regardless of whether the recommendation was
actually right. A clinician who reads the agent's "94% confident" as
comparable to the trained model's "94% confident" would be making a
real error — the two numbers are not measuring the same thing, and only
one of them tracks whether the system was actually correct.

---

## 5. Confidence scores, the interface, and how to tell which method is "best"

### 5.1 What "confidence" means — it is not the same number across methods

`MedicationRecommendation.confidence` is a single 0.0-1.0 field on every
recommendation, but each method fills it with a fundamentally different
kind of number, and treating them as comparable would be a mistake:

- **Baseline** (`cds/baseline/recommend.py`): confidence is the
  *training-set frequency* of that medication class among discharges for
  this patient's admission reason — e.g. `confidence = 1.0` for
  `antidiabetic - biguanide` means it appeared in 100% of training-set
  "Diabetes, uncomplicated" discharges, not that the system is "100%
  sure." It is a population statistic computed once from
  `DiagnosisMedicationStats`, identical for every patient sharing an
  admission reason.
- **Trained model** (`cds/model/predict.py`): confidence is the
  RandomForest's `predict_proba` output for the positive class of that
  one-vs-rest label — a genuine per-patient probability estimate
  conditioned on this patient's structured features and note text, but
  only as calibrated as a random forest trained on 515 admissions can be
  (the small-cohort caveat in Section 5.4/6.4 applies directly to this
  number).
- **Agent** (`cds/agent/agent.py`): confidence is **self-reported by the
  LLM** in its final JSON answer, per the system prompt's
  `"confidence": 0.0-1.0` instruction. It is not computed, calibrated, or
  cross-checked by any backend code — it is the model's own stated
  certainty in natural-language-generation terms, a fundamentally
  different (and much less trustworthy) kind of number than the other two.
  This is exactly the automation-bias risk flagged in Section 6.2: a
  clinician seeing "confidence: 0.9" from the agent has no guarantee it
  reflects anything beyond the model's tendency to sound confident.

### 5.2 The "Existing Patient" page

(`frontend/public/existing-patient.html` + `existing-patient.js`) walks
through a held-out **test-split** admission (never train-split, so no
method has seen it during training/indexing):

1. `GET /cohort` populates the dropdown with every test-set admission
   (subject_id, hadm_id, age, gender, admission reason, hypertension
   status) from the currently loaded `AppState`.
2. Selecting a patient calls
   `GET /explain/{subject_id}/{hadm_id}?method=baseline` to fetch that
   admission's **true discharge medications** (the ground truth used for
   evaluation, Sections 2.4/4.1) — shown in its own card so it is never
   confused with a prediction.
3. "Run baseline / model / agent" fires all three `POST /predict/*`
   endpoints in parallel (`Promise.allSettled`), so one method failing
   (e.g. the agent without an API key) does not block the other two from
   rendering.
4. Each method's card renders its recommended classes with a pill showing
   whether that class **matched** the true discharge medications, was
   **extra** (predicted but not actually prescribed), or — listed
   separately below the table — **missed** (actually prescribed but not
   predicted), plus that method's safety warnings and reasoning trace (the
   agent's is populated; baseline/model's explicitly says only the agent
   produces one).

This page is Section 4.1's comparison table made interactive one admission
at a time, with the actual evidence visible instead of only an aggregate
score.

### 5.3 The "New Patient" page

(`frontend/public/new-patient.html` + `new-patient.js`) runs the **agent
only** — baseline and the trained model need an admission already present
in `AppState`; this page's entire purpose is a patient that isn't. Every
field is a `<select>` populated from a fixed vocabulary
(`frontend/public/js/vocabulary.js`) — no free text is accepted for
admission reason, hypertension status, or medications, matching the
project's global "select-only medication input" requirement. "Auto-fill
with synthetic example" calls `GET /synthetic/new-patient`
(`cds.synthetic.generate_synthetic_clinical_state`, clearly disclaimed as
fabricated data, never a real patient) purely to save typing. Submitting
calls `POST /predict/agent/new-patient`, which re-validates every
submitted medication class server-side against `default_dictionary()`
(rejecting unknown ones with a 400, regardless of what the frontend's
static reference list contains) before building a `ClinicalState` with
`subject_id = hadm_id = -1` and running the identical `recommend_agent()`
used everywhere else in the system.

### 5.4 How to tell which method is "best"

One function answers this, and it is the same one for all three methods:
`compute_multilabel_metrics()` (Section 2.14), which scores predicted vs.
true discharge medication-class *sets* per admission
(precision/recall/F1) and aggregates two ways:

- **Micro-averaged**: pool every true-positive/false-positive/false-negative
  across all admissions first, then compute precision/recall/F1 once —
  dominated by common classes, answers "how good is this method on a
  typical admission."
- **Macro-averaged**: compute precision/recall/F1 per medication class,
  then average across classes unweighted — answers "how good is this
  method across the *full range* of classes," and is dragged down hard by
  rare classes with little training signal (Sections 4.1, 6.4).

`evaluate_all_methods()` runs all three methods over the *identical*
test-set admissions in the *identical* order (with an explicit alignment
check, Section 2.14) and returns one comparison table, exposed via
`POST /evaluate`. **This is exactly how Section 4.1's numbers were
produced** — not a separate ad hoc calculation — and it is what "the
baseline wins on F1" concretely means: run the same 128 held-out
admissions through all three methods, score every one against its actual
discharge medications with the same metric, and compare the aggregate.
There is no "best" method independent of this evaluation — a method that
looks better on a handful of manually inspected cases in the "Existing
Patient" page (Section 5.2) is not evidence of anything until it is run
through this same evaluation at scale.

---

## 6. Ethics

### 6.1 Data privacy: synthetic vs. real MIMIC-III data

Real MIMIC-III data (via the local extract) was used only for: the Phase 0
audit, building/training the cohort/model/index in this report's
integration run, and the automated test suite's *own* infrastructure never
touches it (every pytest test uses fake or synthetic fixtures — this was
a deliberate, enforced separation throughout, not incidental). Synthetic
data (Phase 7) is used for the frontend's "new patient" demo and clearly
disclaimed in the README, with negative sentinel IDs specifically so
synthetic and real patients can never be confused downstream. No real
patient identifier, name, or free-text note content appears in this
report. The project never attempted to re-identify or cross-reference
MIMIC-III subjects against any external source.

### 6.2 Automation bias

This is the project's central ethical risk, not a boilerplate mention.
Per Lyell et al. (2017, Section 3), automation bias in prescribing
contexts is empirically real and bidirectional: clinicians using decision
support can both miss errors the system fails to flag *and* over-flag
things it incorrectly raises. Three design choices in this project push
against that risk, and their limits should be stated plainly:

- Every recommendation carries an explicit, per-class hypertension
  reasoning string and safety warnings — but a clinician skimming a UI
  under time pressure may not read past the recommended action itself.
- The agent's hypertension-compatibility check is *forced*, not merely
  requested — but this protects one specific, narrow safety property; it
  says nothing about the correctness of the *indication* itself (whether
  the medication class is actually right for this patient's admission
  condition), which remains entirely un-verified by any deterministic
  check in this system.
- The baseline explicitly has no per-patient evidence
  (`evidence_ids: []`, flagged by the safety checker itself) — but a
  clinician who doesn't understand *why* the baseline lacks evidence
  could mistake its high-confidence-looking percentage score for clinical
  confidence rather than population frequency.

None of this system's outputs should be presented to a clinician without
the full reasoning trace, safety warnings, and (for the baseline/model)
the explicit absence of per-patient evidence, visible alongside the
recommendation — never the recommendation alone.

### 6.3 Safety-critical failure modes

- **Silent leakage regression:** the two leakage guards (Section 2.5) are
  the single most important correctness property in this codebase. If any
  future phase re-derives "current medications" from raw `PRESCRIPTIONS`
  instead of consuming the already-built `ClinicalState`, or passes a
  query patient's own timeline into the raw retrieval helpers (Section
  2.11's flagged risk), the model/agent would be silently handed the
  answer — inflating apparent accuracy without anyone noticing, since the
  metrics would look *better*, not obviously broken.
- **Renal/hypertension rule gaps:** the safety checker's renal-sensitivity
  and hypertension-compatibility rules are hand-curated lists (Sections
  2.10, 2.12), not derived from a maintained drug-interaction database —
  they will miss real interactions and unsafe combinations not on the
  list, with no fallback signal that something was missed.
- **No real blood pressure data:** hypertension status is inferred, never
  measured (Section 2.5) — a patient with well-documented diagnosis-coded
  hypertension but currently normotensive, or vice versa, cannot be
  distinguished by this system at all.
- **Agent non-determinism:** unlike baseline and trained model (fully
  deterministic given the same inputs), the agent's recommendations can
  vary run-to-run against a live LLM — a property this report's tests
  control for via the mock provider, but that a real deployment must
  handle explicitly (e.g. via confidence thresholds, repeated-run
  agreement checks, or requiring the tool-verified compatibility check to
  gate display regardless of the model's phrasing).
- **Vocabulary hallucination bypassing a safety check (realized, not just
  hypothetical):** live testing (Section 4.2) found the agent proposing
  medication-class names outside the fixed vocabulary, which the
  hypertension-compatibility check — keyed on exact string match against a
  small unsafe-class list — defaulted to "compatible" rather than flagging
  as unrecognized. Fixed with vocabulary-constrained prompting plus a hard
  backend validation backstop, but it is the clearest concrete illustration
  in this project of why an LLM's free-text output can never be trusted as
  the sole gate on a safety-relevant field — a lesson the design in Section
  2.13 already applied to `hypertension_compatible` itself, and one this
  incident shows extends to *which class is even being checked*. Section
  4.3's Test 3 confirms this held at scale, not just in the one case that
  found it: across the full 128-admission test split, the backstop
  triggered 8 more times and caught every one, with zero leaks.

### 6.4 Limitations of the evaluation given cohort size

The entire evaluation in Section 4 rests on **128 test-set admissions**
drawn from a **643-admission, 622-patient cohort** — itself drawn from a
~10,000-subject MIMIC-III *subset*, not the full ~46,500-subject database.
Concretely:

- Macro-averaged metrics (0.185 baseline F1, 0.167 model F1) are much
  lower than micro-averaged ones, meaning performance is uneven across the
  66-90 medication classes in play — some classes have single-digit
  support in training, which the Phase 0 audit already flagged and which
  this report's real numbers confirm is still true after Phase 4's
  dictionary expansion.
- A single 80/20 split (one seed) was evaluated — no cross-validation or
  repeated-split variance estimate exists, so the gap between baseline and
  model F1 (0.548 vs. 0.405) has no confidence interval attached; it could
  narrow or reverse under a different split.
- The result in Section 4.1 (baseline beating the trained model) should
  not be read as "simple methods are always better" — it should be read
  as "this specific model, trained on this specific small cohort, did not
  clear the bar this specific baseline set," which is a narrower and more
  defensible claim.
- The agent's Section 4.3 numbers rest on an even smaller effective sample
  (125 admissions, after excluding the 3 that exceeded the tool-call
  budget) than baseline/trained model's 128 — the same small-sample
  caveats above apply to it at least as strongly, and its 2.3% no-answer
  rate is itself a sample-size-sensitive number worth re-checking at
  scale.

This is a prototype demonstrating a complete, safety-conscious pipeline
architecture — not a validated clinical tool, and not a benchmark result
that should inform real prescribing decisions at any scale evaluated here.

---

## 7. Known gaps and recommended next steps

1. ~~Run the agent across the full 128-admission test split~~ — done
   (Section 4.3). Remaining follow-up: re-attempt the 3 admissions that
   still exceeded the tool-call budget (possibly with a higher budget or a
   retry-with-fresh-context strategy) to get true 128/128 coverage instead
   of 125/128; and re-run periodically as a regression check, since
   `gpt-4o-mini`'s behavior is not guaranteed stable across OpenAI model
   updates the way the deterministic baseline/trained model are.
2. Revalidate the Elixhauser mapping (`cds/cohort/elixhauser.py`) against
   a maintained reference implementation (e.g. the R `comorbidity`
   package) before treating cohort membership as ground truth.
3. Expand the medication dictionary further (still ~8% unmapped on real
   discharge prescriptions) and the drug-interaction rule table, ideally
   against a maintained source (RxNorm/DrugBank) rather than hand-curation.
4. Re-run Phase 0's audit and Section 4's integration run against full
   MIMIC-III once available, to test whether the baseline's F1 advantage
   over the trained model holds at ~4-5x the training data.
5. Wire Phase 2's standard-CSV loader against a real production MIMIC-III
   deployment layout (this report's integration run used the DuckDB/
   parquet cache as a documented, deliberate substitute — see Section 2.2).
