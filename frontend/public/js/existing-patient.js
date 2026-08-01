const patientSelect = document.getElementById("patient-select");
const runButton = document.getElementById("run-button");
const patientSummary = document.getElementById("patient-summary");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const comparisonSummaryEl = document.getElementById("comparison-summary");
const resultsToolbarEl = document.getElementById("results-toolbar");

const METHOD_LABELS = { baseline: "Baseline", model: "Trained model", agent: "Agent" };
const METHOD_ORDER = ["baseline", "model", "agent"];

let cohort = [];
let lastResults = {}; // key -> RecommendationResult, only successful ones
let lastGroundTruthSet = new Set();

async function init() {
  try {
    cohort = await api.cohort();
  } catch (err) {
    patientSelect.innerHTML = `<option>Failed to load cohort</option>`;
    statusEl.innerHTML = `<div class="error-box">${escapeHtml(err.message)}</div>`;
    return;
  }
  patientSelect.innerHTML = cohort
    .map((p) => `<option value="${p.subject_id}:${p.hadm_id}">#${p.subject_id}/${p.hadm_id} — ${escapeHtml(p.admission_reason)} (${p.age}${p.gender})</option>`)
    .join("");
  runButton.disabled = cohort.length === 0;
  if (cohort.length > 0) showSummary(cohort[0]);
}

function currentSelection() {
  const [subjectId, hadmId] = patientSelect.value.split(":").map(Number);
  return { subjectId, hadmId };
}

function showSummary(entry) {
  patientSummary.innerHTML = `<p class="muted">Admission reason: <strong>${escapeHtml(entry.admission_reason)}</strong> · Hypertension: <strong>${escapeHtml(entry.hypertension_status)}</strong></p>`;
}

patientSelect.addEventListener("change", () => {
  const { subjectId, hadmId } = currentSelection();
  const entry = cohort.find((p) => p.subject_id === subjectId && p.hadm_id === hadmId);
  if (entry) showSummary(entry);
  resultsEl.innerHTML = "";
  comparisonSummaryEl.innerHTML = "";
  resultsToolbarEl.innerHTML = "";
  lastResults = {};
});

function methodCardBodyHtml(result, groundTruthSet) {
  return `${renderMedicationTable(result.recommended_medications, groundTruthSet)}
    <h4>Safety warnings</h4>
    ${renderSafetyWarnings(result.safety_warnings)}
    <h4>Reasoning trace</h4>
    ${renderReasoningTrace(result.reasoning_trace)}`;
}

function renderMethodCard(key, title, result, groundTruthSet, errorMessage) {
  if (errorMessage) {
    return `<div class="card"><h3>${title}</h3><div class="error-box">${escapeHtml(errorMessage)}</div></div>`;
  }
  return `<div class="card">
    <div class="card-title-row">
      <h3>${title} <span class="muted">(${escapeHtml(result.model_version || "")})</span></h3>
      <button type="button" class="icon-button card-expand-btn" data-expand="${key}" aria-label="Expand ${title}" title="Fullscreen">&#9974;</button>
    </div>
    ${methodCardBodyHtml(result, groundTruthSet)}
  </div>`;
}

function openFocusForKey(key) {
  const items = METHOD_ORDER.filter((k) => lastResults[k]).map((k) => ({
    title: `${METHOD_LABELS[k]} recommendation`,
    bodyHtml: methodCardBodyHtml(lastResults[k], lastGroundTruthSet),
  }));
  const startIndex = METHOD_ORDER.filter((k) => lastResults[k]).indexOf(key);
  FocusModal.open(items, Math.max(startIndex, 0));
}

function renderResultsToolbar() {
  const keys = Object.keys(lastResults);
  if (keys.length === 0) {
    resultsToolbarEl.innerHTML = "";
    return;
  }
  resultsToolbarEl.innerHTML = `
    <button type="button" id="copy-results-btn" class="secondary">&#128203; Copy summary</button>
    <button type="button" id="export-pdf-btn" class="secondary">&#128196; Export / Print</button>
    <span id="copy-flash" class="copy-flash"></span>`;

  document.getElementById("copy-results-btn").addEventListener("click", async () => {
    const text = METHOD_ORDER.filter((k) => lastResults[k])
      .map((k) => resultToPlainText(METHOD_LABELS[k], lastResults[k]))
      .join("\n\n");
    try {
      await navigator.clipboard.writeText(text);
      const flash = document.getElementById("copy-flash");
      flash.textContent = "Copied to clipboard.";
      setTimeout(() => (flash.textContent = ""), 2500);
    } catch (err) {
      document.getElementById("copy-flash").textContent = "Copy failed — select and copy manually.";
    }
  });

  document.getElementById("export-pdf-btn").addEventListener("click", () => window.print());
}

runButton.addEventListener("click", async () => {
  const { subjectId, hadmId } = currentSelection();
  runButton.disabled = true;
  statusEl.innerHTML = '<p class="muted">Running baseline, model, and agent…</p>';
  resultsEl.innerHTML = "";
  comparisonSummaryEl.innerHTML = "";
  resultsToolbarEl.innerHTML = "";
  lastResults = {};

  let groundTruthSet = new Set();
  try {
    const explainBaseline = await api.explain(subjectId, hadmId, "baseline");
    groundTruthSet = new Set(explainBaseline.ground_truth_medications);
  } catch (err) {
    // non-fatal -- still show predictions without the ground-truth annotation
  }
  lastGroundTruthSet = groundTruthSet;

  const [baseline, model, agent] = await Promise.allSettled([
    api.predictBaseline(subjectId, hadmId),
    api.predictModel(subjectId, hadmId),
    api.predictAgent(subjectId, hadmId),
  ]);

  lastResults = {};
  if (baseline.status === "fulfilled") lastResults.baseline = baseline.value;
  if (model.status === "fulfilled") lastResults.model = model.value;
  if (agent.status === "fulfilled") lastResults.agent = agent.value;

  const trueMedsHtml = groundTruthSet.size
    ? `<div class="card"><h3>True discharge medications</h3><p>${[...groundTruthSet].map((c) => `<span class="pill match">${escapeHtml(c)}</span>`).join(" ")}</p></div>`
    : "";

  resultsEl.innerHTML =
    trueMedsHtml +
    renderMethodCard("baseline", "Baseline", baseline.value, groundTruthSet, baseline.status === "rejected" ? baseline.reason.message : null) +
    renderMethodCard("model", "Trained model", model.value, groundTruthSet, model.status === "rejected" ? model.reason.message : null) +
    renderMethodCard("agent", "Agent", agent.value, groundTruthSet, agent.status === "rejected" ? agent.reason.message : null);

  resultsEl.querySelectorAll("[data-expand]").forEach((btn) => {
    btn.addEventListener("click", () => openFocusForKey(btn.dataset.expand));
  });

  comparisonSummaryEl.innerHTML = "";
  comparisonSummaryEl.appendChild(buildComparisonBanner(lastResults, METHOD_LABELS, openFocusForKey));

  renderResultsToolbar();

  statusEl.innerHTML = "";
  runButton.disabled = false;
});

init();
