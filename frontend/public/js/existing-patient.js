const patientSelect = document.getElementById("patient-select");
const runButton = document.getElementById("run-button");
const patientSummary = document.getElementById("patient-summary");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");

let cohort = [];

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
});

function renderMethodCard(title, result, groundTruthSet, errorMessage) {
  if (errorMessage) {
    return `<div class="card"><h3>${title}</h3><div class="error-box">${escapeHtml(errorMessage)}</div></div>`;
  }
  return `<div class="card">
    <h3>${title} <span class="muted">(${escapeHtml(result.model_version || "")})</span></h3>
    ${renderMedicationTable(result.recommended_medications, groundTruthSet)}
    <h4>Safety warnings</h4>
    ${renderSafetyWarnings(result.safety_warnings)}
    <h4>Reasoning trace</h4>
    ${renderReasoningTrace(result.reasoning_trace)}
  </div>`;
}

runButton.addEventListener("click", async () => {
  const { subjectId, hadmId } = currentSelection();
  runButton.disabled = true;
  statusEl.innerHTML = '<p class="muted">Running baseline, model, and agent…</p>';
  resultsEl.innerHTML = "";

  let groundTruthSet = new Set();
  try {
    const explainBaseline = await api.explain(subjectId, hadmId, "baseline");
    groundTruthSet = new Set(explainBaseline.ground_truth_medications);
  } catch (err) {
    // non-fatal -- still show predictions without the ground-truth annotation
  }

  const [baseline, model, agent] = await Promise.allSettled([
    api.predictBaseline(subjectId, hadmId),
    api.predictModel(subjectId, hadmId),
    api.predictAgent(subjectId, hadmId),
  ]);

  const trueMedsHtml = groundTruthSet.size
    ? `<div class="card"><h3>True discharge medications</h3><p>${[...groundTruthSet].map((c) => `<span class="pill match">${escapeHtml(c)}</span>`).join(" ")}</p></div>`
    : "";

  resultsEl.innerHTML =
    trueMedsHtml +
    renderMethodCard("Baseline", baseline.value, groundTruthSet, baseline.status === "rejected" ? baseline.reason.message : null) +
    renderMethodCard("Trained model", model.value, groundTruthSet, model.status === "rejected" ? model.reason.message : null) +
    renderMethodCard("Agent", agent.value, groundTruthSet, agent.status === "rejected" ? agent.reason.message : null);

  statusEl.innerHTML = "";
  runButton.disabled = false;
});

init();
