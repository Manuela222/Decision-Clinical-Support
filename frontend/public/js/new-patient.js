const form = document.getElementById("new-patient-form");
const ageInput = document.getElementById("age");
const genderSelect = document.getElementById("gender");
const admissionReasonSelect = document.getElementById("admission-reason");
const hypertensionStatusSelect = document.getElementById("hypertension-status");
const creatinineInput = document.getElementById("creatinine");
const medicationsSelect = document.getElementById("current-medications");
const autofillButton = document.getElementById("autofill-button");
const submitButton = document.getElementById("submit-button");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");

admissionReasonSelect.innerHTML = ADMISSION_REASONS.map((r) => `<option value="${escapeHtml(r)}">${escapeHtml(r)}</option>`).join("");
hypertensionStatusSelect.innerHTML = HYPERTENSION_STATUSES.map((s) => `<option value="${s.value}">${escapeHtml(s.label)}</option>`).join("");
medicationsSelect.innerHTML = MEDICATION_CLASSES.map((m) => `<option value="${escapeHtml(m)}">${escapeHtml(m)}</option>`).join("");

function setSelectedOptions(selectEl, values) {
  const valueSet = new Set(values || []);
  [...selectEl.options].forEach((opt) => {
    opt.selected = valueSet.has(opt.value);
  });
}

autofillButton.addEventListener("click", async () => {
  statusEl.innerHTML = '<p class="muted">Fetching synthetic example…</p>';
  try {
    const example = await api.syntheticNewPatient();
    ageInput.value = example.age;
    genderSelect.value = example.gender;
    admissionReasonSelect.value = example.admission_reason;
    hypertensionStatusSelect.value = example.hypertension_status;
    creatinineInput.value = example.recent_creatinine ?? "";
    setSelectedOptions(medicationsSelect, example.current_medication_classes);
    statusEl.innerHTML = '<p class="muted">Auto-filled with a synthetic example (not a real patient).</p>';
  } catch (err) {
    statusEl.innerHTML = `<div class="error-box">${escapeHtml(err.message)}</div>`;
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submitButton.disabled = true;
  statusEl.innerHTML = '<p class="muted">Running the agent…</p>';
  resultEl.innerHTML = "";

  const payload = {
    age: Number(ageInput.value),
    gender: genderSelect.value,
    admission_reason: admissionReasonSelect.value,
    hypertension_status: hypertensionStatusSelect.value,
    current_medication_classes: [...medicationsSelect.selectedOptions].map((o) => o.value),
    recent_creatinine: creatinineInput.value ? Number(creatinineInput.value) : null,
  };

  try {
    const result = await api.predictAgentNewPatient(payload);
    const bodyHtml = `${renderMedicationTable(result.recommended_medications, null)}
      <h4>Safety warnings</h4>
      ${renderSafetyWarnings(result.safety_warnings)}
      <h4>Reasoning trace</h4>
      ${renderReasoningTrace(result.reasoning_trace)}`;
    resultEl.innerHTML = `<div class="results-toolbar no-print">
        <button type="button" id="copy-results-btn" class="secondary">&#128203; Copy summary</button>
        <button type="button" id="export-pdf-btn" class="secondary">&#128196; Export / Print</button>
        <span id="copy-flash" class="copy-flash"></span>
      </div>
      <div class="card">
        <div class="card-title-row">
          <h3>Agent recommendation <span class="muted">(${escapeHtml(result.model_version || "")})</span></h3>
          <button type="button" class="icon-button card-expand-btn" data-expand="agent" aria-label="Expand" title="Fullscreen">&#9974;</button>
        </div>
        ${bodyHtml}
      </div>`;

    resultEl.querySelector("[data-expand]").addEventListener("click", () => {
      FocusModal.open([{ title: "Agent recommendation", bodyHtml }], 0);
    });
    document.getElementById("copy-results-btn").addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(resultToPlainText("Agent recommendation", result));
        const flash = document.getElementById("copy-flash");
        flash.textContent = "Copied to clipboard.";
        setTimeout(() => (flash.textContent = ""), 2500);
      } catch (err) {
        document.getElementById("copy-flash").textContent = "Copy failed — select and copy manually.";
      }
    });
    document.getElementById("export-pdf-btn").addEventListener("click", () => window.print());

    statusEl.innerHTML = "";
  } catch (err) {
    statusEl.innerHTML = `<div class="error-box">${escapeHtml(err.message)}</div>`;
  } finally {
    submitButton.disabled = false;
  }
});
