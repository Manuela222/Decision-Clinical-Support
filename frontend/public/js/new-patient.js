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
    resultEl.innerHTML = `<div class="card">
      <h3>Agent recommendation <span class="muted">(${escapeHtml(result.model_version || "")})</span></h3>
      ${renderMedicationTable(result.recommended_medications, null)}
      <h4>Safety warnings</h4>
      ${renderSafetyWarnings(result.safety_warnings)}
      <h4>Reasoning trace</h4>
      ${renderReasoningTrace(result.reasoning_trace)}
    </div>`;
    statusEl.innerHTML = "";
  } catch (err) {
    statusEl.innerHTML = `<div class="error-box">${escapeHtml(err.message)}</div>`;
  } finally {
    submitButton.disabled = false;
  }
});
