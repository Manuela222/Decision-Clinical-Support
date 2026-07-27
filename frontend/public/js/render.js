// Shared rendering helpers used by both pages.

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[ch]);
}

function renderMedicationTable(recommendedMedications, groundTruthSet) {
  if (!recommendedMedications || recommendedMedications.length === 0) {
    return '<p class="muted">No medication classes recommended.</p>';
  }
  const rows = recommendedMedications
    .map((rec) => {
      let pill = "";
      if (groundTruthSet) {
        pill = groundTruthSet.has(rec.medication_class)
          ? '<span class="pill match">matched</span>'
          : '<span class="pill extra">extra</span>';
      }
      const htnClass = rec.hypertension_compatible ? "htn-ok" : "htn-bad";
      const htnLabel = rec.hypertension_compatible ? "Compatible" : "NOT compatible";
      return `<tr>
        <td data-label="Class">${escapeHtml(rec.medication_class)} ${pill}</td>
        <td data-label="Action">${escapeHtml(rec.action)}</td>
        <td data-label="Confidence">${(rec.confidence * 100).toFixed(0)}%</td>
        <td data-label="Hypertension" class="${htnClass}" title="${escapeHtml(rec.hypertension_reasoning)}">${htnLabel}</td>
        <td data-label="Rationale">${escapeHtml(rec.rationale)}</td>
      </tr>`;
    })
    .join("");

  let missedRow = "";
  if (groundTruthSet) {
    const predictedSet = new Set(recommendedMedications.map((r) => r.medication_class));
    const missed = [...groundTruthSet].filter((c) => !predictedSet.has(c));
    if (missed.length > 0) {
      missedRow = `<p class="muted">Missed (in true discharge meds, not recommended): ${missed
        .map((c) => `<span class="pill missed">${escapeHtml(c)}</span>`)
        .join(" ")}</p>`;
    }
  }

  return `<div class="table-scroll"><table>
    <thead><tr><th>Class</th><th>Action</th><th>Confidence</th><th>Hypertension</th><th>Rationale</th></tr></thead>
    <tbody>${rows}</tbody>
  </table></div>${missedRow}`;
}

function renderSafetyWarnings(warnings) {
  if (!warnings || warnings.length === 0) {
    return '<p class="muted">No safety warnings.</p>';
  }
  return warnings
    .map((w) => {
      const cls = w.severity === "critical" ? "critical" : w.severity === "warning" ? "warning-level" : "info";
      return `<div class="warning ${cls}"><strong>${escapeHtml(w.severity.toUpperCase())}</strong> — ${escapeHtml(w.category)}: ${escapeHtml(w.message)}</div>`;
    })
    .join("");
}

function renderReasoningTrace(steps) {
  if (!steps || steps.length === 0) {
    return '<p class="muted">No reasoning trace (only the agent method produces one).</p>';
  }
  return steps
    .map((step) => {
      const toolBits = step.tool_name
        ? `<pre>${escapeHtml(JSON.stringify({ tool_input: step.tool_input, tool_output: step.tool_output }, null, 2))}</pre>`
        : "";
      return `<div class="reasoning-step">
        <div class="step-type">${escapeHtml(step.step_type)}</div>
        <div>${escapeHtml(step.description)}</div>
        ${toolBits}
      </div>`;
    })
    .join("");
}
