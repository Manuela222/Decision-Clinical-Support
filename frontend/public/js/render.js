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

function summarizeReasoningTrace(steps) {
  const counts = {};
  for (const s of steps) counts[s.step_type] = (counts[s.step_type] || 0) + 1;
  const parts = [];
  if (counts.retrieval) parts.push(`looked up ${counts.retrieval} similar-patient search${counts.retrieval > 1 ? "es" : ""}`);
  if (counts.tool_call) parts.push(`ran ${counts.tool_call} tool call${counts.tool_call > 1 ? "s" : ""}`);
  if (counts.hypertension_compatibility_check) {
    parts.push(`checked hypertension compatibility for ${counts.hypertension_compatibility_check} medication class${counts.hypertension_compatibility_check > 1 ? "es" : ""}`);
  }
  const rejected = steps.filter((s) => s.step_type === "reasoning" && /Rejected candidate/.test(s.description || ""));
  if (rejected.length) parts.push(`rejected ${rejected.length} invalid medication class${rejected.length > 1 ? "es" : ""} not in the fixed vocabulary`);

  let sentence = parts.length ? `The agent ${parts.join(", ")}.` : "The agent reasoned directly from the patient profile.";
  const conclusion = steps.find((s) => s.step_type === "conclusion");
  if (conclusion) sentence += ` Result: ${conclusion.description}`;
  return sentence;
}

function renderReasoningTrace(steps) {
  if (!steps || steps.length === 0) {
    return '<p class="muted">No reasoning trace (only the agent method produces one).</p>';
  }
  const detailRows = steps
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

  return `<p class="agent-summary">${escapeHtml(summarizeReasoningTrace(steps))}</p>
    <details class="tech-details">
      <summary>Technical details (${steps.length} step${steps.length > 1 ? "s" : ""}, raw tool calls)</summary>
      <div class="tech-details-body">${detailRows}</div>
    </details>`;
}

function resultToPlainText(title, result) {
  const lines = [`${title}${result.model_version ? " (" + result.model_version + ")" : ""}`, "=".repeat(40)];
  if (!result.recommended_medications || result.recommended_medications.length === 0) {
    lines.push("No medication classes recommended.");
  }
  for (const rec of result.recommended_medications || []) {
    lines.push("");
    lines.push(`- ${rec.medication_class} [${rec.action}]  (confidence: ${(rec.confidence * 100).toFixed(0)}%)`);
    lines.push(`  Hypertension: ${rec.hypertension_compatible ? "Compatible" : "NOT compatible"} -- ${rec.hypertension_reasoning}`);
    lines.push(`  Rationale: ${rec.rationale}`);
  }
  if (result.safety_warnings && result.safety_warnings.length) {
    lines.push("", "Safety warnings:");
    for (const w of result.safety_warnings) lines.push(`  [${w.severity.toUpperCase()}] ${w.category}: ${w.message}`);
  }
  return lines.join("\n");
}

function buildComparisonBanner(resultsByMethod, methodLabels, onJumpTo) {
  const keys = Object.keys(resultsByMethod).filter((k) => resultsByMethod[k]);
  const banner = document.createElement("div");
  if (keys.length < 2) return banner;

  const sets = {};
  for (const k of keys) sets[k] = new Set(resultsByMethod[k].recommended_medications.map((r) => r.medication_class));
  const allClasses = [...new Set(keys.flatMap((k) => [...sets[k]]))];
  const common = allClasses.filter((c) => keys.every((k) => sets[k].has(c)));
  const identical = allClasses.length === common.length && keys.every((k) => sets[k].size === common.length);

  if (identical) {
    const pills = common.map((c) => `<span class="pill match">${escapeHtml(c)}</span>`).join(" ");
    banner.className = "compare-banner compare-ok";
    banner.innerHTML = `<strong>&#10003; All ${keys.length} methods agree</strong> — same recommended classes:
      <div class="compare-pills">${pills || '<span class="muted">none</span>'}</div>`;
    return banner;
  }

  const perMethod = keys
    .map((k) => {
      const unique = [...sets[k]].filter((c) => !common.includes(c));
      if (unique.length === 0) return "";
      const pills = unique
        .map((c) => `<button type="button" class="pill-button" data-jump="${k}">${escapeHtml(c)} &#8599;</button>`)
        .join(" ");
      return `<div class="compare-row"><span class="compare-method-label">${escapeHtml(methodLabels[k] || k)} only:</span> ${pills}</div>`;
    })
    .join("");

  const commonHtml = common.length
    ? `<div class="compare-row"><span class="compare-method-label">Agreed by all:</span> ${common.map((c) => `<span class="pill match">${escapeHtml(c)}</span>`).join(" ")}</div>`
    : `<div class="compare-row muted">No class is recommended by all ${keys.length} methods.</div>`;

  banner.className = "compare-banner compare-diff";
  banner.innerHTML = `<strong>&#9888; Methods differ</strong> — click a class to jump to that method's full view:
    ${commonHtml}${perMethod}`;
  if (onJumpTo) {
    banner.querySelectorAll("[data-jump]").forEach((btn) => {
      btn.addEventListener("click", () => onJumpTo(btn.dataset.jump));
    });
  }
  return banner;
}
