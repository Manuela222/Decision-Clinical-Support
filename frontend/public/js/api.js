// Shared FastAPI client helper. The frontend only ever talks to this
// backend's HTTP API -- never to the LLM or any data module directly.
let cachedApiBaseUrl = null;

async function getApiBaseUrl() {
  if (cachedApiBaseUrl) return cachedApiBaseUrl;
  const res = await fetch("/config");
  const config = await res.json();
  cachedApiBaseUrl = config.apiBaseUrl;
  return cachedApiBaseUrl;
}

async function apiRequest(path, options = {}) {
  const base = await getApiBaseUrl();
  const res = await fetch(base + path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = body && body.detail ? body.detail : res.statusText;
    throw new Error(`${res.status}: ${detail}`);
  }
  return body;
}

const api = {
  health: () => apiRequest("/health"),
  cohort: () => apiRequest("/cohort"),
  timeline: (subjectId, hadmId) => apiRequest(`/patients/${subjectId}/admissions/${hadmId}/timeline`),
  clinicalState: (subjectId, hadmId) => apiRequest(`/patients/${subjectId}/admissions/${hadmId}/clinical-state`),
  predictBaseline: (subjectId, hadmId) =>
    apiRequest("/predict/baseline", { method: "POST", body: JSON.stringify({ subject_id: subjectId, hadm_id: hadmId }) }),
  predictModel: (subjectId, hadmId) =>
    apiRequest("/predict/model", { method: "POST", body: JSON.stringify({ subject_id: subjectId, hadm_id: hadmId }) }),
  predictAgent: (subjectId, hadmId) =>
    apiRequest("/predict/agent", { method: "POST", body: JSON.stringify({ subject_id: subjectId, hadm_id: hadmId }) }),
  predictAgentNewPatient: (payload) =>
    apiRequest("/predict/agent/new-patient", { method: "POST", body: JSON.stringify(payload) }),
  explain: (subjectId, hadmId, method) => apiRequest(`/explain/${subjectId}/${hadmId}?method=${encodeURIComponent(method)}`),
  syntheticNewPatient: () => apiRequest("/synthetic/new-patient"),
};
