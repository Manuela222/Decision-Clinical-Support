// Phase 17: thin Express static server. Serves the two pages and a small
// /config endpoint telling the browser where the FastAPI backend lives --
// the frontend never talks to the LLM or any data module directly, only
// ever to that backend's HTTP API.
const express = require("express");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 3000;
const API_BASE_URL = process.env.API_BASE_URL || "http://localhost:8000";

app.get("/config", (req, res) => {
  res.json({ apiBaseUrl: API_BASE_URL });
});

app.use(express.static(path.join(__dirname, "..", "public")));

app.listen(PORT, () => {
  console.log(`cds-frontend listening on http://localhost:${PORT} (backend: ${API_BASE_URL})`);
});
