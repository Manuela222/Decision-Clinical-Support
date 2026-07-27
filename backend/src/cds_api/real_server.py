"""Real-data entrypoint: wires an AppState built from the actual
MIMIC_III_10k extract (see `real_data.build_real_app_state`) into the
FastAPI app, with a real `OpenAIProvider` for the agent (requires
OPENAI_API_KEY in the environment). Unlike `dev_server.py`, this serves
real cohort patients, not the synthetic demo cohort.

Run with: python -m cds_api.real_server
"""
from .dependencies import set_app_state
from .real_data import build_real_app_state
from .routes import app

print("Building AppState from real MIMIC_III_10k data (cohort selection, timelines, model training, RAG index)...")
set_app_state(build_real_app_state())
print("Done. Real cohort loaded.")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
