"""Local dev/demo entrypoint: wires a fake AppState (synthetic data, see
`app_state.build_fake_app_state`) into the FastAPI app and runs it with
uvicorn. NOT the production wiring — a real deployment would build AppState
from the actual MIMIC-III extract (Phase 2/3/6) and a real trained model
(Phase 9) instead of the synthetic cohort. The agent panel makes real
OpenAI API calls (requires OPENAI_API_KEY in the environment).

Run with: python -m cds_api.dev_server
"""
from cds.agent import OpenAIProvider

from .app_state import build_fake_app_state
from .dependencies import set_app_state
from .routes import app

set_app_state(build_fake_app_state(llm_provider_factory=OpenAIProvider))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
