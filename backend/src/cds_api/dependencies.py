"""FastAPI dependency provider for AppState.

`get_app_state` reads a module-level singleton set by `set_app_state` (the
real deployment entrypoint would call this once at startup after loading
real MIMIC data, training the real model, etc. — outside this phase's
scope). Tests override this dependency directly via
`app.dependency_overrides`, per FastAPI's standard testing pattern, rather
than relying on the singleton at all.
"""
from typing import Optional

from .app_state import AppState

_app_state: Optional[AppState] = None


def set_app_state(state: AppState) -> None:
    global _app_state
    _app_state = state


def get_app_state() -> AppState:
    if _app_state is None:
        raise RuntimeError(
            "AppState has not been initialized. Call cds_api.dependencies.set_app_state() at startup."
        )
    return _app_state
