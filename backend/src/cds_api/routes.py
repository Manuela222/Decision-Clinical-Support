"""Phase 16: FastAPI routes. Thin only — every handler extracts request
data, calls exactly one services.py function, and shapes the HTTP
response/error. No business logic here."""
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from cds.agent import AgentError
from cds.schemas import ClinicalState, PatientTimeline, RecommendationResult
from cds.explainability import ExplainabilityReport

from . import services
from .app_state import AppState
from .dependencies import get_app_state
from .schemas import CohortPatientResponse, EvaluateRequest, EvaluateResponse, NewPatientRequest, PredictRequest

app = FastAPI(title="Clinical Decision Support API")

# The Node.js frontend (Phase 17) runs on a different origin/port and calls
# this API directly from the browser -- CORS must be open to it. Wide open
# here because this is a local prototype with no auth/cookies in play, not
# a public deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/cohort", response_model=list[CohortPatientResponse])
def get_cohort(state: AppState = Depends(get_app_state)):
    return [
        CohortPatientResponse(
            subject_id=c.subject_id, hadm_id=c.hadm_id, age=c.age, gender=c.gender,
            admission_reason=c.admission_reason, hypertension_status=c.hypertension_status,
        )
        for c in services.get_cohort(state)
    ]


@app.get("/patients/{subject_id}/admissions/{hadm_id}/timeline", response_model=PatientTimeline)
def get_timeline(subject_id: int, hadm_id: int, state: AppState = Depends(get_app_state)):
    try:
        return services.get_timeline(state, subject_id, hadm_id)
    except services.NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.get("/patients/{subject_id}/admissions/{hadm_id}/clinical-state", response_model=ClinicalState)
def get_clinical_state(subject_id: int, hadm_id: int, state: AppState = Depends(get_app_state)):
    try:
        return services.get_clinical_state(state, subject_id, hadm_id)
    except services.NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.post("/predict/baseline", response_model=RecommendationResult)
def predict_baseline(request: PredictRequest, state: AppState = Depends(get_app_state)):
    try:
        return services.predict_baseline(state, request.subject_id, request.hadm_id)
    except services.NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.post("/predict/model", response_model=RecommendationResult)
def predict_model(request: PredictRequest, state: AppState = Depends(get_app_state)):
    try:
        return services.predict_model(state, request.subject_id, request.hadm_id)
    except services.NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.post("/predict/agent", response_model=RecommendationResult)
def predict_agent(request: PredictRequest, state: AppState = Depends(get_app_state)):
    try:
        return services.predict_agent(state, request.subject_id, request.hadm_id)
    except services.NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except AgentError as e:
        raise HTTPException(status_code=502, detail=f"Agent failed: {e}") from e
    except Exception as e:  # noqa: BLE001 - the LLM provider (network, auth, ...) is an external
        # dependency outside this service's control; surface it as a clean 502 instead of an
        # unhandled 500 with no body (which browsers report as an opaque "Failed to fetch").
        raise HTTPException(status_code=502, detail=f"Agent failed unexpectedly: {e}") from e


@app.post("/predict/agent/new-patient", response_model=RecommendationResult)
def predict_agent_new_patient(request: NewPatientRequest, state: AppState = Depends(get_app_state)):
    try:
        return services.predict_agent_new_patient(state, request)
    except services.ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except AgentError as e:
        raise HTTPException(status_code=502, detail=f"Agent failed: {e}") from e
    except Exception as e:  # noqa: BLE001 - see predict_agent's comment above
        raise HTTPException(status_code=502, detail=f"Agent failed unexpectedly: {e}") from e


@app.get("/synthetic/new-patient", response_model=NewPatientRequest)
def synthetic_new_patient():
    """Not one of Phase 16's originally-listed endpoints -- see
    services.get_synthetic_new_patient_example's docstring."""
    return services.get_synthetic_new_patient_example()


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(request: EvaluateRequest, state: AppState = Depends(get_app_state)):
    try:
        report = services.run_evaluation(state, request.hadm_ids, request.include_agent)
    except services.ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return EvaluateResponse(
        comparison_table=report.table.to_dict(orient="records"),
        evaluations=report.evaluations,
    )


@app.get("/explain/{subject_id}/{hadm_id}", response_model=ExplainabilityReport)
def explain(subject_id: int, hadm_id: int, method: str = "baseline", state: AppState = Depends(get_app_state)):
    try:
        return services.explain(state, subject_id, hadm_id, method)
    except services.NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except services.ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
