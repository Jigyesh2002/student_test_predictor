"""
main.py — FastAPI REST API for the Student Score Predictor.

Endpoints:
  GET  /health        — liveness check
  GET  /stats         — descriptive statistics
  GET  /metrics       — model evaluation metrics
  GET  /correlation   — Pearson correlation matrix
  GET  /scatter-data  — sampled dataset rows for visualisation
  POST /predict       — predict exam_score for given inputs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .model import get_model
from .schemas import PredictRequest, PredictResponse

app = FastAPI(
    title="Student Score Predictor API",
    description="Predicts exam_score from study_hours_per_day and sleep_hours.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pre-load model at startup so first request is fast
@app.on_event("startup")
async def _preload():
    get_model()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Utility"])
def health():
    return {"status": "ok"}


@app.get("/stats", tags=["Data"])
def stats():
    """Descriptive statistics for study_hours_per_day, sleep_hours, exam_score."""
    return get_model().summary_stats()


@app.get("/metrics", tags=["Model"])
def metrics():
    """MAE, RMSE, R² for Linear Regression and Random Forest."""
    return get_model().metrics()


@app.get("/correlation", tags=["Data"])
def correlation():
    """Pearson correlation matrix for the three key columns."""
    return get_model().correlation()


@app.get("/scatter-data", tags=["Data"])
def scatter_data():
    """Up to 2 000 sampled rows with study_hours, sleep_hours, exam_score."""
    return get_model().scatter_data()


@app.post("/predict", response_model=PredictResponse, tags=["Model"])
def predict(req: PredictRequest):
    """Predict exam_score and return a 95 % prediction interval."""
    try:
        result = get_model().predict(req.study_hours, req.sleep_hours, req.model)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictResponse(
        **result,
        model_used=req.model,
    )
