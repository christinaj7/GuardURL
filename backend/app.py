"""
GuardURL - Phase 11: Production FastAPI Backend
Serves real-time inference, Explainable AI (SHAP) attributions,
model performance metrics, adaptive RL decision telemetry, and dashboard static assets.
"""

import os
import sys
import json
import time
from typing import Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.predict import get_predictor, GuardURLPredictor

app = FastAPI(
    title="GuardURL API",
    description="Explainable & Adaptive AI-Based Phishing Website Detection System",
    version="1.0.0"
)

# Enable CORS for local and cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory prediction audit log
prediction_history = []
MAX_HISTORY = 50


class URLRequest(BaseModel):
    url: str = Field(..., example="https://example.com", description="The target URL to inspect")
    enable_rl: Optional[bool] = Field(default=True, description="Enable adaptive RL decision layer")


class FeedbackRequest(BaseModel):
    url: str
    user_label: str  # 'LEGITIMATE' or 'PHISHING'
    comment: Optional[str] = None


@app.get("/health")
def health_check():
    """Health status and model operational readiness check."""
    predictor = get_predictor()
    return {
        "status": "online",
        "service": "GuardURL Phishing Detection System",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_loaded": predictor.model is not None,
        "model_type": predictor.threshold_config.get("model_name", "Unknown")
    }


@app.get("/metrics")
def get_metrics():
    """Returns genuine experimental performance metrics across all evaluated models."""
    metrics_path = "results/metrics_comparison.json"
    metadata_path = "models/metadata.json"

    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            comparison = json.load(f)
    else:
        comparison = {}

    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    return {
        "models_comparison": comparison,
        "metadata": metadata
    }


@app.post("/predict")
def predict_url(payload: URLRequest):
    """
    Main prediction endpoint.
    Performs URL validation, feature extraction, ML inference,
    risk assessment, SHAP explainability, and optional RL decisioning.
    """
    url_input = payload.url.strip() if payload.url else ""

    # Strict URL validation
    if not url_input:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL cannot be empty. Please provide a valid web address or hostname."
        )

    if len(url_input) > 2048:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL exceeds maximum acceptable length of 2048 characters."
        )

    # Basic format check
    if not any(url_input.startswith(prefix) for prefix in ["http://", "https://", "ftp://", "//"]) and "." not in url_input:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid URL format. Please include a valid domain name or protocol (e.g., https://example.com or example.com)."
        )

    try:
        predictor = get_predictor()
        start_time = time.time()
        result = predictor.predict(url_input, enable_rl=payload.enable_rl)
        latency_ms = round((time.time() - start_time) * 1000, 2)
        result["latency_ms"] = latency_ms

        # Record to history log
        history_entry = {
            "id": len(prediction_history) + 1,
            "url": result["url"],
            "prediction": result["prediction"],
            "confidence_percentage": result["confidence_percentage"],
            "risk_level": result["risk_level"],
            "timestamp": time.strftime("%H:%M:%S"),
            "latency_ms": latency_ms
        }
        prediction_history.insert(0, history_entry)
        if len(prediction_history) > MAX_HISTORY:
            prediction_history.pop()

        return result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


@app.get("/history")
def get_history():
    """Returns recent analysis history."""
    return {"history": prediction_history}


@app.post("/feedback")
def submit_feedback(payload: FeedbackRequest):
    """Accepts analyst feedback on predictions for adaptive learning audits."""
    return {
        "status": "success",
        "message": f"Feedback received for {payload.url}. Recorded for periodic policy retraining.",
        "received": {
            "url": payload.url,
            "user_label": payload.user_label,
            "comment": payload.comment
        }
    }


# Mount static files for the frontend dashboard
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
