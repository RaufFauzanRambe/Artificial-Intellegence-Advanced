"""
FastAPI Server for AI Model Serving

Provides a REST API with input validation, CORS support, health checks,
and automatic OpenAPI documentation.
"""

import os
import sys
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Project root on path
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.model import AIModel

# ---------------------------------------------------------------------------
# Pydantic models for request / response
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Request body for the /predict endpoint.

    The ``features`` field must contain exactly ``input_dim`` float values
    representing a single sample.  For batch prediction, use ``/predict_batch``.
    """

    features: List[float] = Field(
        ...,
        description="A list of feature values (floats) for a single sample.",
        min_length=1,
    )

    @field_validator("features", mode="after")
    @classmethod
    def check_feature_values(cls, v: List[float]) -> List[float]:
        for i, val in enumerate(v):
            if not isinstance(val, (int, float)):
                raise ValueError(f"Feature at index {i} is not a number: {val!r}")
        return v


class PredictResponse(BaseModel):
    """Response body for single-sample prediction."""

    logits: List[float] = Field(..., description="Raw model output logits.")
    probabilities: Optional[List[float]] = Field(None, description="Softmax/sigmoid probabilities.")
    predicted_class: int = Field(..., description="The class with the highest probability.")
    confidence: float = Field(..., description="Confidence score of the predicted class.")


class BatchPredictRequest(BaseModel):
    """Request body for batch prediction."""

    samples: List[List[float]] = Field(
        ...,
        description="A list of samples, where each sample is a list of floats.",
    )


class BatchPredictResponse(BaseModel):
    """Response body for batch prediction."""

    predictions: List[PredictResponse] = Field(..., description="List of prediction results.")
    batch_size: int = Field(..., description="Number of samples processed.")


class HealthResponse(BaseModel):
    """Response body for the health check endpoint."""

    status: str = Field(..., description="Service status, typically 'healthy'.")
    model_loaded: bool = Field(..., description="Whether the model is loaded and ready.")
    input_dim: int = Field(..., description="Expected input feature dimension.")
    output_dim: int = Field(..., description="Model output dimension.")
    device: str = Field(..., description="Device the model is running on (cpu/cuda).")


# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Advanced – Prediction API",
    description=(
        "A FastAPI-based REST service for the Artificial Intelligence Advanced project. "
        "Provides endpoints for single and batch predictions, health checks, and "
        "automatic OpenAPI documentation."
    ),
    version="1.0.0",
)

# CORS – allow all origins for development; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

# Configurable via environment variables
INPUT_DIM = int(os.environ.get("AI_INPUT_DIM", "128"))
OUTPUT_DIM = int(os.environ.get("AI_OUTPUT_DIM", "1"))
HIDDEN_DIMS = [int(x) for x in os.environ.get("AI_HIDDEN_DIMS", "256,128,64").split(",")]
DROPOUT_RATE = float(os.environ.get("AI_DROPOUT", "0.3"))
CHECKPOINT_PATH = os.environ.get(
    "AI_CHECKPOINT_PATH",
    os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt"),
)

_model: Optional[AIModel] = None
_device: Optional[torch.device] = None


def _load_model() -> AIModel:
    """Build the model and optionally load weights from a checkpoint."""
    global _model, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = AIModel(
        input_dim=INPUT_DIM,
        hidden_dims=HIDDEN_DIMS,
        output_dim=OUTPUT_DIM,
        dropout_rate=DROPOUT_RATE,
    ).to(_device)
    _model.eval()

    if os.path.isfile(CHECKPOINT_PATH):
        state = torch.load(CHECKPOINT_PATH, map_location=_device)
        _model.load_state_dict(state["model_state_dict"])
        print(f"[fastapi] Loaded checkpoint from {CHECKPOINT_PATH} (epoch {state.get('epoch', '?')})")
    else:
        print("[fastapi] No checkpoint found – using freshly initialized model weights")

    return _model


@app.on_event("startup")
def startup_event():
    """Load the model when the FastAPI application starts."""
    _load_model()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirect the root URL to the Swagger docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Return the health status of the service and model."""
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        device=str(_device) if _device else "unknown",
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """Run inference on a single feature vector.

    The input must contain exactly ``input_dim`` float values.  Returns the
    raw logits, probabilities, predicted class, and confidence score.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    features = request.features
    if len(features) != INPUT_DIM:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {INPUT_DIM} features, received {len(features)}.",
        )

    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = _model(x).cpu().numpy().flatten().tolist()

    # Compute probabilities and predicted class
    if OUTPUT_DIM == 1:
        sigmoid_val = torch.sigmoid(torch.tensor(logits)).item()
        probabilities = [sigmoid_val, 1.0 - sigmoid_val]
        predicted_class = 1 if sigmoid_val >= 0.5 else 0
        confidence = max(sigmoid_val, 1.0 - sigmoid_val)
    else:
        probs = torch.softmax(torch.tensor(logits), dim=0).numpy().tolist()
        probabilities = probs
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])

    return PredictResponse(
        logits=logits,
        probabilities=probabilities,
        predicted_class=predicted_class,
        confidence=round(confidence, 6),
    )


@app.post("/predict_batch", response_model=BatchPredictResponse, tags=["Prediction"])
def predict_batch(request: BatchPredictRequest):
    """Run batch inference on multiple feature vectors."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    samples = request.samples
    if len(samples) == 0:
        raise HTTPException(status_code=422, detail="No samples provided.")

    for idx, sample in enumerate(samples):
        if len(sample) != INPUT_DIM:
            raise HTTPException(
                status_code=422,
                detail=f"Sample at index {idx} has {len(sample)} features, expected {INPUT_DIM}.",
            )

    x = torch.tensor(samples, dtype=torch.float32).to(_device)

    with torch.no_grad():
        logits = _model(x).cpu().numpy()

    predictions: List[PredictResponse] = []
    for row in logits:
        row_list = row.tolist()
        if OUTPUT_DIM == 1:
            sig = torch.sigmoid(torch.tensor(row_list)).item()
            probs = [sig, 1.0 - sig]
            cls = 1 if sig >= 0.5 else 0
            conf = max(sig, 1.0 - sig)
        else:
            probs = torch.softmax(torch.tensor(row_list), dim=0).numpy().tolist()
            cls = int(np.argmax(probs))
            conf = float(probs[cls])
        predictions.append(
            PredictResponse(
                logits=row_list,
                probabilities=probs,
                predicted_class=cls,
                confidence=round(conf, 6),
            )
        )

    return BatchPredictResponse(predictions=predictions, batch_size=len(samples))


# ---------------------------------------------------------------------------
# Entrypoint (for running directly with ``python fastapi_server.py``)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("fastapi_server:app", host="0.0.0.0", port=port, reload=False)
