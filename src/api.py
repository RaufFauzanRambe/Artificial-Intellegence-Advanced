"""
REST API module for the Artificial Intelligence Advanced project.

Provides a FastAPI application with CORS support, health and model-info
endpoints, and a ``/predict`` POST endpoint.  The trained model is loaded
once at startup and reused for all subsequent inference requests.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Lazy imports — FastAPI may not be installed in all environments
# ------------------------------------------------------------------

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    FastAPI = object  # type: ignore[assignment,misc]
    CORSMiddleware = object  # type: ignore[assignment,misc]
    HTTPException = Exception  # type: ignore[assignment,misc]

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore[assignment]


# ------------------------------------------------------------------
# Global state
# ------------------------------------------------------------------

_model: Any = None
_model_metadata: Dict[str, Any] = {}
_device: str = "cpu"


# ------------------------------------------------------------------
# Pydantic schemas
# ------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Schema for the ``/predict`` request body.

    Attributes
    ----------
    features : list[list[float]]
        2-D array of input features — one row per sample.
    model_name : str, optional
        Name of the model to use (if multiple are available).
    """
    features: List[List[float]] = Field(
        ...,
        description="2-D array of input features (batch of samples).",
        example=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    )
    model_name: Optional[str] = Field(None, description="Optional model selector.")


class PredictResponse(BaseModel):
    """Schema for the ``/predict`` response.

    Attributes
    ----------
    predictions : list
        Model predictions (class labels or scores).
    model_name : str
        Name of the model that produced the predictions.
    input_shape : list[int]
        Shape of the input feature matrix.
    """
    predictions: List[Any]
    model_name: str = "unknown"
    input_shape: List[int] = [0, 0]


class HealthResponse(BaseModel):
    """Schema for the ``/health`` endpoint."""
    status: str = "ok"
    model_loaded: bool = False
    device: str = "cpu"


class ModelInfoResponse(BaseModel):
    """Schema for the ``/model/info`` endpoint."""
    model_name: str = "unknown"
    model_type: str = "unknown"
    input_dim: int = 0
    output_dim: int = 0
    num_parameters: int = 0
    device: str = "cpu"
    metadata: Dict[str, Any] = {}


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------

def load_model(
    model_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Any:
    """Load a trained model from disk.

    Supports ``.pt`` / ``.pth`` (PyTorch) and ``.npy`` (NumPy proxy) files.

    Parameters
    ----------
    model_path : str, optional
        Path to the saved model checkpoint.  Falls back to the environment
        variable ``AI_MODEL_PATH`` or ``models/model.pt``.
    device : str, optional
        Target device.  Falls back to ``AI_DEVICE`` or ``"cpu"``.

    Returns
    -------
    Any
        The loaded model.
    """
    global _model, _model_metadata, _device

    if model_path is None:
        model_path = os.environ.get("AI_MODEL_PATH", "models/model.pt")

    if device is None:
        device = os.environ.get("AI_DEVICE", "cpu")

    _device = device
    model_path = Path(model_path)

    if not model_path.exists():
        logger.warning("Model file not found at %s — running without a model.", model_path)
        _model_metadata["load_error"] = f"File not found: {model_path}"
        return None

    suffix = model_path.suffix.lower()

    if suffix in (".pt", ".pth") and HAS_TORCH:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        _model = checkpoint.get("model", checkpoint)

        # Move to device
        if hasattr(_model, "to"):
            _model = _model.to(device)
        if hasattr(_model, "eval"):
            _model.eval()

        _model_metadata = checkpoint.get("metadata", {})
        _model_metadata.setdefault("model_type", type(_model).__name__)
        logger.info("Loaded PyTorch model from %s", model_path)

    elif suffix == ".npy":
        _model = np.load(model_path, allow_pickle=True).item()
        _model_metadata["model_type"] = "numpy"
        logger.info("Loaded NumPy model from %s", model_path)

    else:
        logger.warning("Unsupported model format: %s", suffix)
        _model_metadata["load_error"] = f"Unsupported format: {suffix}"
        _model = None

    return _model


# ------------------------------------------------------------------
# Inference helper
# ------------------------------------------------------------------

def _run_inference(features: np.ndarray) -> np.ndarray:
    """Run model inference on a batch of features.

    Parameters
    ----------
    features : np.ndarray of shape (n_samples, n_features)

    Returns
    -------
    np.ndarray of shape (n_samples,) or (n_samples, n_classes)
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="No model loaded on server.")

    if HAS_TORCH and hasattr(_model, "forward"):
        tensor = torch.tensor(features, dtype=torch.float32).to(_device)
        with torch.no_grad():
            output = _model(tensor)
        if hasattr(output, "cpu"):
            output = output.cpu()
        return output.numpy()

    # Fallback: try calling as a callable (NumPy-based models)
    if callable(_model):
        return np.asarray(_model(features))

    raise HTTPException(status_code=500, detail="Model is not callable.")


# ------------------------------------------------------------------
# FastAPI application factory
# ------------------------------------------------------------------

def create_app(
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    cors_origins: Optional[List[str]] = None,
) -> "FastAPI":
    """Create and configure the FastAPI application.

    Parameters
    ----------
    model_path : str, optional
        Path to the model checkpoint.
    device : str, optional
        Compute device.
    cors_origins : list[str], optional
        Allowed CORS origins.  Defaults to ``["*"]``.

    Returns
    -------
    FastAPI
    """
    if not HAS_FASTAPI:
        raise RuntimeError(
            "FastAPI is not installed. Install it with: pip install fastapi uvicorn"
        )

    app = FastAPI(
        title="AI Advanced — Inference API",
        description="REST API for model predictions.",
        version="1.0.0",
    )

    # CORS
    if cors_origins is None:
        cors_origins = ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ----------------------------------------------------------------
    # Startup event — load model
    # ----------------------------------------------------------------
    @app.on_event("startup")
    async def _startup() -> None:
        load_model(model_path=model_path, device=device)
        logger.info("API startup complete. Model loaded: %s", _model is not None)

    # ----------------------------------------------------------------
    # Endpoints
    # ----------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        """Liveness and readiness check."""
        return HealthResponse(
            status="ok",
            model_loaded=_model is not None,
            device=_device,
        )

    @app.get("/model/info", response_model=ModelInfoResponse, tags=["model"])
    async def model_info() -> ModelInfoResponse:
        """Return metadata about the loaded model."""
        info = ModelInfoResponse(
            model_name=_model_metadata.get("model_name", "unknown"),
            model_type=_model_metadata.get("model_type", type(_model).__name__ if _model else "none"),
            input_dim=_model_metadata.get("input_dim", 0),
            output_dim=_model_metadata.get("output_dim", 0),
            num_parameters=_model_metadata.get("num_parameters", 0),
            device=_device,
            metadata=_model_metadata,
        )
        return info

    @app.post("/predict", response_model=PredictResponse, tags=["inference"])
    async def predict(request: PredictRequest) -> PredictResponse:
        """Run inference on a batch of features.

        The response contains the model's predicted labels or scores.
        """
        if not request.features:
            raise HTTPException(status_code=400, detail="features array must not be empty.")

        features = np.array(request.features, dtype=np.float32)

        try:
            outputs = _run_inference(features)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

        # Convert probabilities to class labels if multi-class output
        if outputs.ndim == 2 and outputs.shape[1] > 1:
            predictions = np.argmax(outputs, axis=1).tolist()
        else:
            predictions = outputs.ravel().tolist()

        return PredictResponse(
            predictions=predictions,
            model_name=_model_metadata.get("model_name", "unknown"),
            input_shape=list(features.shape),
        )

    return app


# ------------------------------------------------------------------
# Module-level convenience
# ------------------------------------------------------------------

# Create a default app instance (model is loaded lazily on startup)
app: Optional["FastAPI"] = None
if HAS_FASTAPI:
    app = create_app()


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
) -> None:
    """Start the uvicorn server programmatically.

    Parameters
    ----------
    host : str
        Bind address.
    port : int
        Bind port.
    model_path : str, optional
        Path to model checkpoint.
    device : str, optional
        Target device.
    """
    if not HAS_FASTAPI:
        raise RuntimeError("FastAPI is required. Install with: pip install fastapi uvicorn")

    srv_app = create_app(model_path=model_path, device=device)

    try:
        import uvicorn
        uvicorn.run(srv_app, host=host, port=port)
    except ImportError:
        raise RuntimeError("uvicorn is required. Install with: pip install uvicorn")
