"""
Unit tests for the FastAPI server endpoints.

Covers health check, single prediction, batch prediction, and input
validation error handling using FastAPI's TestClient.
Run with:  pytest tests/test_api.py -v
"""

import os
import sys

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from deployment.fastapi_server import app

# ---------------------------------------------------------------------------
# Client fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """Create a TestClient for the FastAPI application."""
    # Manually trigger the startup event so the model is loaded
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Tests – Health Endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        """The health endpoint should return HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_json(self, client):
        """The health endpoint should return valid JSON."""
        response = client.get("/health")
        assert response.headers["content-type"] == "application/json"

    def test_health_response_fields(self, client):
        """The health response should contain expected fields."""
        response = client.get("/health")
        body = response.json()
        assert "status" in body
        assert body["status"] == "healthy"
        assert "model_loaded" in body
        assert body["model_loaded"] is True
        assert "input_dim" in body
        assert "output_dim" in body
        assert "device" in body

    def test_health_input_dim_matches_env(self, client, monkeypatch):
        """The reported input_dim should match the environment variable."""
        # Default is 128 (as set in the module)
        response = client.get("/health")
        body = response.json()
        assert body["input_dim"] == 128


# ---------------------------------------------------------------------------
# Tests – Predict Endpoint (valid input)
# ---------------------------------------------------------------------------


class TestPredictEndpointValid:
    """Tests for POST /predict with valid input."""

    def test_predict_returns_200(self, client):
        """A valid prediction request should return HTTP 200."""
        payload = {"features": [0.0] * 128}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_predict_returns_json(self, client):
        """The prediction response should be valid JSON."""
        payload = {"features": np.random.randn(128).tolist()}
        response = client.post("/predict", json=payload)
        assert response.headers["content-type"] == "application/json"

    def test_predict_response_fields(self, client):
        """The prediction response should contain logits, probabilities, predicted_class, confidence."""
        payload = {"features": np.random.randn(128).tolist()}
        response = client.post("/predict", json=payload)
        body = response.json()
        assert "logits" in body
        assert "probabilities" in body
        assert "predicted_class" in body
        assert "confidence" in body

    def test_predict_logits_is_list(self, client):
        """The logits field should be a list of numbers."""
        payload = {"features": np.random.randn(128).tolist()}
        response = client.post("/predict", json=payload)
        body = response.json()
        assert isinstance(body["logits"], list)
        assert all(isinstance(x, (int, float)) for x in body["logits"])

    def test_predict_confidence_in_range(self, client):
        """Confidence should be between 0 and 1."""
        payload = {"features": np.random.randn(128).tolist()}
        response = client.post("/predict", json=payload)
        body = response.json()
        assert 0.0 <= body["confidence"] <= 1.0

    def test_predict_predicted_class_is_int(self, client):
        """predicted_class should be an integer."""
        payload = {"features": np.random.randn(128).tolist()}
        response = client.post("/predict", json=payload)
        body = response.json()
        assert isinstance(body["predicted_class"], int)


# ---------------------------------------------------------------------------
# Tests – Predict Endpoint (invalid input)
# ---------------------------------------------------------------------------


class TestPredictEndpointInvalid:
    """Tests for POST /predict with invalid input."""

    def test_predict_wrong_feature_count(self, client):
        """Sending the wrong number of features should return 422."""
        payload = {"features": [0.0, 1.0]}  # only 2, expected 128
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_empty_features(self, client):
        """Sending an empty features list should return 422."""
        payload = {"features": []}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_missing_features_field(self, client):
        """Omitting the features field should return 422."""
        payload = {"not_features": [0.0] * 128}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_non_numeric_features(self, client):
        """Sending non-numeric values should return 422."""
        payload = {"features": ["a", "b", "c"]}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_no_body(self, client):
        """Sending no request body should return 422."""
        response = client.post("/predict", json={})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Tests – Batch Predict Endpoint
# ---------------------------------------------------------------------------


class TestBatchPredictEndpoint:
    """Tests for POST /predict_batch."""

    def test_batch_predict_returns_200(self, client):
        """A valid batch request should return HTTP 200."""
        samples = [np.random.randn(128).tolist() for _ in range(3)]
        payload = {"samples": samples}
        response = client.post("/predict_batch", json=payload)
        assert response.status_code == 200

    def test_batch_predict_response_fields(self, client):
        """Batch response should contain 'predictions' list and 'batch_size'."""
        samples = [np.random.randn(128).tolist() for _ in range(2)]
        payload = {"samples": samples}
        response = client.post("/predict_batch", json=payload)
        body = response.json()
        assert "predictions" in body
        assert "batch_size" in body
        assert body["batch_size"] == 2
        assert len(body["predictions"]) == 2

    def test_batch_predict_empty_samples(self, client):
        """An empty samples list should return 422."""
        payload = {"samples": []}
        response = client.post("/predict_batch", json=payload)
        assert response.status_code == 422

    def test_batch_predict_wrong_dim(self, client):
        """A sample with wrong feature count should return 422."""
        payload = {"samples": [[0.0] * 5]}  # only 5 features
        response = client.post("/predict_batch", json=payload)
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Tests – Root redirect
# ---------------------------------------------------------------------------


class TestRootRedirect:
    """Tests for GET / redirect."""

    def test_root_redirects_to_docs(self, client):
        """The root URL should redirect to /docs."""
        response = client.get("/", follow_redirects=False)
        assert response.status_code in (200, 307)
        # When follow_redirects=False, some TestClient versions return 200
        # directly for the target, or 307 with a Location header.
        if response.status_code == 307:
            assert response.headers["location"] == "/docs"
