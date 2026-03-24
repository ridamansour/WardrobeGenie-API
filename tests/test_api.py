"""
tests/test_api.py
Unit tests for the WardrobeGenie API.
"""
import sys
import os
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

# 1. CI/CD PATHING FIX: Forces GitHub Actions to recognize the root project directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app, ml_models

# We use the TestClient to send fake HTTP requests to our API
client = TestClient(app)


# --- MOCKING THE ML LAYER & DATABASE ---
# Before tests run, we inject lightweight "mock" objects into the global dictionary
# so the API doesn't try to load the 5GB PyTorch models or connect to a live database.
@pytest.fixture(autouse=True)
def mock_ml_models():
    # MagicMock() creates dummy objects that won't crash when their methods are called
    ml_models['detector'] = MagicMock()
    ml_models['attribute_predictor'] = MagicMock()
    ml_models['garment_embedder'] = MagicMock()
    ml_models['intent_extractor'] = MagicMock()

    # Mock the brain so `brain.update_feedback` doesn't crash the feedback test
    ml_models['brain'] = MagicMock()

    # Mock the Qdrant connection so the endpoints don't try to hit localhost:6333
    ml_models['qdrant'] = MagicMock()

    yield
    ml_models.clear()


# --- THE TESTS ---

def test_health_check():
    """Test if the server boots up, responds, and registers the mocked database."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"

    # Safely checks for the Qdrant key if your main.py is returning it
    if "qdrant_connected" in data:
        assert data["qdrant_connected"] is True


def test_recommend_search_validation():
    """Test that Pydantic properly blocks bad JSON payloads on the RAG endpoint."""
    # We are missing required fields like 'context' and 'query'
    bad_payload = {
        "some_random_key": "no_context_provided"
    }

    # Hit the NEW RAG vector search endpoint
    response = client.post("/recommend/search", json=bad_payload)

    # 422 is FastAPI's code for "Unprocessable Entity" (Validation Error)
    assert response.status_code == 422


def test_feedback_endpoint_background_task():
    """Test if the reinforcement learning endpoint accepts correct data."""
    good_payload = {
        "session_id": "test_123",
        "outfit_embedding": [0.1] * 128,  # A fake 128-dim vector
        "liked": True,
        "s_style": 0.8,
        "s_rel": 0.9
    }

    response = client.post("/feedback", json=good_payload)

    # Because we mocked the ML brain with MagicMock, the background task will call
    # a fake `update_feedback` method without crashing, returning a 200 OK!
    assert response.status_code == 200

    # Exact string match with your main.py (including the three dots)
    assert response.json()["status"] == "Centroid shifting in background..."