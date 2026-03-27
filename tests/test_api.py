from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_ready" in response.json()


def test_home_page_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "TextSense" in response.text


def test_analyze_returns_503_when_model_missing():
    model_path = Path("saved_models/sentiment_model.pkl")
    vectorizer_path = Path("saved_models/vectorizer.pkl")
    if model_path.exists() and vectorizer_path.exists():
        return

    response = client.post("/analyze", json={"text": "Ce projet semble prometteur et utile."})
    assert response.status_code == 503


def test_validation_on_short_text():
    response = client.post("/analyze", json={"text": "bad"})
    assert response.status_code == 422
