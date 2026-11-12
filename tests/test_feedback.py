"""Retours utilisateurs : endpoints, statistiques, export et migration."""

import pytest
from fastapi.testclient import TestClient

from app.database import SessionLocal
from app.main import app
from app.models import Analysis

client = TestClient(app)


@pytest.fixture()
def analysis_id():
    """Crée une analyse directement en base (aucun modèle ML nécessaire)."""
    with SessionLocal() as db:
        record = Analysis(
            text="Le service était correct sans plus.",
            sentiment="positive",
            confidence=0.61,
            low_confidence=False,
            keywords="[]",
            model_name="modele-de-test",
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        record_id = record.id
    yield record_id
    with SessionLocal() as db:
        record = db.get(Analysis, record_id)
        if record is not None:
            db.delete(record)
            db.commit()


class TestFeedbackEndpoints:
    def test_set_feedback_correction(self, analysis_id):
        response = client.put(f"/analysis/{analysis_id}/feedback", json={"label": "neutral"})
        assert response.status_code == 200
        data = response.json()
        assert data["user_label"] == "neutral"
        assert data["sentiment"] == "positive"  # la prédiction d'origine est conservée
        assert data["user_label_at"].endswith(("Z", "+00:00"))

    def test_confirmation_uses_same_label(self, analysis_id):
        data = client.put(f"/analysis/{analysis_id}/feedback", json={"label": "positive"}).json()
        assert data["user_label"] == data["sentiment"]

    def test_feedback_is_persisted(self, analysis_id):
        client.put(f"/analysis/{analysis_id}/feedback", json={"label": "negative"})
        assert client.get(f"/analysis/{analysis_id}").json()["user_label"] == "negative"

    def test_clear_feedback(self, analysis_id):
        client.put(f"/analysis/{analysis_id}/feedback", json={"label": "negative"})
        data = client.delete(f"/analysis/{analysis_id}/feedback").json()
        assert data["user_label"] is None
        assert data["user_label_at"] is None

    def test_rejects_unknown_label(self, analysis_id):
        response = client.put(f"/analysis/{analysis_id}/feedback", json={"label": "joyeux"})
        assert response.status_code == 422

    def test_returns_404_for_unknown_analysis(self):
        response = client.put("/analysis/999999/feedback", json={"label": "neutral"})
        assert response.status_code == 404

    def test_model_name_is_exposed(self, analysis_id):
        assert client.get(f"/analysis/{analysis_id}").json()["model_name"] == "modele-de-test"


class TestFeedbackStats:
    def test_counts_feedback_and_corrections(self, analysis_id):
        before = client.get("/stats").json()
        client.put(f"/analysis/{analysis_id}/feedback", json={"label": "neutral"})
        after = client.get("/stats").json()
        assert after["with_feedback"] == before["with_feedback"] + 1
        assert after["corrected"] == before["corrected"] + 1

    def test_confirmation_is_not_a_correction(self, analysis_id):
        before = client.get("/stats").json()
        client.put(f"/analysis/{analysis_id}/feedback", json={"label": "positive"})
        after = client.get("/stats").json()
        assert after["with_feedback"] == before["with_feedback"] + 1
        assert after["corrected"] == before["corrected"]


class TestFeedbackExport:
    def test_collects_only_rows_with_feedback(self, analysis_id):
        from textsense_ml.export_feedback import collect_feedback

        with SessionLocal() as db:
            assert f"fb-{analysis_id}" not in set(collect_feedback(db)["id"])
        client.put(f"/analysis/{analysis_id}/feedback", json={"label": "neutral"})
        with SessionLocal() as db:
            df = collect_feedback(db)
            only_corrections = collect_feedback(db, only_corrections=True)
        row = df[df["id"] == f"fb-{analysis_id}"].iloc[0]
        assert row["label"] == "neutral"
        assert row["model_label"] == "positive"
        assert f"fb-{analysis_id}" in set(only_corrections["id"])


class TestHistoryLimitValidation:
    def test_rejects_non_positive_limit(self):
        assert client.get("/history?limit=0").status_code == 422
        assert client.get("/history?limit=-5").status_code == 422
