"""Mode démo public : cloisonnement par visiteur, limite de débit, sauvegarde des retours."""

import json
import uuid

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.database import SessionLocal
from app.demo import SESSION_HEADER, RateLimiter, reset_rate_limiter
from app.feedback_backup import FeedbackBackup
from app.main import app
from app.models import Analysis

client = TestClient(app)


def _session() -> dict[str, str]:
    return {SESSION_HEADER: str(uuid.uuid4())}


@pytest.fixture()
def demo(monkeypatch):
    """Active le mode démo et remplace le modèle par une prédiction fixe."""
    settings = get_settings()
    monkeypatch.setattr(settings, "demo_mode", True)
    monkeypatch.setattr(settings, "rate_limit_per_minute", 1000)
    monkeypatch.setattr(settings, "rate_limit_global_per_minute", 1000)
    monkeypatch.setattr(
        "app.routes.predict_sentiment",
        lambda text: {"sentiment": "positive", "confidence": 0.9, "low_confidence": False, "model": "faux-modele"},
    )
    reset_rate_limiter()
    yield settings
    reset_rate_limiter()


def _analyze(headers, text="Une phrase de test pour la démo."):
    response = client.post("/analyze", json={"text": text}, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()["id"]


class TestSessionIsolation:
    def test_history_only_shows_own_analyses(self, demo):
        alice, bob = _session(), _session()
        alice_id = _analyze(alice)
        bob_id = _analyze(bob)

        alice_ids = {item["id"] for item in client.get("/history", headers=alice).json()}
        assert alice_id in alice_ids
        assert bob_id not in alice_ids

    def test_cannot_read_or_modify_other_visitor_analysis(self, demo):
        alice, bob = _session(), _session()
        alice_id = _analyze(alice)

        assert client.get(f"/analysis/{alice_id}", headers=bob).status_code == 404
        assert client.put(f"/analysis/{alice_id}/feedback", json={"label": "neutral"}, headers=bob).status_code == 404
        assert client.delete(f"/analysis/{alice_id}/feedback", headers=bob).status_code == 404
        assert client.delete(f"/history/{alice_id}", headers=bob).status_code == 404
        # L'analyse est intacte pour sa propriétaire.
        data = client.get(f"/analysis/{alice_id}", headers=alice).json()
        assert data["user_label"] is None

    def test_owner_can_give_feedback_and_delete(self, demo):
        alice = _session()
        alice_id = _analyze(alice)
        response = client.put(f"/analysis/{alice_id}/feedback", json={"label": "neutral"}, headers=alice)
        assert response.status_code == 200
        assert client.delete(f"/history/{alice_id}", headers=alice).status_code == 200

    def test_stats_are_per_visitor(self, demo):
        alice, bob = _session(), _session()
        _analyze(alice)
        _analyze(alice)
        alice_id = _analyze(alice)
        client.put(f"/analysis/{alice_id}/feedback", json={"label": "negative"}, headers=alice)

        stats = client.get("/stats", headers=alice).json()
        assert stats["total"] == 3
        assert stats["with_feedback"] == 1
        assert stats["corrected"] == 1
        assert client.get("/stats", headers=bob).json()["total"] == 0

    @pytest.mark.parametrize("value", [None, "pas-un-uuid", "00000000-0000-1000-8000-000000000000"])
    def test_missing_or_invalid_session_gets_empty_history(self, demo, value):
        _analyze(_session())
        headers = {} if value is None else {SESSION_HEADER: value}
        assert client.get("/history", headers=headers).json() == []

    def test_session_id_is_never_returned(self, demo):
        alice = _session()
        alice_id = _analyze(alice)
        assert "session_id" not in client.get(f"/analysis/{alice_id}", headers=alice).json()

    def test_without_demo_mode_history_is_shared(self, monkeypatch):
        monkeypatch.setattr(get_settings(), "demo_mode", False)
        with SessionLocal() as db:
            record = Analysis(
                text="Analyse d'un autre visiteur.",
                sentiment="neutral",
                confidence=0.7,
                keywords="[]",
                session_id=str(uuid.uuid4()),
            )
            db.add(record)
            db.commit()
            record_id = record.id
        assert client.get(f"/analysis/{record_id}", headers=_session()).status_code == 200


class TestRateLimit:
    def test_limiter_sliding_window(self):
        now = [0.0]
        limiter = RateLimiter(limit=2, window=60, clock=lambda: now[0])
        assert limiter.retry_after("ip") == 0
        assert limiter.retry_after("ip") == 0
        assert limiter.retry_after("ip") == pytest.approx(60)
        assert limiter.retry_after("autre-ip") == 0  # compteur par adresse
        now[0] = 30.0
        assert limiter.retry_after("ip") == pytest.approx(30)
        now[0] = 60.0  # les deux premières requêtes sortent de la fenêtre
        assert limiter.retry_after("ip") == 0

    def test_api_returns_429_with_retry_after(self, demo):
        demo.rate_limit_per_minute = 2
        reset_rate_limiter()
        headers = _session()
        _analyze(headers)
        _analyze(headers)
        response = client.post("/analyze", json={"text": "Une requête de trop."}, headers=headers)
        assert response.status_code == 429
        assert int(response.headers["Retry-After"]) >= 1
        # La lecture n'est pas limitée.
        assert client.get("/history", headers=headers).status_code == 200

    def test_global_limit_applies_across_ip_addresses(self, demo):
        demo.rate_limit_global_per_minute = 2
        reset_rate_limiter()
        for ip in ("203.0.113.1", "203.0.113.2"):
            _analyze({**_session(), "X-Forwarded-For": ip})
        response = client.post(
            "/analyze", json={"text": "Une requête de trop."}, headers={"X-Forwarded-For": "203.0.113.3"}
        )
        assert response.status_code == 429
        assert "sollicitée" in response.json()["detail"]

    def test_no_limit_outside_demo_mode(self, monkeypatch):
        settings = get_settings()
        monkeypatch.setattr(settings, "demo_mode", False)
        monkeypatch.setattr(settings, "rate_limit_per_minute", 1)
        reset_rate_limiter()
        for _ in range(3):
            assert client.get("/stats").status_code == 200
            response = client.put("/analysis/999999/feedback", json={"label": "neutral"})
            assert response.status_code == 404  # et non 429


class TestDemoPage:
    def test_notice_and_links_only_in_demo_mode(self, monkeypatch):
        settings = get_settings()
        monkeypatch.setattr(settings, "demo_mode", True)
        page = client.get("/").text
        assert "Démo publique" in page
        assert "github.com/AhamedT04/TexteSense" in page
        assert client.get("/health").json()["demo_mode"] is True

        monkeypatch.setattr(settings, "demo_mode", False)
        assert "Démo publique" not in client.get("/").text


class FakeHubApi:
    def __init__(self, fail: bool = False):
        self.uploads: list[dict] = []
        self.fail = fail

    def upload_file(self, **kwargs):
        if self.fail:
            raise ConnectionError("réseau indisponible")
        self.uploads.append(kwargs)


class TestFeedbackBackup:
    @pytest.fixture()
    def records(self):
        """Une analyse avec avis (sauvegardée) et une sans avis (ignorée)."""
        session = str(uuid.uuid4())
        with SessionLocal() as db:
            with_feedback = Analysis(
                text="Texte annoté par le visiteur.",
                sentiment="positive",
                confidence=0.8,
                keywords="[]",
                model_name="faux-modele",
                user_label="neutral",
                session_id=session,
            )
            without_feedback = Analysis(
                text="Texte sans avis.", sentiment="negative", confidence=0.9, keywords="[]", session_id=session
            )
            db.add_all([with_feedback, without_feedback])
            db.commit()
            ids = (with_feedback.id, without_feedback.id)
        yield ids
        with SessionLocal() as db:
            for record_id in ids:
                record = db.get(Analysis, record_id)
                if record is not None:
                    db.delete(record)
            db.commit()

    def _rows(self, upload) -> list[dict]:
        return [json.loads(line) for line in upload["path_or_fileobj"].decode("utf-8").splitlines()]

    def test_pushes_only_annotated_analyses_without_session(self, records):
        api = FakeHubApi()
        backup = FeedbackBackup("compte/retours", SessionLocal, api=api)
        assert backup.push() is True

        upload = api.uploads[0]
        assert upload["repo_type"] == "dataset"
        assert upload["repo_id"] == "compte/retours"
        assert upload["path_in_repo"].startswith("feedback/") and upload["path_in_repo"].endswith(".jsonl")
        rows = {row["id"]: row for row in self._rows(upload)}
        annotated, not_annotated = records
        assert f"fb-{annotated}" in rows
        assert f"fb-{not_annotated}" not in rows
        row = rows[f"fb-{annotated}"]
        assert row["label"] == "neutral" and row["model_label"] == "positive"
        assert "session_id" not in row
        assert row["created_at"].endswith("+00:00")

    def test_skips_unchanged_snapshot_and_pushes_deletions(self, records):
        api = FakeHubApi()
        backup = FeedbackBackup("compte/retours", SessionLocal, api=api)
        assert backup.push() is True
        assert backup.push() is False  # rien n'a changé
        assert len(api.uploads) == 1

        annotated, _ = records
        with SessionLocal() as db:
            db.delete(db.get(Analysis, annotated))
            db.commit()
        assert backup.push() is True  # la suppression est répercutée
        ids = {row["id"] for row in self._rows(api.uploads[-1])}
        assert f"fb-{annotated}" not in ids
        assert api.uploads[-1]["path_in_repo"] == api.uploads[0]["path_in_repo"]

    def test_network_error_is_logged_and_retried(self, records, caplog):
        api = FakeHubApi(fail=True)
        backup = FeedbackBackup("compte/retours", SessionLocal, api=api)
        assert backup.push() is False
        assert "impossible" in caplog.text
        api.fail = False
        assert backup.push() is True  # nouvel essai au cycle suivant
