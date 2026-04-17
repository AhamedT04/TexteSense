"""
Suite de tests pour l'API TextSense.

Chaque groupe de tests couvre un endpoint ou un comportement précis.
Les tests d'analyse réelle de sentiment nécessitent un modèle disponible
(HF ou sklearn) ; ils sont ignorés automatiquement si aucun modèle n'est prêt.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.ml.predictor import is_model_ready

client = TestClient(app)

MODEL_AVAILABLE = is_model_ready()
requires_model = pytest.mark.skipif(
    not MODEL_AVAILABLE,
    reason="Aucun modèle disponible (HF ou sklearn).",
)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_has_required_fields(self):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_ready" in data
        assert "primary_model" in data
        assert "fallback_model" in data
        assert "version" in data

    def test_status_reflects_model_readiness(self):
        data = client.get("/health").json()
        expected_status = "ok" if MODEL_AVAILABLE else "degraded"
        assert data["status"] == expected_status


# ---------------------------------------------------------------------------
# /  (page d'accueil)
# ---------------------------------------------------------------------------


class TestHomePage:
    def test_returns_200(self):
        assert client.get("/").status_code == 200

    def test_contains_app_name(self):
        assert "TextSense" in client.get("/").text


# ---------------------------------------------------------------------------
# /analyze — validation des entrées
# ---------------------------------------------------------------------------


class TestAnalyzeValidation:
    def test_rejects_text_too_short(self):
        response = client.post("/analyze", json={"text": "ab"})
        assert response.status_code == 422

    def test_rejects_empty_text(self):
        response = client.post("/analyze", json={"text": ""})
        assert response.status_code == 422

    def test_rejects_missing_field(self):
        response = client.post("/analyze", json={})
        assert response.status_code == 422

    def test_rejects_text_too_long(self):
        response = client.post("/analyze", json={"text": "a" * 1001})
        assert response.status_code == 422

    def test_rejects_whitespace_only(self):
        # 5 espaces → strip → < 5 caractères
        response = client.post("/analyze", json={"text": "     "})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# /analyze — résultats réels (nécessite un modèle)
# ---------------------------------------------------------------------------


class TestAnalyzeResults:
    @requires_model
    def test_returns_201_on_valid_text(self):
        response = client.post(
            "/analyze",
            json={"text": "Ce projet est vraiment excellent et très utile."},
        )
        assert response.status_code == 201

    @requires_model
    def test_response_has_all_fields(self):
        response = client.post(
            "/analyze",
            json={"text": "Ce projet est vraiment excellent et très utile."},
        )
        data = response.json()
        assert "id" in data
        assert "text" in data
        assert "sentiment" in data
        assert "confidence" in data
        assert "low_confidence" in data
        assert "keywords" in data
        assert "created_at" in data

    @requires_model
    def test_sentiment_is_valid_value(self):
        response = client.post(
            "/analyze",
            json={"text": "Ce projet est vraiment excellent et très utile."},
        )
        assert response.json()["sentiment"] in {"positive", "negative", "neutral"}

    @requires_model
    def test_confidence_is_between_0_and_1(self):
        response = client.post(
            "/analyze",
            json={"text": "Ce projet est vraiment excellent et très utile."},
        )
        confidence = response.json()["confidence"]
        assert 0.0 <= confidence <= 1.0

    @requires_model
    def test_low_confidence_is_boolean(self):
        response = client.post(
            "/analyze",
            json={"text": "Ce projet est vraiment excellent et très utile."},
        )
        assert isinstance(response.json()["low_confidence"], bool)

    @requires_model
    def test_positive_text_predicts_positive(self):
        response = client.post(
            "/analyze",
            json={"text": "Ce service est fantastique, je suis vraiment très satisfait du résultat."},
        )
        assert response.json()["sentiment"] == "positive"

    @requires_model
    def test_negative_text_predicts_negative(self):
        response = client.post(
            "/analyze",
            json={"text": "Ce produit est horrible et complètement inutile, je suis très déçu."},
        )
        assert response.json()["sentiment"] == "negative"

    @requires_model
    def test_keywords_are_list_of_strings(self):
        response = client.post(
            "/analyze",
            json={"text": "Ce projet utilise Python et FastAPI pour analyser les sentiments."},
        )
        keywords = response.json()["keywords"]
        assert isinstance(keywords, list)
        assert all(isinstance(k, str) for k in keywords)

    @requires_model
    def test_keywords_preserve_french_accents(self):
        response = client.post(
            "/analyze",
            json={"text": "L'évaluation de qualité révèle des résultats intéressants."},
        )
        keywords = response.json()["keywords"]
        # Les accents ne doivent pas être encodés en \uXXXX
        joined = " ".join(keywords)
        assert "\\u" not in joined


# ---------------------------------------------------------------------------
# /history
# ---------------------------------------------------------------------------


class TestHistory:
    @requires_model
    def test_returns_list(self):
        response = client.get("/history")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    @requires_model
    def test_pagination_limit(self):
        # Soumet 3 analyses puis vérifie la limite
        for i in range(3):
            client.post(
                "/analyze",
                json={"text": f"Texte de test numéro {i} pour la pagination de l'historique."},
            )
        response = client.get("/history?limit=2")
        assert response.status_code == 200
        assert len(response.json()) <= 2

    @requires_model
    def test_pagination_skip(self):
        response_full = client.get("/history?limit=10&skip=0")
        response_skipped = client.get("/history?limit=10&skip=1")
        full = response_full.json()
        skipped = response_skipped.json()
        if len(full) > 1:
            assert full[1]["id"] == skipped[0]["id"]

    def test_invalid_skip_rejected(self):
        response = client.get("/history?skip=-1")
        assert response.status_code == 422

    @requires_model
    def test_ordered_by_most_recent(self):
        for _ in range(2):
            client.post(
                "/analyze",
                json={"text": "Texte supplémentaire pour vérifier l'ordre des résultats."},
            )
        records = client.get("/history?limit=10").json()
        if len(records) >= 2:
            from datetime import datetime
            dates = [datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")) for r in records]
            assert dates == sorted(dates, reverse=True)


# ---------------------------------------------------------------------------
# /analysis/{id}
# ---------------------------------------------------------------------------


class TestGetAnalysis:
    @requires_model
    def test_returns_existing_analysis(self):
        created = client.post(
            "/analyze",
            json={"text": "Test de récupération d'une analyse par identifiant."},
        ).json()
        response = client.get(f"/analysis/{created['id']}")
        assert response.status_code == 200
        assert response.json()["id"] == created["id"]

    def test_returns_404_for_unknown_id(self):
        response = client.get("/analysis/999999")
        assert response.status_code == 404

    def test_returns_422_for_invalid_id(self):
        response = client.get("/analysis/abc")
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# DELETE /history/{id}
# ---------------------------------------------------------------------------


class TestDeleteAnalysis:
    @requires_model
    def test_deletes_existing_analysis(self):
        created = client.post(
            "/analyze",
            json={"text": "Analyse destinée à être supprimée immédiatement après création."},
        ).json()
        delete_response = client.delete(f"/history/{created['id']}")
        assert delete_response.status_code == 200
        # Vérification : l'entrée n'existe plus
        get_response = client.get(f"/analysis/{created['id']}")
        assert get_response.status_code == 404

    def test_returns_404_for_unknown_id(self):
        response = client.delete("/history/999999")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# /stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_returns_200(self):
        assert client.get("/stats").status_code == 200

    def test_has_required_fields(self):
        data = client.get("/stats").json()
        assert "total" in data
        assert "positive" in data
        assert "negative" in data
        assert "neutral" in data

    def test_counts_are_non_negative(self):
        data = client.get("/stats").json()
        assert data["total"] >= 0
        assert data["positive"] >= 0
        assert data["negative"] >= 0
        assert data["neutral"] >= 0

    @requires_model
    def test_total_equals_sum_of_sentiments(self):
        data = client.get("/stats").json()
        assert data["total"] == data["positive"] + data["negative"] + data["neutral"]

    @requires_model
    def test_stats_update_after_analysis(self):
        before = client.get("/stats").json()["total"]
        client.post(
            "/analyze",
            json={"text": "Nouvelle analyse pour vérifier la mise à jour des statistiques."},
        )
        after = client.get("/stats").json()["total"]
        assert after == before + 1


# ---------------------------------------------------------------------------
# Extracteur de mots-clés
# ---------------------------------------------------------------------------


class TestKeywordExtractor:
    def test_filters_stopwords(self):
        from app.ml.keyword_extractor import extract_keywords
        keywords = extract_keywords("Je suis très content de ce résultat.")
        stopwords_found = {"suis", "très", "tres", "de", "ce"} & set(keywords)
        assert not stopwords_found, f"Stopwords trouvés dans les mots-clés : {stopwords_found}"

    def test_returns_at_most_limit(self):
        from app.ml.keyword_extractor import extract_keywords
        keywords = extract_keywords("Python FastAPI SQLite analyse sentiment", limit=3)
        assert len(keywords) <= 3

    def test_returns_empty_for_only_stopwords(self):
        from app.ml.keyword_extractor import extract_keywords
        keywords = extract_keywords("je tu il elle nous")
        assert keywords == []

    def test_minimum_token_length(self):
        from app.ml.keyword_extractor import extract_keywords
        # "ab" fait 2 caractères → doit être filtré
        keywords = extract_keywords("ab abc abcd efgh")
        assert "ab" not in keywords

    def test_preserves_french_accents(self):
        from app.ml.keyword_extractor import extract_keywords
        keywords = extract_keywords("L'évaluation révèle des résultats intéressants.")
        assert any("é" in k or "è" in k or "ê" in k for k in keywords)
