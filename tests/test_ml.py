"""Tests des briques ML : étiquettes, métriques, jeu d'évaluation, calibration."""

import numpy as np
import pandas as pd
import pytest

from app.ml.labels import SENTIMENTS, build_mapping_matrix, label_to_sentiment


class TestLabelMapping:
    def test_star_labels_are_grouped(self):
        id2label = {0: "1 star", 1: "2 stars", 2: "3 stars", 3: "4 stars", 4: "5 stars"}
        matrix = build_mapping_matrix(id2label)
        probabilities = np.array([[0.1, 0.2, 0.3, 0.25, 0.15]])
        np.testing.assert_allclose(probabilities @ matrix, [[0.3, 0.3, 0.4]])

    def test_three_class_labels_are_identity(self):
        matrix = build_mapping_matrix({0: "negative", 1: "neutral", 2: "positive"})
        np.testing.assert_array_equal(matrix, np.eye(3))

    def test_three_class_labels_in_another_order(self):
        matrix = build_mapping_matrix({0: "positive", 1: "negative", 2: "neutral"})
        np.testing.assert_allclose(np.array([[0.7, 0.2, 0.1]]) @ matrix, [[0.2, 0.1, 0.7]])

    def test_unknown_label_fails_loudly(self):
        with pytest.raises(ValueError, match="inconnue"):
            label_to_sentiment("LABEL_0")

    def test_sentiment_order_is_stable(self):
        assert SENTIMENTS == ("negative", "neutral", "positive")


class TestMetrics:
    def test_perfect_predictions(self):
        from textsense_ml.metrics import classification_metrics

        probabilities = np.eye(3)[[0, 1, 2, 0, 1, 2]]
        metrics = classification_metrics(["negative", "neutral", "positive"] * 2, probabilities, n_bootstrap=50)
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert metrics["confusion_matrix"]["rows_true_cols_pred"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        assert metrics["ci95"]["macro_f1"] == [1.0, 1.0]

    def test_confusion_matrix_rows_are_truth(self):
        from textsense_ml.metrics import classification_metrics

        # vérité « neutral », prédiction « positive »
        metrics = classification_metrics(["neutral"], np.array([[0.1, 0.2, 0.7]]), n_bootstrap=10)
        assert metrics["confusion_matrix"]["rows_true_cols_pred"][1] == [0, 0, 1]
        assert metrics["per_class"]["neutral"]["recall"] == 0.0

    def test_ece_is_zero_when_confidence_matches_accuracy(self):
        from textsense_ml.metrics import expected_calibration_error

        confidence = np.array([0.75] * 4)
        correct = np.array([1, 1, 1, 0])
        assert expected_calibration_error(confidence, correct) == pytest.approx(0.0)

    def test_ece_detects_overconfidence(self):
        from textsense_ml.metrics import expected_calibration_error

        confidence = np.array([0.99] * 4)
        correct = np.array([1, 0, 1, 0])
        assert expected_calibration_error(confidence, correct) == pytest.approx(0.49)

    def test_selective_accuracy(self):
        from textsense_ml.metrics import selective_accuracy

        probabilities = np.array([[0.9, 0.05, 0.05], [0.4, 0.35, 0.25]])
        rows = selective_accuracy(["negative", "neutral"], probabilities, [0.3, 0.5])
        assert rows[0] == {"threshold": 0.3, "coverage": 1.0, "accuracy": 0.5}
        assert rows[1] == {"threshold": 0.5, "coverage": 0.5, "accuracy": 1.0}

    def test_accuracy_by_group(self):
        from textsense_ml.metrics import accuracy_by_group

        df = pd.DataFrame(
            {"label": ["a", "a", "b"], "pred": ["a", "b", "b"], "phenomenon": ["irony", "irony", "negation"]}
        )
        assert accuracy_by_group(df, "phenomenon") == {
            "irony": {"n": 2, "accuracy": 0.5},
            "negation": {"n": 1, "accuracy": 1.0},
        }


class TestCalibration:
    def test_picks_lowest_threshold_reaching_target(self):
        from textsense_ml.calibrate import choose_threshold

        curve = [
            {"threshold": 0.4, "coverage": 1.0, "accuracy": 0.70},
            {"threshold": 0.6, "coverage": 0.8, "accuracy": 0.86},
            {"threshold": 0.8, "coverage": 0.5, "accuracy": 0.95},
        ]
        assert choose_threshold(curve, 0.85)["threshold"] == 0.6
        assert choose_threshold(curve, 0.99) is None


class TestEvaluationSet:
    """Garde-fous sur le jeu de test : toute modification doit être volontaire (nouvelle version)."""

    @pytest.fixture(scope="class")
    def eval_df(self):
        from textsense_ml.data import load_textsense_eval

        return load_textsense_eval("v1")

    def test_size_and_balance(self, eval_df):
        assert len(eval_df) == 360
        assert eval_df["label"].value_counts().to_dict() == {"positive": 120, "negative": 120, "neutral": 120}

    def test_unique_ids_and_texts(self, eval_df):
        assert eval_df["id"].is_unique
        assert eval_df["text"].is_unique

    def test_metadata_columns_are_filled(self, eval_df):
        assert eval_df["phenomenon"].notna().all()
        assert eval_df["domain"].notna().all()

    def test_no_overlap_with_fallback_training_data(self, eval_df):
        import re

        from app.config import get_settings

        def normalize(text: str) -> str:
            return re.sub(r"\W+", " ", text.lower()).strip()

        train = {normalize(t) for t in pd.read_csv(get_settings().dataset_path)["text"]}
        assert not {normalize(t) for t in eval_df["text"]} & train

    def test_invalid_label_is_rejected(self):
        from textsense_ml.data import _validate

        with pytest.raises(ValueError, match="étiquettes inconnues"):
            _validate(pd.DataFrame({"id": [1], "text": ["x"], "label": ["joie"]}), "test")


class TestClassifiers:
    def test_sklearn_classifier_columns_follow_sentiment_order(self, tmp_path):
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        from app.ml.classifiers import SklearnClassifier

        texts = ["super génial", "horrible nul", "réunion jeudi", "excellent top", "mauvais pire", "bus 42"]
        labels = ["positive", "negative", "neutral", "positive", "negative", "neutral"]
        vectorizer = TfidfVectorizer().fit(texts)
        model = LogisticRegression().fit(vectorizer.transform(texts), labels)
        joblib.dump(model, tmp_path / "sentiment_model.pkl")
        joblib.dump(vectorizer, tmp_path / "vectorizer.pkl")

        probabilities = SklearnClassifier(tmp_path).predict_proba(["super génial"])
        assert probabilities.shape == (1, 3)
        assert SENTIMENTS[int(probabilities.argmax())] == "positive"

    def test_load_classifier_rejects_unknown_spec(self):
        from app.ml.classifiers import load_classifier

        with pytest.raises(ValueError, match="invalide"):
            load_classifier("tensorflow:modele")


class TestConfidenceThreshold:
    """Ordre de priorité : CONFIDENCE_THRESHOLD explicite > seuil calibré du modèle > défaut."""

    class _Fake:
        name = "fake"

        def __init__(self, threshold):
            self.recommended_threshold = threshold

    def test_uses_model_threshold_when_not_configured(self, monkeypatch):
        from app.config import Settings
        from app.ml import predictor

        monkeypatch.delenv("CONFIDENCE_THRESHOLD", raising=False)
        monkeypatch.delenv("HF_CONFIDENCE_THRESHOLD", raising=False)
        monkeypatch.setattr(predictor, "get_settings", lambda: Settings(_env_file=None))
        assert predictor.confidence_threshold_for(self._Fake(0.81)) == 0.81

    def test_falls_back_to_default(self, monkeypatch):
        from app.config import Settings
        from app.ml import predictor

        monkeypatch.delenv("CONFIDENCE_THRESHOLD", raising=False)
        monkeypatch.delenv("HF_CONFIDENCE_THRESHOLD", raising=False)
        monkeypatch.setattr(predictor, "get_settings", lambda: Settings(_env_file=None))
        assert predictor.confidence_threshold_for(self._Fake(None)) == 0.55

    def test_explicit_setting_wins(self, monkeypatch):
        from app.config import Settings
        from app.ml import predictor

        monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.7")
        monkeypatch.setattr(predictor, "get_settings", lambda: Settings(_env_file=None))
        assert predictor.confidence_threshold_for(self._Fake(0.81)) == 0.7

    def test_legacy_variable_name_still_works(self, monkeypatch):
        from app.config import Settings

        monkeypatch.delenv("CONFIDENCE_THRESHOLD", raising=False)
        monkeypatch.setenv("HF_CONFIDENCE_THRESHOLD", "0.6")
        assert Settings(_env_file=None).confidence_threshold == 0.6
