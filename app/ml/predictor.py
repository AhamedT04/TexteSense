import logging
import threading
from functools import lru_cache
from pathlib import Path

import joblib

from app.config import get_settings

logger = logging.getLogger(__name__)

HF_LABEL_MAP = {
    "1 star": "negative",
    "2 stars": "negative",
    "3 stars": "neutral",
    "4 stars": "positive",
    "5 stars": "positive",
}


class ModelUnavailableError(RuntimeError):
    """Le modèle Hugging Face n'a pas pu être chargé (non téléchargé, dépendance absente…)."""


# État du chargement HF. `lru_cache` ne mémorise pas les exceptions : sans ce
# cache explicite, chaque requête retenterait un chargement voué à l'échec.
_hf_pipeline = None
_hf_error: ModelUnavailableError | None = None
_hf_lock = threading.Lock()


def _model_paths() -> tuple[Path, Path]:
    settings = get_settings()
    return (
        settings.model_dir / "sentiment_model.pkl",
        settings.model_dir / "vectorizer.pkl",
    )


def load_huggingface_pipeline():
    """Charge le pipeline HF une seule fois ; un échec est mémorisé jusqu'au redémarrage."""
    global _hf_pipeline, _hf_error

    if _hf_pipeline is not None:
        return _hf_pipeline
    if _hf_error is not None:
        raise _hf_error

    with _hf_lock:
        if _hf_pipeline is not None:
            return _hf_pipeline
        if _hf_error is not None:
            raise _hf_error

        settings = get_settings()
        logger.info("Chargement du modèle Hugging Face : %s", settings.hf_model_id)
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                pipeline,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                settings.hf_model_id, local_files_only=True
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                settings.hf_model_id, local_files_only=True
            )
            _hf_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        except Exception as exc:
            _hf_error = ModelUnavailableError(
                f"Modèle Hugging Face indisponible ({type(exc).__name__}: {exc}). "
                "Téléchargez-le puis redémarrez l'application."
            )
            logger.warning("%s Le modèle fallback sklearn sera utilisé.", _hf_error)
            raise _hf_error from exc

        logger.info("Modèle Hugging Face chargé avec succès.")
        return _hf_pipeline


def reset_model_cache() -> None:
    """Oublie les modèles chargés (utile pour les tests ou après un téléchargement)."""
    global _hf_pipeline, _hf_error
    with _hf_lock:
        _hf_pipeline = None
        _hf_error = None
    load_artifacts.cache_clear()


@lru_cache(maxsize=1)
def load_artifacts():
    model_path, vectorizer_path = _model_paths()
    if not model_path.exists() or not vectorizer_path.exists():
        raise FileNotFoundError(
            "Aucun modèle disponible : téléchargez le modèle Hugging Face "
            "ou exécutez `python -m app.ml.train_model`."
        )
    logger.info("Chargement du modèle fallback scikit-learn.")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    logger.info("Modèle fallback chargé avec succès.")
    return model, vectorizer


def is_hf_model_ready() -> bool:
    try:
        load_huggingface_pipeline()
        return True
    except ModelUnavailableError:
        return False


def is_model_ready() -> bool:
    model_path, vectorizer_path = _model_paths()
    if model_path.exists() and vectorizer_path.exists():
        return True
    return is_hf_model_ready()


def _predict_with_huggingface(text: str) -> tuple[str, float]:
    settings = get_settings()
    classifier = load_huggingface_pipeline()
    results = classifier(
        text,
        truncation=True,
        max_length=settings.hf_max_tokens,
        top_k=None,
    )
    # Probabilité de chaque sentiment = somme des probabilités des étoiles correspondantes.
    grouped_scores: dict[str, float] = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
    for item in results:
        mapped = HF_LABEL_MAP.get(item["label"])
        if mapped is not None:
            grouped_scores[mapped] += float(item["score"])

    best_sentiment = max(grouped_scores, key=grouped_scores.get)
    return best_sentiment, grouped_scores[best_sentiment]


def _predict_with_sklearn(text: str) -> tuple[str, float]:
    model, vectorizer = load_artifacts()
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    return str(prediction), float(max(probabilities))


def predict_sentiment(text: str) -> dict[str, float | str | bool]:
    settings = get_settings()

    try:
        sentiment, confidence = _predict_with_huggingface(text)
        source = "HF"
    except ModelUnavailableError:
        # Déjà signalé une fois au chargement : pas de log bruyant à chaque requête.
        sentiment, confidence = _predict_with_sklearn(text)
        source = "sklearn"
    except Exception as exc:
        logger.error(
            "Erreur inattendue avec le modèle HF : %s — bascule sur sklearn.",
            exc,
            exc_info=True,
        )
        sentiment, confidence = _predict_with_sklearn(text)
        source = "sklearn"

    low_confidence = confidence < settings.hf_confidence_threshold
    if low_confidence:
        logger.info(
            "%s : confiance faible (%.4f < %.2f) pour : %.60r",
            source,
            confidence,
            settings.hf_confidence_threshold,
            text,
        )
    else:
        logger.debug("%s → sentiment=%s confidence=%.4f", source, sentiment, confidence)

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
        "low_confidence": low_confidence,
    }
