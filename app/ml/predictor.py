"""Prédiction de sentiment pour l'API.

Ordre de priorité :
1. le modèle principal (Transformers ou ONNX selon ``MODEL_BACKEND``) ;
2. à défaut, le modèle fallback TF-IDF + régression logistique.

Les modèles sont chargés une seule fois. Un échec de chargement du modèle
principal est mémorisé (jusqu'au redémarrage ou à :func:`reset_model_cache`)
pour ne pas retenter un chargement voué à l'échec à chaque requête.
"""

import logging
import threading
from functools import lru_cache
from typing import TypedDict

from app.config import get_settings
from app.ml.classifiers import (
    HuggingFaceClassifier,
    OnnxClassifier,
    SentimentClassifier,
    SklearnClassifier,
)
from app.ml.labels import SENTIMENTS

logger = logging.getLogger(__name__)


class Prediction(TypedDict):
    sentiment: str
    confidence: float
    low_confidence: bool
    model: str


class ModelUnavailableError(RuntimeError):
    """Le modèle principal n'a pas pu être chargé (non téléchargé, dépendance absente…)."""


_primary: SentimentClassifier | None = None
_primary_error: ModelUnavailableError | None = None
_lock = threading.Lock()


def _build_primary_classifier() -> SentimentClassifier:
    settings = get_settings()
    if settings.model_backend == "onnx":
        if settings.onnx_model_dir is None:
            raise ValueError("MODEL_BACKEND=onnx nécessite ONNX_MODEL_DIR.")
        return OnnxClassifier(
            settings.onnx_model_dir,
            onnx_file=settings.onnx_model_file,
            max_length=settings.hf_max_tokens,
        )
    return HuggingFaceClassifier(
        settings.hf_model_id,
        max_length=settings.hf_max_tokens,
        local_files_only=True,
    )


def load_primary_classifier() -> SentimentClassifier:
    """Charge le modèle principal une seule fois ; un échec est mémorisé."""
    global _primary, _primary_error

    if _primary is not None:
        return _primary
    if _primary_error is not None:
        raise _primary_error

    with _lock:
        if _primary is not None:
            return _primary
        if _primary_error is not None:
            raise _primary_error

        settings = get_settings()
        target = settings.onnx_model_dir if settings.model_backend == "onnx" else settings.hf_model_id
        logger.info("Chargement du modèle principal (%s) : %s", settings.model_backend, target)
        try:
            _primary = _build_primary_classifier()
        except Exception as exc:
            _primary_error = ModelUnavailableError(
                f"Modèle principal indisponible ({type(exc).__name__}: {exc}). "
                "Téléchargez-le ou corrigez la configuration, puis redémarrez l'application."
            )
            logger.warning("%s Le modèle fallback sklearn sera utilisé.", _primary_error)
            raise _primary_error from exc

        logger.info("Modèle principal chargé : %s", _primary.name)
        return _primary


@lru_cache(maxsize=1)
def load_fallback_classifier() -> SklearnClassifier:
    logger.info("Chargement du modèle fallback scikit-learn.")
    return SklearnClassifier(get_settings().model_dir)


def reset_model_cache() -> None:
    """Oublie les modèles chargés (tests, ou après un téléchargement)."""
    global _primary, _primary_error
    with _lock:
        _primary = None
        _primary_error = None
    load_fallback_classifier.cache_clear()


def is_primary_model_ready() -> bool:
    try:
        load_primary_classifier()
        return True
    except ModelUnavailableError:
        return False


def _fallback_files_exist() -> bool:
    model_dir = get_settings().model_dir
    return (model_dir / "sentiment_model.pkl").exists() and (model_dir / "vectorizer.pkl").exists()


def is_model_ready() -> bool:
    return _fallback_files_exist() or is_primary_model_ready()


def _select_classifier() -> SentimentClassifier:
    try:
        return load_primary_classifier()
    except ModelUnavailableError:
        # Déjà signalé une fois au chargement : pas de log à chaque requête.
        return load_fallback_classifier()


def confidence_threshold_for(classifier: SentimentClassifier) -> float:
    """Seuil appliqué : CONFIDENCE_THRESHOLD s'il est défini, sinon celui calibré pour le modèle."""
    settings = get_settings()
    if "confidence_threshold" in settings.model_fields_set:
        return settings.confidence_threshold
    if classifier.recommended_threshold is not None:
        return float(classifier.recommended_threshold)
    return settings.confidence_threshold


def predict_sentiment(text: str) -> Prediction:
    classifier = _select_classifier()
    try:
        probabilities = classifier.predict_proba([text])[0]
    except Exception:
        if isinstance(classifier, SklearnClassifier):
            raise
        logger.error(
            "Erreur inattendue avec le modèle principal — bascule sur sklearn.", exc_info=True
        )
        classifier = load_fallback_classifier()
        probabilities = classifier.predict_proba([text])[0]

    best = int(probabilities.argmax())
    sentiment, confidence = SENTIMENTS[best], float(probabilities[best])
    threshold = confidence_threshold_for(classifier)
    low_confidence = confidence < threshold

    if low_confidence:
        logger.info(
            "%s : confiance faible (%.4f < %.2f) pour : %.60r",
            classifier.name,
            confidence,
            threshold,
            text,
        )
    else:
        logger.debug("%s → sentiment=%s confidence=%.4f", classifier.name, sentiment, confidence)

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
        "low_confidence": low_confidence,
        "model": classifier.name,
    }
