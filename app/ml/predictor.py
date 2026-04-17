import logging
from functools import lru_cache
from pathlib import Path

import joblib
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

from app.config import get_settings

logger = logging.getLogger(__name__)

HF_LABEL_MAP = {
    "1 star": "negative",
    "2 stars": "negative",
    "3 stars": "neutral",
    "4 stars": "positive",
    "5 stars": "positive",
}


def _model_paths() -> tuple[Path, Path]:
    settings = get_settings()
    return (
        settings.model_dir / "sentiment_model.pkl",
        settings.model_dir / "vectorizer.pkl",
    )


# Expose l'ID du modèle HF pour main.py
@property
def HF_MODEL_ID() -> str:  # noqa: N802
    return get_settings().hf_model_id


# Compatibilité import direct
HF_MODEL_ID = get_settings().hf_model_id


@lru_cache(maxsize=1)
def load_huggingface_pipeline():
    settings = get_settings()
    logger.info("Chargement du modèle Hugging Face : %s", settings.hf_model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        settings.hf_model_id, local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        settings.hf_model_id,
        local_files_only=True,
    )
    clf = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
    )
    logger.info("Modèle Hugging Face chargé avec succès.")
    return clf


@lru_cache(maxsize=1)
def load_artifacts():
    model_path, vectorizer_path = _model_paths()
    if not model_path.exists() or not vectorizer_path.exists():
        raise FileNotFoundError(
            "Fichiers modèle absents. Exécutez app/ml/train_model.py."
        )
    logger.info("Chargement du modèle fallback scikit-learn.")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    logger.info("Modèle fallback chargé avec succès.")
    return model, vectorizer


def is_model_ready() -> bool:
    model_path, vectorizer_path = _model_paths()
    if model_path.exists() and vectorizer_path.exists():
        return True
    try:
        load_huggingface_pipeline()
        return True
    except Exception:
        return False


def predict_sentiment(text: str) -> dict[str, float | str | bool]:
    settings = get_settings()

    # --- Tentative avec le modèle Hugging Face ---
    try:
        classifier = load_huggingface_pipeline()
        results = classifier(
            text,
            truncation=True,
            max_length=settings.hf_max_tokens,
            top_k=None,
        )
        grouped_scores: dict[str, float] = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}

        for item in results:
            mapped = HF_LABEL_MAP.get(item["label"])
            if mapped is not None:
                grouped_scores[mapped] += float(item["score"])

        best_sentiment = max(grouped_scores, key=grouped_scores.get)
        confidence = grouped_scores[best_sentiment]

        if confidence >= settings.hf_confidence_threshold:
            logger.debug(
                "HF → sentiment=%s confidence=%.4f", best_sentiment, confidence
            )
            return {
                "sentiment": best_sentiment,
                "confidence": round(confidence, 4),
                "low_confidence": False,
            }

        # Confiance insuffisante : on renvoie quand même le résultat HF
        # avec le flag low_confidence plutôt que de se rabattre sur sklearn
        logger.info(
            "HF confidence faible (%.4f < %.2f) pour : %.60r — résultat retourné avec flag low_confidence.",
            confidence,
            settings.hf_confidence_threshold,
            text,
        )
        return {
            "sentiment": best_sentiment,
            "confidence": round(confidence, 4),
            "low_confidence": True,
        }

    except FileNotFoundError:
        logger.warning(
            "Modèle HF introuvable (fichiers non téléchargés). Bascule sur sklearn."
        )
    except Exception as exc:
        logger.error(
            "Erreur inattendue avec le modèle HF : %s — bascule sur sklearn.", exc,
            exc_info=True,
        )

    # --- Fallback scikit-learn ---
    logger.info("Utilisation du modèle fallback scikit-learn.")
    model, vectorizer = load_artifacts()
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = float(max(probabilities))
    logger.debug(
        "sklearn → sentiment=%s confidence=%.4f", prediction, confidence
    )
    return {
        "sentiment": str(prediction),
        "confidence": round(confidence, 4),
        "low_confidence": confidence < settings.hf_confidence_threshold,
    }
