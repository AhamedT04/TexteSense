from functools import lru_cache
from pathlib import Path

import joblib
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

MODEL_DIR = Path("saved_models")
MODEL_PATH = MODEL_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"
HF_MODEL_ID = "cmarkea/distilcamembert-base-sentiment"
HF_LABEL_MAP = {
    "1 star": "negative",
    "2 stars": "negative",
    "3 stars": "neutral",
    "4 stars": "positive",
    "5 stars": "positive",
}
HF_CONFIDENCE_THRESHOLD = 0.55


@lru_cache(maxsize=1)
def load_huggingface_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        HF_MODEL_ID,
        local_files_only=True,
    )
    return pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
    )


@lru_cache(maxsize=1)
def load_artifacts():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            "Les fichiers du modele sont absents. Executez app/ml/train_model.py."
        )
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def is_model_ready() -> bool:
    if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
        return True

    try:
        load_huggingface_pipeline()
        return True
    except Exception:
        return False


def predict_sentiment(text: str) -> dict[str, float | str]:
    try:
        classifier = load_huggingface_pipeline()
        results = classifier(text, truncation=True, max_length=128, top_k=None)
        grouped_scores = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}

        for item in results:
            sentiment = HF_LABEL_MAP.get(item["label"])
            if sentiment is not None:
                grouped_scores[sentiment] += float(item["score"])

        sentiment = max(grouped_scores, key=grouped_scores.get)
        confidence = grouped_scores[sentiment]
        if confidence >= HF_CONFIDENCE_THRESHOLD:
            return {"sentiment": sentiment, "confidence": round(confidence, 4)}
    except Exception:
        pass

    model, vectorizer = load_artifacts()
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = float(max(probabilities))
    return {"sentiment": str(prediction), "confidence": round(confidence, 4)}
