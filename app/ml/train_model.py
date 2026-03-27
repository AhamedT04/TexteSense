from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

DATASET_PATH = Path("data/sentiment_dataset.csv")
MODEL_DIR = Path("saved_models")
MODEL_PATH = MODEL_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"


def build_synthetic_samples() -> pd.DataFrame:
    positive_subjects = [
        "projet",
        "outil",
        "service",
        "cours",
        "application",
        "tableau de bord",
        "rapport",
        "resultat",
    ]
    positive_adjectives = [
        "excellent",
        "utile",
        "fluide",
        "convaincant",
        "fiable",
        "rapide",
        "propre",
        "pratique",
    ]
    negative_subjects = [
        "projet",
        "outil",
        "service",
        "cours",
        "application",
        "tableau de bord",
        "rapport",
        "resultat",
    ]
    negative_adjectives = [
        "horrible",
        "lent",
        "confus",
        "decevant",
        "fragile",
        "inutile",
        "bugge",
        "frustrant",
    ]
    neutral_subjects = [
        "projet",
        "application",
        "serveur",
        "rapport",
        "formulaire",
        "historique",
        "modele",
        "analyse",
    ]
    neutral_actions = [
        "contient trois sections",
        "utilise Python et SQLite",
        "affiche les informations attendues",
        "enregistre une analyse",
        "retourne une prediction",
        "demarre sur le port local",
        "charge la page principale",
        "propose une route API",
    ]
    positive_expressions = [
        "je suis content",
        "je suis tres content",
        "je suis vraiment content",
        "je suis heureux",
        "je suis tres heureux",
        "je suis ravi",
        "je suis tres satisfait",
        "je me sens confiant",
        "cela me rend content",
        "cela me fait plaisir",
    ]
    negative_expressions = [
        "je suis triste",
        "je suis tres triste",
        "je suis vraiment decu",
        "je suis en colere",
        "je suis tres mecontent",
        "je me sens frustre",
        "cela me rend furieux",
        "cela me fait perdre patience",
        "je suis insatisfait",
        "je suis tres decourage",
    ]
    neutral_expressions = [
        "je suis disponible aujourd'hui",
        "je suis present a la reunion",
        "je suis sur le projet aujourd'hui",
        "je consulte le tableau de bord",
        "je regarde les resultats",
        "je viens de recevoir le message",
        "la tache est en cours",
        "le traitement se termine aujourd'hui",
    ]
    time_markers = [
        "aujourd'hui",
        "ce matin",
        "cet apres-midi",
        "en ce moment",
    ]

    rows: list[dict[str, str]] = []

    for subject in positive_subjects:
        for adjective in positive_adjectives:
            rows.append({"text": f"Ce {subject} est {adjective} et tres agreable a utiliser", "label": "positive"})
            rows.append({"text": f"Je trouve ce {subject} {adjective} et vraiment utile", "label": "positive"})

    for subject in negative_subjects:
        for adjective in negative_adjectives:
            rows.append({"text": f"Ce {subject} est {adjective} et difficile a utiliser", "label": "negative"})
            rows.append({"text": f"Je trouve ce {subject} {adjective} et tres penible", "label": "negative"})

    for subject in neutral_subjects:
        for action in neutral_actions:
            rows.append({"text": f"Le {subject} {action}", "label": "neutral"})
            rows.append({"text": f"Cette information indique que le {subject} {action}", "label": "neutral"})

    for expression in positive_expressions:
        rows.append({"text": expression, "label": "positive"})
        for marker in time_markers:
            rows.append({"text": f"{expression} {marker}", "label": "positive"})
            rows.append({"text": f"{expression} et satisfait {marker}", "label": "positive"})

    for expression in negative_expressions:
        rows.append({"text": expression, "label": "negative"})
        for marker in time_markers:
            rows.append({"text": f"{expression} {marker}", "label": "negative"})
            rows.append({"text": f"{expression} et frustre {marker}", "label": "negative"})

    for expression in neutral_expressions:
        rows.append({"text": expression, "label": "neutral"})
        for marker in time_markers:
            rows.append({"text": f"{expression} {marker}", "label": "neutral"})

    return pd.DataFrame(rows)


def train() -> float:
    df = pd.read_csv(DATASET_PATH).dropna()
    df = pd.concat([df, build_synthetic_samples()], ignore_index=True)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        strip_accents="unicode",
    )
    X_train_vectors = vectorizer.fit_transform(X_train)
    X_test_vectors = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1500, class_weight="balanced")
    model.fit(X_train_vectors, y_train)

    predictions = model.predict(X_test_vectors)
    accuracy = accuracy_score(y_test, predictions)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")
    return accuracy


if __name__ == "__main__":
    train()
