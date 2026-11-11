"""Entraîne le modèle fallback TF-IDF + régression logistique.

Usage (depuis la racine du projet) :
    python -m app.ml.train_model
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    # Permet aussi `python app/ml/train_model.py` : rend le paquet `app` importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from app.config import get_settings


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


def _build_pipeline() -> tuple[TfidfVectorizer, LogisticRegression]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        strip_accents="unicode",
    )
    model = LogisticRegression(max_iter=1500, class_weight="balanced")
    return vectorizer, model


def train() -> float:
    """Entraîne, évalue sur des phrases réelles jamais vues, puis sauvegarde.

    Les phrases synthétiques sont générées à partir de gabarits : si elles se
    retrouvaient dans le jeu de test, des quasi-doublons du jeu d'entraînement
    gonfleraient artificiellement le score. Le test se fait donc uniquement sur
    une partie du CSV réel, et les phrases synthétiques servent seulement à
    l'entraînement.
    """
    settings = get_settings()
    model_path = settings.model_dir / "sentiment_model.pkl"
    vectorizer_path = settings.model_dir / "vectorizer.pkl"

    real = pd.read_csv(settings.dataset_path).dropna()
    synthetic = build_synthetic_samples()

    real_train, real_test = train_test_split(
        real,
        test_size=0.2,
        random_state=42,
        stratify=real["label"],
    )
    train_df = pd.concat([real_train, synthetic], ignore_index=True)

    vectorizer, model = _build_pipeline()
    model.fit(vectorizer.fit_transform(train_df["text"]), train_df["label"])
    predictions = model.predict(vectorizer.transform(real_test["text"]))
    macro_f1 = f1_score(real_test["label"], predictions, average="macro")

    print(f"Évaluation sur {len(real_test)} phrases réelles non vues à l'entraînement")
    print(f"(petit échantillon : score indicatif, à confirmer sur un vrai jeu d'évaluation)")
    print(f"Macro-F1 : {macro_f1:.4f}")
    print(classification_report(real_test["label"], predictions, zero_division=0))

    # Modèle final : réentraîné sur toutes les données disponibles.
    vectorizer, model = _build_pipeline()
    all_df = pd.concat([real, synthetic], ignore_index=True)
    model.fit(vectorizer.fit_transform(all_df["text"]), all_df["label"])

    settings.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Modèle sauvegardé : {model_path}")
    print(f"Vectoriseur sauvegardé : {vectorizer_path}")
    return macro_f1


if __name__ == "__main__":
    train()
