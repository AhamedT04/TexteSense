"""Chargement des jeux de données, avec versions figées pour la reproductibilité.

Jeux disponibles (voir :data:`DATASETS`) :

* ``textsense_eval_v1`` — 360 phrases annotées du projet (``data/eval/``),
  **réservé à l'évaluation finale** : jamais utilisé pour entraîner ni pour
  régler un hyperparamètre ;
* ``tweets_fr_train`` / ``tweets_fr_validation`` / ``tweets_fr_test`` — partie
  française de ``cardiffnlp/tweet_sentiment_multilingual`` (3 classes
  équilibrées), téléchargée à une révision précise du Hub Hugging Face.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from app.config import BASE_DIR
from app.ml.labels import SENTIMENTS

EVAL_DIR = BASE_DIR / "data" / "eval"

TWEETS_REPO_ID = "cardiffnlp/tweet_sentiment_multilingual"
# Révision figée (commit du dépôt de données sur le Hub) : garantit que tout
# le monde entraîne et évalue sur exactement les mêmes exemples.
TWEETS_REVISION = "14b13edfbc4046892f6011d114c29c0f83170589"
TWEETS_LABELS = {0: "negative", 1: "neutral", 2: "positive"}

REQUIRED_COLUMNS = ("id", "text", "label")


def _validate(df: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{name} : colonnes manquantes {missing}")
    unknown = set(df["label"]) - set(SENTIMENTS)
    if unknown:
        raise ValueError(f"{name} : étiquettes inconnues {sorted(unknown)}")
    if df["id"].duplicated().any():
        raise ValueError(f"{name} : identifiants dupliqués")
    return df.reset_index(drop=True)


def load_textsense_eval(version: str = "v1") -> pd.DataFrame:
    path = EVAL_DIR / f"textsense_eval_{version}.csv"
    return _validate(pd.read_csv(path), path.name)


def load_tweets_fr(split: str) -> pd.DataFrame:
    """Tweets français annotés (negative / neutral / positive)."""
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"split inconnu : {split}")
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=TWEETS_REPO_ID,
        repo_type="dataset",
        filename=f"data/french/{split}.jsonl",
        revision=TWEETS_REVISION,
    )
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    df = pd.DataFrame(
        {
            "id": [f"tweets-fr-{split}-{i:04d}" for i in range(len(rows))],
            "text": [r["text"] for r in rows],
            "label": [TWEETS_LABELS[int(r["label"])] for r in rows],
        }
    )
    return _validate(df, f"tweets_fr_{split}")


DATASETS: dict[str, Callable[[], pd.DataFrame]] = {
    "textsense_eval_v1": lambda: load_textsense_eval("v1"),
    "tweets_fr_train": lambda: load_tweets_fr("train"),
    "tweets_fr_validation": lambda: load_tweets_fr("validation"),
    "tweets_fr_test": lambda: load_tweets_fr("test"),
}

# Jeux dont le texte peut être recopié dans les rapports versionnés.
# Les tweets appartiennent à un jeu tiers : on ne les redistribue pas.
REDISTRIBUTABLE = {"textsense_eval_v1"}


def load_dataset(name: str) -> pd.DataFrame:
    try:
        return DATASETS[name]()
    except KeyError as exc:
        raise ValueError(f"Jeu inconnu : {name}. Disponibles : {sorted(DATASETS)}") from exc
