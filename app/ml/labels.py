"""Étiquettes de sentiment et conversion des sorties de modèles.

Deux familles de modèles sont supportées :

* modèles « étoiles » (ex. ``cmarkea/distilcamembert-base-sentiment``) qui
  prédisent une note de 1 à 5 : on regroupe 1-2★ en négatif, 3★ en neutre,
  4-5★ en positif ;
* modèles « 3 classes » (ex. le modèle fine-tuné du projet) qui prédisent
  directement ``negative`` / ``neutral`` / ``positive``.

La conversion est exprimée comme une matrice ``M`` (nb_labels_modèle × 3) :
``probas_sentiment = probas_modèle @ M``. Une étiquette inconnue lève une
erreur explicite plutôt que d'être ignorée silencieusement.
"""

from collections.abc import Mapping

import numpy as np

SENTIMENTS: tuple[str, ...] = ("negative", "neutral", "positive")
SENTIMENT_INDEX: dict[str, int] = {name: i for i, name in enumerate(SENTIMENTS)}

_LABEL_TO_SENTIMENT: dict[str, str] = {
    "1 star": "negative",
    "2 stars": "negative",
    "3 stars": "neutral",
    "4 stars": "positive",
    "5 stars": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
}


def label_to_sentiment(label: str) -> str:
    try:
        return _LABEL_TO_SENTIMENT[label.strip().lower()]
    except KeyError as exc:
        raise ValueError(
            f"Étiquette de modèle inconnue : {label!r}. "
            f"Étiquettes supportées : {sorted(_LABEL_TO_SENTIMENT)}"
        ) from exc


def build_mapping_matrix(id2label: Mapping[int, str]) -> np.ndarray:
    """Matrice (nb_labels × 3) qui convertit les probabilités du modèle en sentiments."""
    matrix = np.zeros((len(id2label), len(SENTIMENTS)), dtype=np.float32)
    for index, label in id2label.items():
        matrix[int(index), SENTIMENT_INDEX[label_to_sentiment(label)]] = 1.0
    return matrix


def is_star_model(id2label: Mapping[int, str]) -> bool:
    return any("star" in label.lower() for label in id2label.values())
