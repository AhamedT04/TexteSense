"""Métriques de classification, intervalles de confiance et calibration."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from app.ml.labels import SENTIMENTS, SENTIMENT_INDEX


def predictions_from_proba(probabilities: np.ndarray) -> list[str]:
    return [SENTIMENTS[i] for i in probabilities.argmax(axis=1)]


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """ECE : écart moyen (pondéré) entre confiance annoncée et exactitude réelle.

    Un modèle bien calibré qui annonce 80 % de confiance a raison 80 % du temps
    (ECE proche de 0).
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, list[float]]:
    """Intervalle de confiance à 95 % par bootstrap (rééchantillonnage avec remise).

    Sur 360 exemples, un écart de 1-2 points de macro-F1 peut être du bruit :
    l'intervalle indique la marge d'incertitude due à la taille du jeu.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accuracies, macro_f1s = [], []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        accuracies.append(accuracy_score(y_true[idx], y_pred[idx]))
        # Sans `labels=` : la moyenne porte sur les classes présentes dans
        # l'échantillon. Une classe absente (possible sur un petit jeu) ne doit
        # pas compter comme un F1 nul.
        macro_f1s.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {
        "accuracy": [float(v) for v in np.percentile(accuracies, q)],
        "macro_f1": [float(v) for v in np.percentile(macro_f1s, q)],
    }


def classification_metrics(
    y_true: Sequence[str],
    probabilities: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(predictions_from_proba(probabilities))
    labels = list(SENTIMENTS)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=labels, zero_division=0
    )
    confidence = probabilities.max(axis=1)
    correct = (y_true_arr == y_pred_arr).astype(float)

    return {
        "n": int(len(y_true_arr)),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0)),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix": {
            "labels": labels,
            "rows_true_cols_pred": confusion_matrix(y_true_arr, y_pred_arr, labels=labels).tolist(),
        },
        "ece": expected_calibration_error(confidence, correct),
        "mean_confidence": float(confidence.mean()) if len(confidence) else 0.0,
        "ci95": bootstrap_ci(y_true_arr, y_pred_arr, n_resamples=n_bootstrap, seed=seed),
    }


def accuracy_by_group(predictions: pd.DataFrame, column: str) -> dict[str, dict]:
    """Exactitude par sous-groupe (ex. phénomène linguistique : négation, ironie…)."""
    result = {}
    for group, part in predictions.groupby(column, sort=True):
        result[str(group)] = {
            "n": int(len(part)),
            "accuracy": float((part["label"] == part["pred"]).mean()),
        }
    return result


def selective_accuracy(
    y_true: Sequence[str], probabilities: np.ndarray, thresholds: Sequence[float]
) -> list[dict]:
    """Pour chaque seuil : part des prédictions jugées fiables (couverture) et leur exactitude."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(predictions_from_proba(probabilities))
    confidence = probabilities.max(axis=1)
    rows = []
    for threshold in thresholds:
        kept = confidence >= threshold
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": float(kept.mean()),
                "accuracy": float((y_true_arr[kept] == y_pred_arr[kept]).mean()) if kept.any() else float("nan"),
            }
        )
    return rows


def labels_to_indices(labels: Sequence[str]) -> np.ndarray:
    return np.array([SENTIMENT_INDEX[label] for label in labels])
