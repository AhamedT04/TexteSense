"""Choisit le seuil ``CONFIDENCE_THRESHOLD`` à partir du jeu de validation.

Principe (classification sélective) : une prédiction dont la confiance est
sous le seuil est signalée « confiance faible ». On cherche le **plus petit
seuil** tel que les prédictions au-dessus atteignent l'exactitude visée : on
signale ainsi le moins de résultats possible tout en garantissant la fiabilité
des autres.

Le seuil est choisi sur la **validation** (jamais sur un jeu de test), à partir
des prédictions produites par ``textsense_ml.evaluate``.

Usage ::

    python -m textsense_ml.calibrate --run distilcamembert-3c-starinit-s42 --target-accuracy 0.85
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import BASE_DIR
from app.ml.labels import SENTIMENTS
from textsense_ml.metrics import selective_accuracy
from textsense_ml.plots import plot_selective_curve
from textsense_ml.tracking import run_metadata, write_json

EVAL_DIR = BASE_DIR / "reports" / "eval"
OUT_DIR = BASE_DIR / "reports" / "calibration"
THRESHOLDS = np.round(np.arange(0.34, 1.0, 0.01), 2)


def load_validation_predictions(run: str, dataset: str) -> tuple[list[str], np.ndarray]:
    path = EVAL_DIR / run / dataset / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable : lancez d'abord textsense_ml.evaluate sur {dataset}.")
    df = pd.read_csv(path)
    probabilities = df[[f"p_{s}" for s in SENTIMENTS]].to_numpy()
    return df["label"].tolist(), probabilities


def choose_threshold(curve: list[dict], target_accuracy: float) -> dict | None:
    for row in curve:  # seuils croissants : le premier qui atteint la cible maximise la couverture
        if row["coverage"] > 0 and row["accuracy"] >= target_accuracy:
            return row
    return None


def write_threshold_to_model(model_dir: Path, threshold: float) -> None:
    """Enregistre le seuil dans le config.json du modèle (lu par l'API au chargement)."""
    import json

    from app.ml.classifiers import THRESHOLD_CONFIG_KEY

    model_dir = model_dir if model_dir.is_absolute() else BASE_DIR / model_dir
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config[THRESHOLD_CONFIG_KEY] = threshold
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Seuil {threshold:.2f} écrit dans {config_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="exécution évaluée à calibrer")
    parser.add_argument("--compare", nargs="*", default=[], help="autres exécutions à tracer pour comparaison")
    parser.add_argument("--dataset", default="tweets_fr_validation")
    parser.add_argument("--target-accuracy", type=float, default=0.85)
    parser.add_argument(
        "--write-to-model",
        type=Path,
        help="dossier(s) de modèle dont le config.json recevra le seuil retenu",
        nargs="*",
        default=[],
    )
    args = parser.parse_args(argv)

    curves = {}
    for run in [args.run, *args.compare]:
        y_true, probabilities = load_validation_predictions(run, args.dataset)
        curves[run] = selective_accuracy(y_true, probabilities, THRESHOLDS)

    chosen = choose_threshold(curves[args.run], args.target_accuracy)
    result = {
        "run": args.run,
        "dataset": args.dataset,
        "target_accuracy": args.target_accuracy,
        "chosen": chosen,
        "recommended_env": {"CONFIDENCE_THRESHOLD": chosen["threshold"]} if chosen else None,
        "curve": curves[args.run],
        "metadata": run_metadata(),
    }
    write_json(OUT_DIR / f"{args.run}.json", result)
    plot_selective_curve(curves, chosen["threshold"] if chosen else None, OUT_DIR / f"{args.run}.png")

    if chosen is not None:
        for model_dir in args.write_to_model:
            write_threshold_to_model(model_dir, chosen["threshold"])

    if chosen is None:
        print(f"Aucun seuil n'atteint {args.target_accuracy:.0%} d'exactitude sur {args.dataset}.")
    else:
        print(
            f"Seuil retenu : {chosen['threshold']:.2f} → {chosen['coverage']:.1%} des prédictions "
            f"jugées fiables, exactitude {chosen['accuracy']:.1%} sur {args.dataset}"
        )


if __name__ == "__main__":
    main()
