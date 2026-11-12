"""Évalue un modèle sur un ou plusieurs jeux de données.

Exemples (depuis la racine du projet) ::

    python -m textsense_ml.evaluate --name baseline-stars \\
        --model hf:cmarkea/distilcamembert-base-sentiment \\
        --datasets textsense_eval_v1 tweets_fr_test

    python -m textsense_ml.evaluate --name tfidf-fallback --model sklearn:saved_models

Pour chaque jeu, écrit dans ``reports/eval/<name>/<dataset>/`` :

* ``metrics.json``   — métriques + métadonnées de reproductibilité ;
* ``predictions.csv`` — une ligne par exemple (le texte n'est inclus que pour
  les jeux redistribuables) ;
* ``confusion_matrix.png``.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from app.config import BASE_DIR
from app.ml.classifiers import load_classifier
from app.ml.labels import SENTIMENTS
from textsense_ml.data import REDISTRIBUTABLE, load_dataset
from textsense_ml.metrics import accuracy_by_group, classification_metrics, predictions_from_proba
from textsense_ml.plots import plot_confusion_matrix
from textsense_ml.tracking import run_metadata, write_json

logger = logging.getLogger(__name__)
REPORTS_DIR = BASE_DIR / "reports" / "eval"
GROUP_COLUMNS = ("phenomenon", "domain")


def evaluate_dataset(classifier, dataset_name: str, run_name: str, model_spec: str, out_root: Path) -> dict:
    df = load_dataset(dataset_name)
    texts = df["text"].tolist()

    start = time.perf_counter()
    probabilities = classifier.predict_proba(texts)
    elapsed = time.perf_counter() - start

    predictions = df.copy()
    predictions["pred"] = predictions_from_proba(probabilities)
    predictions["confidence"] = probabilities.max(axis=1).round(4)
    for i, sentiment in enumerate(SENTIMENTS):
        predictions[f"p_{sentiment}"] = probabilities[:, i].round(4)

    metrics = classification_metrics(df["label"].tolist(), probabilities)
    metrics["by_group"] = {
        column: accuracy_by_group(predictions, column) for column in GROUP_COLUMNS if column in predictions
    }
    metrics["throughput"] = {
        "seconds_total": round(elapsed, 3),
        "ms_per_text": round(1000 * elapsed / max(len(texts), 1), 2),
        "note": "inférence par lots, CPU ; indicatif, dépend de la machine",
    }

    out_dir = out_root / run_name / dataset_name
    write_json(
        out_dir / "metrics.json",
        {
            "run": run_name,
            "dataset": dataset_name,
            "model": model_spec,
            "metrics": metrics,
            "metadata": run_metadata(),
        },
    )
    if dataset_name not in REDISTRIBUTABLE:
        predictions = predictions.drop(columns=["text"])
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    plot_confusion_matrix(
        metrics["confusion_matrix"]["rows_true_cols_pred"],
        list(SENTIMENTS),
        f"{run_name} — {dataset_name}\nmacro-F1 = {metrics['macro_f1']:.3f}",
        out_dir / "confusion_matrix.png",
    )
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="nom de l'exécution (dossier du rapport)")
    parser.add_argument("--model", required=True, help="hf:<id|dossier>[@rev], onnx:<dossier> ou sklearn:<dossier>")
    parser.add_argument("--datasets", nargs="+", default=["textsense_eval_v1", "tweets_fr_test"])
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--out", type=Path, default=REPORTS_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    kwargs = {} if args.model.startswith("sklearn:") else {"max_length": args.max_length}
    classifier = load_classifier(args.model, **kwargs)

    rows = []
    for dataset_name in args.datasets:
        metrics = evaluate_dataset(classifier, dataset_name, args.name, args.model, args.out)
        low, high = metrics["ci95"]["macro_f1"]
        rows.append(
            {
                "dataset": dataset_name,
                "n": metrics["n"],
                "macro_f1": f"{metrics['macro_f1']:.3f} [{low:.3f}–{high:.3f}]",
                "accuracy": f"{metrics['accuracy']:.3f}",
                "f1_neutral": f"{metrics['per_class']['neutral']['f1']:.3f}",
                "ece": f"{metrics['ece']:.3f}",
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
