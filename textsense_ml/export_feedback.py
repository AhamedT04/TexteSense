"""Exporte les retours utilisateurs en CSV pour constituer un jeu de données réel.

Usage ::

    python -m textsense_ml.export_feedback                     # tous les retours
    python -m textsense_ml.export_feedback --only-corrections  # seulement les désaccords

Le fichier est écrit dans ``data/feedback/`` (non versionné : il contient des
textes saisis par des utilisateurs). Colonnes : ``id, text, label`` (étiquette
donnée par l'utilisateur) puis le contexte de la prédiction.

Avant d'utiliser ces données pour entraîner un modèle : relire les étiquettes,
retirer les doublons et toute donnée personnelle, et ne jamais mélanger ces
exemples avec ``textsense_eval_v1``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from app.config import BASE_DIR
from app.database import SessionLocal
from app.models import Analysis

FEEDBACK_DIR = BASE_DIR / "data" / "feedback"
COLUMNS = [
    "id",
    "text",
    "label",
    "model_label",
    "confidence",
    "model_name",
    "created_at",
    "user_label_at",
]


def collect_feedback(db: Session, only_corrections: bool = False) -> pd.DataFrame:
    query = db.query(Analysis).filter(Analysis.user_label.is_not(None))
    if only_corrections:
        query = query.filter(Analysis.user_label != Analysis.sentiment)
    rows = [
        {
            "id": f"fb-{record.id}",
            "text": record.text,
            "label": record.user_label,
            "model_label": record.sentiment,
            "confidence": record.confidence,
            "model_name": record.model_name,
            "created_at": record.created_at,
            "user_label_at": record.user_label_at,
        }
        for record in query.order_by(Analysis.id).all()
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only-corrections", action="store_true", help="exporter seulement les désaccords")
    parser.add_argument("--out", type=Path, help="fichier de sortie (défaut : data/feedback/feedback_<date>.csv)")
    args = parser.parse_args(argv)

    with SessionLocal() as db:
        df = collect_feedback(db, only_corrections=args.only_corrections)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = args.out or FEEDBACK_DIR / f"feedback_{stamp}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    agreement = (df["label"] == df["model_label"]).mean() if len(df) else float("nan")
    print(f"{len(df)} retours exportés vers {out}")
    if len(df):
        print(f"Accord modèle / utilisateur : {agreement:.1%}")
        print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
