"""Déploie la démo publique de TextSense sur Hugging Face Spaces.

Le Space reçoit uniquement ce qui est nécessaire pour faire tourner l'application :
le dossier ``app/``, le petit jeu de données du modèle fallback, les dépendances,
la licence, et deux fichiers propres au Space (``deploy/space/Dockerfile`` et
``deploy/space/README.md``, copiés à la racine). Hugging Face construit ensuite
l'image Docker et lance l'application.

Le script crée aussi, si besoin, un dataset **privé** qui recevra les retours
utilisateurs (voir ``app/feedback_backup.py``) et renseigne son nom dans la
variable ``FEEDBACK_REPO_ID`` du Space.

Prérequis :

1. ``hf auth login`` (jeton en écriture, saisi par vous) ;
2. le modèle ONNX publié dans le dépôt du modèle (dossier ``onnx/``) ::

       python -m textsense_ml.publish_model --model-dir models/distilcamembert-3c-starinit-s42 \\
           --repo-id AhamedT04/textsense-distilcamembert-sentiment-3c \\
           --onnx models/distilcamembert-3c-starinit-s42-onnx/model.int8.onnx

Usage ::

    python deploy/deploy_space.py --dry-run   # liste les fichiers, n'envoie rien
    python deploy/deploy_space.py             # crée ou met à jour le Space

Après le premier déploiement, ajoutez dans *Settings → Variables and secrets*
du Space un secret ``HF_TOKEN`` : un jeton *fine-grained* limité à l'écriture
sur le dataset des retours. Sans lui, la démo fonctionne mais les retours ne
sont pas sauvegardés.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE_DIR = Path(__file__).resolve().parent / "space"

# (source relative à la racine du projet, destination dans le Space)
FILES = (
    ("requirements.txt", "requirements.txt"),
    ("LICENSE", "LICENSE"),
    ("data/sentiment_dataset.csv", "data/sentiment_dataset.csv"),
    ("deploy/space/Dockerfile", "Dockerfile"),
    ("deploy/space/README.md", "README.md"),
)
IGNORED = shutil.ignore_patterns("__pycache__", "*.py[cod]")


def stage(staging: Path) -> list[Path]:
    """Copie dans `staging` l'arborescence exacte du Space."""
    shutil.copytree(ROOT / "app", staging / "app", ignore=IGNORED)
    for source, destination in FILES:
        target = staging / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / source, target)
    return sorted(p for p in staging.rglob("*") if p.is_file())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space-id", default="AhamedT04/textsense")
    parser.add_argument("--feedback-repo", default="AhamedT04/textsense-feedback")
    parser.add_argument("--dry-run", action="store_true", help="liste les fichiers sans rien envoyer")
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        files = stage(staging)
        total = sum(p.stat().st_size for p in files)
        print(f"{len(files)} fichiers ({total / 1e6:.1f} Mo) à envoyer vers le Space {args.space_id} :")
        for path in files:
            print(f"  {path.relative_to(staging)}")
        if args.dry_run:
            print("--dry-run : rien n'a été envoyé.")
            return

        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.feedback_repo, repo_type="dataset", private=True, exist_ok=True)
        api.create_repo(args.space_id, repo_type="space", space_sdk="docker", exist_ok=True)
        api.add_space_variable(
            args.space_id,
            "FEEDBACK_REPO_ID",
            args.feedback_repo,
            description="Dataset privé qui reçoit les retours utilisateurs",
        )
        commit = api.upload_folder(
            repo_id=args.space_id,
            repo_type="space",
            folder_path=str(staging),
            # Supprime du Space les fichiers de code qui n'existent plus localement.
            delete_patterns=["app/**"],
            commit_message="Déploiement de la démo",
        )
        print(f"Déployé ({commit.oid[:10]}) : https://huggingface.co/spaces/{args.space_id}")
        print(
            "Rappel : ajoutez le secret HF_TOKEN dans les paramètres du Space "
            "pour activer la sauvegarde des retours."
        )


if __name__ == "__main__":
    main()
