"""Publie un modèle entraîné sur le Hub Hugging Face, avec sa fiche (model card).

Les poids (~270 Mo) sont trop lourds pour git : on les publie sur le Hub, et
l'application les télécharge via ``HF_MODEL_ID=<compte>/<nom>``.

Prérequis (une seule fois) : un compte Hugging Face et un jeton d'accès en
écriture, saisi par **vous** dans le terminal :

    hf auth login

Usage ::

    # Vérifier ce qui serait envoyé, sans rien publier :
    python -m textsense_ml.publish_model --model-dir models/distilcamembert-3c-starinit-s42 \\
        --repo-id <compte>/textsense-distilcamembert-sentiment-3c --dry-run

    # Publier :
    python -m textsense_ml.publish_model --model-dir models/distilcamembert-3c-starinit-s42 \\
        --repo-id <compte>/textsense-distilcamembert-sentiment-3c
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from app.config import BASE_DIR

MODEL_CARD = BASE_DIR / "docs" / "MODEL_CARD.md"
# Fichiers publiés ; tout le reste (checkpoints, métadonnées locales) est exclu.
PUBLISHED_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "training_info.json",
)


def stage(model_dir: Path, staging: Path) -> list[Path]:
    files = []
    for name in PUBLISHED_FILES:
        source = model_dir / name
        if source.exists():
            shutil.copy(source, staging / name)
            files.append(staging / name)
    shutil.copy(MODEL_CARD, staging / "README.md")
    files.append(staging / "README.md")
    return files


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True, help="<compte>/<nom-du-modèle>")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="liste les fichiers sans publier")
    args = parser.parse_args(argv)

    model_dir = args.model_dir if args.model_dir.is_absolute() else BASE_DIR / args.model_dir
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        files = stage(model_dir, staging)
        print(f"Fichiers à publier vers {args.repo_id} :")
        for path in files:
            print(f"  {path.name:<28} {path.stat().st_size / 1e6:8.1f} Mo")
        if args.dry_run:
            print("--dry-run : rien n'a été publié.")
            return

        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
        commit = api.upload_folder(
            repo_id=args.repo_id,
            folder_path=str(staging),
            commit_message=f"Publication de {model_dir.name}",
        )
        print(f"Publié : https://huggingface.co/{args.repo_id} ({commit.oid[:10]})")


if __name__ == "__main__":
    main()
