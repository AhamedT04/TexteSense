"""Retours utilisateurs : extraction et sauvegarde vers un dataset du Hub.

Sur un Space Hugging Face gratuit, le disque est éphémère : la base SQLite est
effacée à chaque redémarrage. Pour ne pas perdre les retours (« Oui » /
« Sinon, c'est plutôt… »), :class:`FeedbackBackup` envoie régulièrement un
instantané des analyses annotées vers un dataset **privé** du Hub.

Choix de conception :

* **minimisation des données** : seules les analyses ayant reçu un avis sont
  sauvegardées, sans identifiant de session ni adresse IP ;
* **un fichier par démarrage** (``feedback/<horodatage>-<id>.jsonl``) réécrit
  à chaque envoi : une analyse supprimée par le visiteur disparaît du fichier
  au prochain envoi, et les fichiers des démarrages précédents sont conservés ;
* **pas d'envoi inutile** : rien n'est envoyé si l'instantané n'a pas changé ;
* une erreur réseau est journalisée sans jamais interrompre l'application.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from app.models import Analysis

logger = logging.getLogger(__name__)

FEEDBACK_FIELDS = (
    "id",
    "text",
    "label",
    "model_label",
    "confidence",
    "model_name",
    "created_at",
    "user_label_at",
)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:  # SQLite relit les dates sans fuseau : elles sont en UTC.
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def feedback_records(db: Session, only_corrections: bool = False) -> list[dict[str, Any]]:
    """Analyses ayant reçu un avis, au format d'export (``label`` = avis de l'utilisateur)."""
    query = db.query(Analysis).filter(Analysis.user_label.is_not(None))
    if only_corrections:
        query = query.filter(Analysis.user_label != Analysis.sentiment)
    return [
        {
            "id": f"fb-{record.id}",
            "text": record.text,
            "label": record.user_label,
            "model_label": record.sentiment,
            "confidence": record.confidence,
            "model_name": record.model_name,
            "created_at": _iso(record.created_at),
            "user_label_at": _iso(record.user_label_at),
        }
        for record in query.order_by(Analysis.id).all()
    ]


class FeedbackBackup:
    """Envoie périodiquement les retours utilisateurs vers un dataset du Hub."""

    def __init__(
        self,
        repo_id: str,
        session_factory: sessionmaker,
        interval_seconds: float = 600,
        api: Any | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.interval_seconds = interval_seconds
        self._session_factory = session_factory
        self._api = api
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path_in_repo = f"feedback/{stamp}-{uuid.uuid4().hex[:6]}.jsonl"
        self._last_digest: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._push_lock = threading.Lock()

    @property
    def api(self):
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi()  # jeton lu dans la variable d'environnement HF_TOKEN
        return self._api

    def snapshot(self) -> bytes:
        with self._session_factory() as db:
            records = feedback_records(db)
        lines = (json.dumps(record, ensure_ascii=False) for record in records)
        return "".join(f"{line}\n" for line in lines).encode("utf-8")

    def push(self) -> bool:
        """Envoie l'instantané s'il a changé. Renvoie True si un envoi a eu lieu."""
        with self._push_lock:
            try:
                data = self.snapshot()
                digest = hashlib.sha256(data).hexdigest()
                if digest == self._last_digest or (not data and self._last_digest is None):
                    return False
                count = data.count(b"\n")
                self.api.upload_file(
                    path_or_fileobj=data,
                    path_in_repo=self.path_in_repo,
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    commit_message=f"Sauvegarde des retours utilisateurs ({count})",
                )
            except Exception as exc:  # réseau, jeton, quota… : on réessaiera au prochain cycle
                logger.warning("Sauvegarde des retours vers %s impossible : %s", self.repo_id, exc)
                return False
            self._last_digest = digest
            logger.info("%d retour(s) sauvegardé(s) vers %s/%s", count, self.repo_id, self.path_in_repo)
            return True

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.push()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="feedback-backup", daemon=True)
        self._thread.start()
        logger.info(
            "Sauvegarde des retours activée : %s toutes les %.0f s", self.repo_id, self.interval_seconds
        )

    def stop(self) -> None:
        """Arrête la boucle et tente un dernier envoi (arrêt propre du Space)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.push()
