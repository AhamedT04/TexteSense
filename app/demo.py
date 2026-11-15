"""Protections du mode démo public (``DEMO_MODE=true``).

Une démo en ligne n'a pas les mêmes contraintes qu'une application locale :

* **cloisonnement** : chaque visiteur ne voit, ne corrige et ne supprime que
  ses propres analyses. Le navigateur génère un identifiant aléatoire (UUID v4)
  et l'envoie dans l'en-tête :data:`SESSION_HEADER`. On n'utilise pas de cookie :
  la démo est affichée dans une iframe sur huggingface.co, où les cookies tiers
  sont souvent bloqués ;
* **limite de débit** : les requêtes d'écriture sont limitées par adresse IP
  (fenêtre glissante d'une minute), pour protéger un serveur gratuit et modeste.
  L'adresse IP vient de l'en-tête ``X-Forwarded-For``, qu'un client peut
  falsifier : une seconde limite, **globale** (tous visiteurs confondus),
  plafonne donc la charge quoi qu'il arrive.

Hors mode démo, ces dépendances ne font rien : l'application locale garde son
comportement (historique commun, pas de limite).
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque

from fastapi import HTTPException, Request, status

from app.config import get_settings

SESSION_HEADER = "X-TextSense-Session"


def get_session_id(request: Request) -> str | None:
    """Identifiant du visiteur en mode démo, ``None`` sinon.

    Un en-tête absent ou invalide donne une session neuve et vide : le visiteur
    peut analyser un texte mais ne retrouve aucune analyse existante.
    """
    if not get_settings().demo_mode:
        return None
    raw = request.headers.get(SESSION_HEADER, "")
    try:
        session = uuid.UUID(raw)
    except ValueError:
        return str(uuid.uuid4())
    # Format attendu : UUID v4, tiré au hasard par le navigateur (122 bits
    # aléatoires : impossible de deviner celui d'un autre visiteur).
    return str(session) if session.version == 4 else str(uuid.uuid4())


class RateLimiter:
    """Fenêtre glissante en mémoire : au plus ``limit`` requêtes par ``window`` secondes.

    Suffisant pour un seul processus (cas d'un Space). Derrière plusieurs
    processus ou serveurs, il faudrait un stockage partagé (Redis).
    """

    def __init__(self, limit: int, window: float = 60.0, clock=time.monotonic) -> None:
        self.limit = limit
        self.window = window
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def retry_after(self, key: str) -> float:
        """Enregistre une requête ; renvoie 0 si elle est acceptée, sinon le délai d'attente (s)."""
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return self.window - (now - hits[0])
            hits.append(now)
            # Évite que le dictionnaire grossisse indéfiniment avec des IP inactives.
            if len(self._hits) > 10_000:
                self._hits = {k: v for k, v in self._hits.items() if v and now - v[-1] < self.window}
            return 0.0


_limiters: dict[str, RateLimiter] = {}


def _get_limiter(name: str, limit: int) -> RateLimiter:
    limiter = _limiters.get(name)
    if limiter is None or limiter.limit != limit:
        limiter = _limiters[name] = RateLimiter(limit)
    return limiter


def reset_rate_limiter() -> None:
    _limiters.clear()


def _too_many(wait: float, detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=detail,
        headers={"Retry-After": str(max(1, round(wait)))},
    )


def rate_limit(request: Request) -> None:
    """Dépendance FastAPI : renvoie 429 au-delà de la limite (mode démo seulement)."""
    settings = get_settings()
    if not settings.demo_mode:
        return
    # Derrière le proxy de Hugging Face, uvicorn est lancé avec --proxy-headers :
    # request.client.host est alors l'adresse transmise par le proxy.
    client = request.client.host if request.client else "inconnu"
    wait = _get_limiter("ip", settings.rate_limit_per_minute).retry_after(client)
    if wait > 0:
        raise _too_many(wait, "Trop de requêtes : réessayez dans une minute.")
    wait = _get_limiter("global", settings.rate_limit_global_per_minute).retry_after("*")
    if wait > 0:
        raise _too_many(wait, "La démo est très sollicitée : réessayez dans une minute.")
