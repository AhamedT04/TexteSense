import logging
from collections.abc import Generator

from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.config import get_settings

logger = logging.getLogger(__name__)

Base = declarative_base()


def _build_engine():
    settings = get_settings()
    connect_args = {}
    if settings.database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(settings.database_url, connect_args=connect_args)


engine = _build_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def run_migrations(bind: Engine | None = None) -> None:
    """Migration légère : ajoute les colonnes manquantes sans Alembic.

    Utilise l'inspecteur SQLAlchemy (et non `PRAGMA`, propre à SQLite)
    pour fonctionner aussi avec PostgreSQL.
    """
    bind = bind or engine
    existing = {column["name"] for column in inspect(bind).get_columns("analyses")}

    pending: list[str] = []
    if "low_confidence" not in existing:
        # FALSE (et non 0) : PostgreSQL refuse un entier comme défaut d'un BOOLEAN.
        pending.append(
            "ALTER TABLE analyses ADD COLUMN low_confidence BOOLEAN NOT NULL DEFAULT FALSE"
        )

    if not pending:
        return

    with bind.begin() as conn:
        for stmt in pending:
            logger.info("Migration : %s", stmt)
            conn.execute(text(stmt))
    logger.info("Migration(s) appliquée(s) avec succès.")


def init_db() -> None:
    """Crée les tables manquantes puis applique les migrations."""
    # Import local : enregistre le modèle `Analysis` dans Base.metadata.
    from app import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    run_migrations()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
