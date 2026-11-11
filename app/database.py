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


# Valeur par défaut SQL des colonnes NOT NULL ajoutées après coup.
# FALSE (et non 0) : PostgreSQL refuse un entier comme défaut d'un BOOLEAN.
_ADDED_COLUMN_DEFAULTS = {"low_confidence": "FALSE"}


def run_migrations(bind: Engine | None = None) -> None:
    """Migration légère : ajoute les colonnes du modèle absentes de la table.

    * utilise l'inspecteur SQLAlchemy (et non `PRAGMA`, propre à SQLite) pour
      fonctionner aussi avec PostgreSQL ;
    * le type SQL de chaque colonne est généré pour le dialecte de la base.

    Limite assumée : ne gère que l'*ajout* de colonnes. Renommer, supprimer ou
    changer un type nécessiterait un vrai outil de migration (Alembic).
    """
    from app.models import Analysis

    bind = bind or engine
    table = Analysis.__table__
    existing = {column["name"] for column in inspect(bind).get_columns(table.name)}

    pending: list[str] = []
    for column in table.columns:
        if column.name in existing:
            continue
        column_type = column.type.compile(dialect=bind.dialect)
        if column.nullable:
            pending.append(f"ALTER TABLE {table.name} ADD COLUMN {column.name} {column_type}")
        elif column.name in _ADDED_COLUMN_DEFAULTS:
            default = _ADDED_COLUMN_DEFAULTS[column.name]
            pending.append(
                f"ALTER TABLE {table.name} ADD COLUMN {column.name} {column_type} NOT NULL DEFAULT {default}"
            )
        else:
            raise RuntimeError(
                f"Colonne obligatoire « {column.name} » absente sans valeur par défaut : "
                "migration manuelle nécessaire."
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
