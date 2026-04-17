import logging
from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

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


def run_migrations() -> None:
    """Migration légère : ajoute les colonnes manquantes sans Alembic."""
    with engine.connect() as conn:
        result = conn.execute(text("PRAGMA table_info(analyses)"))
        existing = {row[1] for row in result}

        pending: list[str] = []
        if "low_confidence" not in existing:
            pending.append(
                "ALTER TABLE analyses ADD COLUMN low_confidence BOOLEAN NOT NULL DEFAULT 0"
            )

        for stmt in pending:
            logger.info("Migration : %s", stmt)
            conn.execute(text(stmt))

        if pending:
            conn.commit()
            logger.info("Migration(s) appliquée(s) avec succès.")


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
