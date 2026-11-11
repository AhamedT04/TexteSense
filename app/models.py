from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    sentiment: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    low_confidence: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    keywords: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    # Modèle ayant produit la prédiction : indispensable pour comparer les
    # versions de modèles et exploiter les retours utilisateurs.
    model_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # Retour utilisateur : sentiment jugé correct (identique à la prédiction
    # si l'utilisateur la confirme, différent s'il la corrige).
    user_label: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    user_label_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
