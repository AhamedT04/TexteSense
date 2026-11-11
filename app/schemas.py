from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Sentiment = Literal["negative", "neutral", "positive"]


class TextInput(BaseModel):
    text: str = Field(..., min_length=5, max_length=1000)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) < 5:
            raise ValueError("Le texte doit contenir au moins 5 caractères utiles.")
        return cleaned


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    text: str
    sentiment: Sentiment
    confidence: float
    low_confidence: bool
    keywords: list[str]
    created_at: datetime
    model_name: str | None = None
    user_label: Sentiment | None = None
    user_label_at: datetime | None = None


class FeedbackInput(BaseModel):
    label: Sentiment = Field(..., description="Sentiment jugé correct par l'utilisateur.")


class StatsResponse(BaseModel):
    total: int
    positive: int
    negative: int
    neutral: int
    with_feedback: int = Field(0, description="Analyses ayant reçu un retour utilisateur.")
    corrected: int = Field(0, description="Retours qui contredisent la prédiction.")


class DeleteResponse(BaseModel):
    message: str
