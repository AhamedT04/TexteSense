from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    sentiment: str
    confidence: float
    low_confidence: bool
    keywords: list[str]
    created_at: datetime


class StatsResponse(BaseModel):
    total: int
    positive: int
    negative: int
    neutral: int


class DeleteResponse(BaseModel):
    message: str
