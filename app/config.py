from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Racine du projet : les chemins par défaut ne dépendent plus du dossier
# depuis lequel l'application est lancée.
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "TextSense"
    debug: bool = False

    # Base de données
    database_url: str = f"sqlite:///{BASE_DIR / 'textsense.db'}"

    # Modèle principal
    # - "transformers" : HF_MODEL_ID = identifiant du Hub ou dossier local ;
    # - "onnx" : ONNX_MODEL_DIR = dossier contenant le .onnx, la config et le tokenizer.
    model_backend: Literal["transformers", "onnx"] = "transformers"
    hf_model_id: str = "cmarkea/distilcamembert-base-sentiment"
    onnx_model_dir: Path | None = None
    onnx_model_file: str = "model.onnx"
    hf_max_tokens: int = 512

    # En dessous de ce seuil, la prédiction est marquée `low_confidence`.
    # Accepte aussi l'ancien nom HF_CONFIDENCE_THRESHOLD.
    confidence_threshold: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("CONFIDENCE_THRESHOLD", "HF_CONFIDENCE_THRESHOLD"),
    )

    # Modèle fallback scikit-learn
    model_dir: Path = BASE_DIR / "saved_models"
    dataset_path: Path = BASE_DIR / "data" / "sentiment_dataset.csv"

    # API
    history_default_limit: int = 50
    history_max_limit: int = 200

    @field_validator("model_dir", "dataset_path", "onnx_model_dir")
    @classmethod
    def _resolve_relative_to_project(cls, value: Path | None) -> Path | None:
        """Un chemin relatif dans .env est interprété depuis la racine du projet."""
        if value is None or value.is_absolute():
            return value
        return BASE_DIR / value

    @field_validator("hf_model_id")
    @classmethod
    def _resolve_local_model_dir(cls, value: str) -> str:
        """Un HF_MODEL_ID qui désigne un dossier du projet devient un chemin absolu."""
        candidate = BASE_DIR / value
        if not Path(value).is_absolute() and candidate.is_dir():
            return str(candidate)
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
