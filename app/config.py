from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
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

    # Modèle Hugging Face
    hf_model_id: str = "cmarkea/distilcamembert-base-sentiment"
    hf_confidence_threshold: float = 0.55
    hf_max_tokens: int = 512

    # Modèle fallback scikit-learn
    model_dir: Path = BASE_DIR / "saved_models"
    dataset_path: Path = BASE_DIR / "data" / "sentiment_dataset.csv"

    # API
    history_default_limit: int = 50
    history_max_limit: int = 200

    @field_validator("model_dir", "dataset_path")
    @classmethod
    def _resolve_relative_to_project(cls, value: Path) -> Path:
        """Un chemin relatif dans .env est interprété depuis la racine du projet."""
        return value if value.is_absolute() else BASE_DIR / value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
