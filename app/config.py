from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "TextSense"
    debug: bool = False

    # Base de données
    database_url: str = "sqlite:///./textsense.db"

    # Modèle Hugging Face
    hf_model_id: str = "cmarkea/distilcamembert-base-sentiment"
    hf_confidence_threshold: float = 0.55
    hf_max_tokens: int = 512

    # Modèle fallback scikit-learn
    model_dir: Path = Path("saved_models")

    # API
    history_default_limit: int = 50
    history_max_limit: int = 200


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
