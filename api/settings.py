"""Configuración de la API (variables de entorno o fichero .env)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Variables soportadas: ARTIFACTS_DIR, DATA_PATH, CORS_ORIGINS (separadas por comas)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    artifacts_dir: Path = REPO_ROOT / "artifacts"
    data_path: Path = REPO_ROOT / "data" / "heart.csv"
    cors_origins: str = "http://localhost:4200"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
