"""Settings from environment variables / .env (see .env.example)."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Supabase project URL, e.g. https://abcd.supabase.co (tokens are checked against its public keys)
    supabase_url: str = ""
    # Only for older Supabase projects that still sign tokens with a shared secret (HS256)
    supabase_jwt_secret: str = ""
    # Supabase Postgres in production; a local SQLite file for development and tests
    database_url: str = "sqlite:///./neurolens.db"
    # Where de-identified scans and results are written (a mounted volume in Docker)
    storage_dir: str = "./storage"
    max_upload_mb: int = 500
    # Folder with classifier.onnx and segmenter3d.onnx (not in git; exported by ml/ export commands)
    model_dir: str = "./models"


@lru_cache
def get_settings() -> Settings:
    return Settings()
