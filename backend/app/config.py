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
    # Browser origins allowed to call the API (the dashboard), comma-separated
    cors_origins: str = "http://localhost:3000"
    # Report drafts: "template" only, or "llm" to also reword them with an OpenAI-compatible model
    report_provider: str = "template"
    llm_base_url: str = ""  # e.g. https://ollama.com/v1 or http://localhost:11434/v1
    llm_api_key: str = ""
    llm_model: str = "gpt-oss:120b"
    llm_timeout_s: float = 60


@lru_cache
def get_settings() -> Settings:
    return Settings()
