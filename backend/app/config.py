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


@lru_cache
def get_settings() -> Settings:
    return Settings()
