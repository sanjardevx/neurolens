"""Test app on an in-memory database, with login tokens signed by a local key instead of Supabase's."""
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient

from app.auth import TokenVerifier, get_verifier
from app.config import Settings
from app.main import create_app

ISSUER_URL = "https://test.supabase.co"
PRIVATE_KEY = ec.generate_private_key(ec.SECP256R1())  # Supabase's default: ES256 on P-256


class LocalKeyVerifier(TokenVerifier):
    """Same checks as production, but the public key comes from the test key, not from Supabase's JWKS."""

    def key_for(self, token: str):
        return PRIVATE_KEY.public_key()


def make_token(sub: str = "doctor-a", role: str | None = "doctor", email: str = "a@clinic.test",
               expires_in: int = 3600, aud: str = "authenticated", iss: str = f"{ISSUER_URL}/auth/v1",
               key=PRIVATE_KEY, alg: str = "ES256") -> str:
    claims = {"sub": sub, "email": email, "aud": aud, "iss": iss, "role": "authenticated",
              "exp": int(time.time()) + expires_in, "app_metadata": {"provider": "email"}}
    if role is not None:
        claims["app_metadata"]["role"] = role
    return jwt.encode(claims, key, algorithm=alg)


def auth(**kw) -> dict:
    return {"Authorization": f"Bearer {make_token(**kw)}"}


DOCTOR_A = auth(sub="doctor-a", role="doctor")
DOCTOR_B = auth(sub="doctor-b", role="doctor", email="b@clinic.test")
ADMIN = auth(sub="admin-1", role="admin", email="admin@clinic.test")


@pytest.fixture
def client(tmp_path):
    app = create_app("sqlite://", storage_dir=str(tmp_path / "storage"))
    verifier = LocalKeyVerifier(Settings(supabase_url=ISSUER_URL))
    app.dependency_overrides[get_verifier] = lambda: verifier
    with TestClient(app) as c:  # runs startup: creates the tables
        yield c
