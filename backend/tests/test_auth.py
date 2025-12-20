import base64
import hashlib
import hmac
import json
import time

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from app.auth import TokenVerifier
from app.config import Settings

from .conftest import ADMIN, DOCTOR_A, ISSUER_URL, PRIVATE_KEY, auth, make_token


def test_health_needs_no_login(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_me_shows_the_role(client):
    assert client.get("/me", headers=DOCTOR_A).json() == {"id": "doctor-a", "email": "a@clinic.test", "role": "doctor"}
    assert client.get("/me", headers=ADMIN).json()["role"] == "admin"


@pytest.mark.parametrize("headers", [
    {},  # no token
    {"Authorization": "Bearer not-a-token"},
    auth(expires_in=-10),  # expired
    auth(aud="anon"),  # not a logged-in user's token
    auth(iss="https://other.supabase.co/auth/v1"),  # another Supabase project
    auth(key=ec.generate_private_key(ec.SECP256R1())),  # signed by someone else
], ids=["missing", "garbage", "expired", "wrong-audience", "wrong-project", "wrong-key"])
def test_bad_tokens_are_rejected(client, headers):
    assert client.get("/me", headers=headers).status_code == 401


def forged(alg: str, secret: bytes = b"") -> str:
    """A hand-made token claiming admin, signed with HMAC (or not at all)."""
    b64 = lambda b: base64.urlsafe_b64encode(b).rstrip(b"=")
    header = b64(json.dumps({"alg": alg, "typ": "JWT"}).encode())
    payload = b64(json.dumps({"sub": "x", "aud": "authenticated", "iss": f"{ISSUER_URL}/auth/v1",
                              "exp": int(time.time()) + 600, "app_metadata": {"role": "admin"}}).encode())
    sig = b64(hmac.new(secret, header + b"." + payload, hashlib.sha256).digest()) if alg == "HS256" else b""
    return b".".join([header, payload, sig]).decode()


def test_forged_tokens_are_rejected(client):
    """Algorithm confusion (HS256 signed with the public key text) and unsigned tokens must not pass."""
    public_pem = PRIVATE_KEY.public_key().public_bytes(serialization.Encoding.PEM,
                                                       serialization.PublicFormat.SubjectPublicKeyInfo)
    for token in [forged("HS256", public_pem), forged("none")]:
        assert client.get("/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401


@pytest.mark.parametrize("role", [None, "patient", "superuser"])
def test_accounts_without_a_neurolens_role_are_refused(client, role):
    assert client.get("/me", headers=auth(role=role)).status_code == 403


def test_legacy_shared_secret_projects():
    secret = "legacy-project-secret-at-least-32-bytes"
    verifier = TokenVerifier(Settings(supabase_url="https://test.supabase.co", supabase_jwt_secret=secret))
    claims = verifier.verify(make_token(key=secret, alg="HS256"))
    assert claims["app_metadata"]["role"] == "doctor"
    with pytest.raises(jwt.InvalidSignatureError):
        verifier.verify(make_token(key="another-secret-that-is-32-bytes-long", alg="HS256"))
