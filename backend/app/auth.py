"""Who is calling: check the Supabase login token and read the user's role.

The dashboard logs in with Supabase Auth and sends the access token as `Authorization: Bearer <token>`.
The role lives in the token's `app_metadata.role`, which only an admin (service key / SQL) can set,
so a user can't make themselves an admin. Only two roles exist: doctor and admin. Patients have no accounts.
"""
from dataclasses import dataclass
from typing import Literal

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import Settings, get_settings

Role = Literal["doctor", "admin"]
ROLES = ("doctor", "admin")
AUDIENCE = "authenticated"  # Supabase sets this on tokens of logged-in users


@dataclass(frozen=True)
class User:
    id: str
    email: str
    role: Role


class TokenVerifier:
    """New Supabase projects sign tokens with a key pair (public keys at /auth/v1/.well-known/jwks.json);
    older ones with a shared secret (HS256). Both are supported."""

    def __init__(self, settings: Settings):
        base = settings.supabase_url.rstrip("/")
        self.issuer = f"{base}/auth/v1" if base else None
        self.secret = settings.supabase_jwt_secret
        self.jwks = jwt.PyJWKClient(f"{base}/auth/v1/.well-known/jwks.json", cache_keys=True) if base else None

    def key_for(self, token: str):
        if self.secret:
            return self.secret
        if self.jwks is None:
            raise jwt.InvalidTokenError("SUPABASE_URL is not set")
        return self.jwks.get_signing_key_from_jwt(token).key

    def verify(self, token: str) -> dict:
        algorithms = ["HS256"] if self.secret else ["ES256", "RS256"]
        return jwt.decode(token, self.key_for(token), algorithms=algorithms, audience=AUDIENCE, issuer=self.issuer,
                          options={"require": ["exp", "sub"], "verify_iss": self.issuer is not None})


_verifier: TokenVerifier | None = None


def get_verifier() -> TokenVerifier:
    global _verifier  # one verifier per process, so the public keys are fetched once
    if _verifier is None:
        _verifier = TokenVerifier(get_settings())
    return _verifier


bearer = HTTPBearer(auto_error=False)


def current_user(creds: HTTPAuthorizationCredentials | None = Depends(bearer),
                 verifier: TokenVerifier = Depends(get_verifier)) -> User:
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not logged in", headers={"WWW-Authenticate": "Bearer"})
    try:
        claims = verifier.verify(creds.credentials)
    except jwt.PyJWTError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Invalid login token: {e}",
                            headers={"WWW-Authenticate": "Bearer"}) from e
    role = (claims.get("app_metadata") or {}).get("role")
    if role not in ROLES:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "No NeuroLens role on this account. Ask an admin.")
    return User(id=claims["sub"], email=claims.get("email", ""), role=role)


def require(*roles: Role):
    """Dependency: the caller must have one of these roles."""
    def check(user: User = Depends(current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Only {' or '.join(roles)} can do this")
        return user
    return check
