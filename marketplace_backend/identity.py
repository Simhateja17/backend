from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx

from .store import Store

# A verified token is re-checked against Supabase at most this often. Short enough
# that a revoked session stops working quickly, long enough that a burst of cart
# reads doesn't become a burst of auth round-trips.
_TOKEN_CACHE_SECONDS = 60


class AuthenticationError(Exception):
    """The caller did not present a token Supabase Auth recognises."""


@dataclass(frozen=True)
class Principal:
    """The authenticated actor for one request. Never built from request body fields."""

    id: str
    email: str
    role: str  # "customer" | "merchant_operator"
    display_name: str | None = None
    # Which Paytm products the merchant runs: "pos" has catalogue and stock, "payments"
    # (QR / Soundbox only) has payment data and nothing about products.
    paytm_plan: str = "pos"


class IdentityService:
    """Derives the principal from a Supabase access token.

    Verification goes to Supabase's own `/auth/v1/user` endpoint rather than a
    locally held signing secret, so key rotation and session revocation take
    effect without redeploying the backend.
    """

    def __init__(self, store: Store, supabase_url: str | None = None, anon_key: str | None = None) -> None:
        self.store = store
        # An explicitly empty value means "unconfigured" and must not fall back to
        # the environment, so a test can assert the unconfigured refusal path.
        self.supabase_url = (os.getenv("SUPABASE_URL", "") if supabase_url is None else supabase_url).rstrip("/")
        self.anon_key = os.getenv("SUPABASE_ANON_KEY", "") if anon_key is None else anon_key
        self._cache: dict[str, tuple[float, Principal]] = {}

    @property
    def configured(self) -> bool:
        return bool(self.supabase_url and self.anon_key)

    @staticmethod
    def bearer(authorization: str | None) -> str:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise AuthenticationError("Sign in to continue")
        token = authorization.split(" ", 1)[1].strip()
        if not token:
            raise AuthenticationError("Sign in to continue")
        return token

    def principal(self, authorization: str | None) -> Principal:
        token = self.bearer(authorization)
        cached = self._cache.get(token)
        if cached and cached[0] > time.monotonic():
            return cached[1]
        principal = self._register(self._verify(token))
        self._cache[token] = (time.monotonic() + _TOKEN_CACHE_SECONDS, principal)
        return principal

    def _verify(self, token: str) -> dict:
        if not self.configured:
            raise AuthenticationError(
                "Supabase Auth is not configured. Set SUPABASE_URL and SUPABASE_ANON_KEY."
            )
        try:
            response = httpx.get(
                f"{self.supabase_url}/auth/v1/user",
                headers={"apikey": self.anon_key, "Authorization": f"Bearer {token}"},
                timeout=10,
            )
        except httpx.HTTPError as exc:
            raise AuthenticationError("Could not reach Supabase Auth to verify the session") from exc
        if response.status_code != 200:
            raise AuthenticationError("Session is invalid or expired")
        user = response.json()
        if not user.get("id") or not user.get("email"):
            raise AuthenticationError("Session is invalid or expired")
        return user

    def _register(self, user: dict) -> Principal:
        """Mirror the verified auth user into the commerce principal tables.

        Role comes from the app metadata Supabase controls, never from the client.
        """
        metadata = user.get("app_metadata") or {}
        role = "merchant_operator" if metadata.get("cartisan_role") == "merchant_operator" else "customer"
        display_name = (user.get("user_metadata") or {}).get("display_name")
        if role == "merchant_operator":
            plan = "payments" if metadata.get("paytm_plan") == "payments" else "pos"
            self.store.execute(
                "INSERT INTO merchant_operators (id,email,display_name,paytm_plan) VALUES (?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET email=excluded.email, "
                "display_name=excluded.display_name, paytm_plan=excluded.paytm_plan",
                (user["id"], user["email"], display_name, plan),
            )
            return Principal(id=user["id"], email=user["email"], role=role,
                             display_name=display_name, paytm_plan=plan)
        self.store.execute(
            "INSERT INTO customers (id,email,display_name) VALUES (?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET email=excluded.email, display_name=excluded.display_name",
            (user["id"], user["email"], display_name),
        )
        return Principal(id=user["id"], email=user["email"], role=role, display_name=display_name)


def paytm_plan_for(store: Store, operator_id: str) -> str:
    """The plan recorded at the operator's last verified sign-in, for channels such as
    Telegram that know the principal id but not the token."""
    rows = store.rows("SELECT paytm_plan FROM merchant_operators WHERE id=?", (operator_id,))
    return "payments" if rows and rows[0]["paytm_plan"] == "payments" else "pos"
