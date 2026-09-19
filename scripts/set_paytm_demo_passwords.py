"""Give the Paytm demo merchants the shared demo operator password so the sign-in
buttons can log them in with one click."""

import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

EMAILS = {"pos.merchant@example.com", "qr.merchant@example.com"}
PASSWORD = "cartisan-demo-operator"

url = os.getenv("SUPABASE_URL", "").rstrip("/")
key = os.getenv("SUPABASE_SERVICE_KEY", "")
if not url or not key:
    sys.exit("Set SUPABASE_URL and SUPABASE_SERVICE_KEY in backend/.env first.")
headers = {"apikey": key, "Authorization": f"Bearer {key}"}
users = httpx.get(f"{url}/auth/v1/admin/users?per_page=200", headers=headers, timeout=20).json()["users"]
for user in users:
    if user["email"] in EMAILS:
        r = httpx.put(f"{url}/auth/v1/admin/users/{user['id']}", headers=headers,
                      json={"password": PASSWORD, "email_confirm": True}, timeout=20)
        print(user["email"], r.status_code)
