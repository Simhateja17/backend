"""Cart recovery: the merchant's approved policy, applied by code (Q3, Q4).

Who authorises a discount is the operator, once, through the maker-checker queue
(`recovery_policy` change kind). Who decides *this* customer gets *this* offer is the
deterministic scan below: an abandoned cart, above the policy's floor, outside the
cooldown, inside the monthly budget. Memory only personalises inside that envelope —
which cart line to lead with, and whether a shopper who reliably buys without a
discount should get a plain reminder instead of money off.

Nothing here is reachable from a model. The coupon is bound to one customer, single
use and expiring; redemption happens at checkout staging, where the discount is
computed from the stored terms, never from anything a client or model sends.
"""

from __future__ import annotations

import html
import logging
import os
import secrets
from datetime import UTC, datetime, timedelta
from typing import Callable, Protocol
from uuid import uuid4

import httpx

from .evidence import Actor, Correlation, EvidenceLedger, Outbox
from .store import Store
from .timeutil import as_datetime, now

logger = logging.getLogger(__name__)

EMAIL_TOPIC = "recovery.email"
# Paid orders in the window that, with no promotion on them, mark a shopper who buys
# without being paid to — they get a reminder, which protects margin.
FULL_PRICE_BUYER_ORDERS = 2
FULL_PRICE_WINDOW_DAYS = 90


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:16]}"


def active_policy(store: Store) -> dict | None:
    rows = store.rows("SELECT * FROM recovery_policies WHERE status='active' "
                      "ORDER BY created_at DESC LIMIT 1")
    return rows[0] if rows else None


def discount_for(offer: dict, subtotal_minor: int) -> int:
    """The discount an offer grants on a subtotal: the percentage, capped."""
    if offer.get("kind") != "coupon" or not offer.get("discount_percentage"):
        return 0
    return min(subtotal_minor * int(offer["discount_percentage"]) // 100,
               int(offer["max_discount_minor"]))


class RecoveryOffers:
    """Scan, issue, look up and redeem."""

    def __init__(self, store: Store, outbox: Outbox, ledger: EvidenceLedger,
                 price_of: Callable[[str], int],
                 brief_of: Callable[[str], dict | None] | None = None) -> None:
        self.store, self.outbox, self.ledger = store, outbox, ledger
        self.price_of, self.brief_of = price_of, brief_of

    # -- the scan ----------------------------------------------------------------

    def scan(self, *, at: datetime | None = None) -> list[dict]:
        policy = active_policy(self.store)
        if policy is None:
            return []
        at = at or datetime.now(UTC)
        cutoff = (at - timedelta(minutes=int(policy["abandon_after_minutes"]))).isoformat()
        carts = self.store.rows(
            "SELECT c.id, c.customer_id, c.state_version, c.updated_at FROM customer_carts c "
            "JOIN customers u ON u.id = c.customer_id "
            "WHERE c.status='active' AND c.updated_at <= ?", (cutoff,))
        issued = []
        for cart in carts:
            offer = self._consider(policy, cart, at)
            if offer is not None:
                issued.append(offer)
        return issued

    def _consider(self, policy: dict, cart: dict, at: datetime) -> dict | None:
        customer_id = cart["customer_id"]
        if self.store.rows("SELECT 1 AS seen FROM recovery_offers WHERE cart_id=? AND cart_state_version=?",
                           (cart["id"], cart["state_version"])):
            return None
        lines = self.store.rows(
            "SELECT l.product_id AS variant_id, l.quantity FROM cart_lines l WHERE l.cart_id=?",
            (cart["id"],))
        if not lines:
            return None
        subtotal = sum(self.price_of(line["variant_id"]) * int(line["quantity"]) for line in lines)
        if subtotal < int(policy["min_cart_minor"]):
            return None
        cooldown = (at - timedelta(days=int(policy["cooldown_days"]))).isoformat()
        if self.store.rows("SELECT 1 AS recent FROM recovery_offers WHERE customer_id=? AND created_at>=?",
                           (customer_id, cooldown)):
            return None
        # A cart touched after the customer's latest order is still a live intention;
        # one the customer already paid for is not abandoned.
        paid_since = self.store.rows(
            "SELECT 1 AS paid FROM commerce_orders WHERE customer_id=? AND status='paid' "
            "AND created_at>=?", (customer_id, str(cart["updated_at"])))
        if paid_since:
            return None

        kind, why = "coupon", "coupon"
        if self._buys_at_full_price(customer_id, at):
            kind, why = "reminder", "reminder (buys without discounts)"
        elif not self._within_budget(policy, int(policy["max_discount_minor"]), at):
            kind, why = "reminder", "reminder (monthly budget reached)"
        headline = self._headline(customer_id, [line["variant_id"] for line in lines])
        expires = (at + timedelta(hours=int(policy["offer_valid_hours"]))).isoformat()
        offer_id = _id("roff")
        code = promotion_id = None
        with self.store.transaction() as tx:
            if kind == "coupon":
                code = f"BACK{secrets.token_hex(3).upper()}"
                promotion_id = f"promo_{uuid4().hex[:12]}"
                tx.execute(
                    "INSERT INTO promotions (id,code,description,discount_kind,discount_value,"
                    "min_subtotal_minor,status,starts_at,ends_at) VALUES (?,?,?,'percentage',?,?,'active',?,?)",
                    (promotion_id, code,
                     f"Cart recovery offer for one customer (policy {policy['id']})",
                     int(policy["discount_percentage"]), int(policy["min_cart_minor"]),
                     at.isoformat(), expires))
            tx.execute(
                "INSERT INTO recovery_offers (id,customer_id,cart_id,cart_state_version,policy_id,kind,"
                "promotion_id,code,discount_percentage,max_discount_minor,headline_variant_id,status,"
                "email_status,created_at,expires_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,'issued','pending',?,?)",
                (offer_id, customer_id, cart["id"], cart["state_version"], policy["id"], kind,
                 promotion_id, code,
                 int(policy["discount_percentage"]) if kind == "coupon" else None,
                 int(policy["max_discount_minor"]) if kind == "coupon" else None,
                 headline, at.isoformat(), expires))
            self.outbox.enqueue(topic=EMAIL_TOPIC, payload={"offer_id": offer_id}, tx=tx)
            self.ledger.record(
                actor=Actor(type="system", id="cart-recovery", surface="recovery"),
                action="recovery.offer_issued",
                reason=(f"Cart abandoned over {policy['abandon_after_minutes']} min; {why} "
                        f"under approved policy {policy['id']}"),
                outcome="applied", target_type="recovery_offer", target_id=offer_id,
                policy_checks={"policy_id": policy["id"], "subtotal_minor": subtotal,
                               "min_cart_minor": policy["min_cart_minor"], "kind": kind},
                correlation=Correlation(), tx=tx)
        return self.get(offer_id)

    def _buys_at_full_price(self, customer_id: str, at: datetime) -> bool:
        since = (at - timedelta(days=FULL_PRICE_WINDOW_DAYS)).isoformat()
        rows = self.store.rows(
            "SELECT COUNT(*) AS n FROM commerce_orders WHERE customer_id=? AND status='paid' "
            "AND discount_minor=0 AND created_at>=?", (customer_id, since))
        return int(rows[0]["n"]) >= FULL_PRICE_BUYER_ORDERS

    def _within_budget(self, policy: dict, max_discount: int, at: datetime) -> bool:
        month_start = at.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
        rows = self.store.rows(
            "SELECT COALESCE(SUM(max_discount_minor),0) AS committed FROM recovery_offers "
            "WHERE kind='coupon' AND created_at>=?", (month_start,))
        return int(rows[0]["committed"]) + max_discount <= int(policy["monthly_budget_minor"])

    def _headline(self, customer_id: str, variant_ids: list[str]) -> str | None:
        """Lead with the line the shopper showed most interest in; else the priciest."""
        brief = self.brief_of(customer_id) if self.brief_of else None
        explored = [item["variant_id"] for item in (brief or {}).get("explored", [])]
        for variant_id in explored:
            if variant_id in variant_ids:
                return variant_id
        return max(variant_ids, key=self.price_of) if variant_ids else None

    # -- reads -------------------------------------------------------------------

    def get(self, offer_id: str) -> dict | None:
        rows = self.store.rows("SELECT * FROM recovery_offers WHERE id=?", (offer_id,))
        return rows[0] if rows else None

    def live_offer(self, customer_id: str, *, at: datetime | None = None) -> dict | None:
        at = at or datetime.now(UTC)
        rows = self.store.rows(
            "SELECT * FROM recovery_offers WHERE customer_id=? AND status='issued' "
            "AND expires_at>? ORDER BY created_at DESC LIMIT 1", (customer_id, at.isoformat()))
        return rows[0] if rows else None

    def view(self, offer: dict | None) -> dict | None:
        """What the storefront shows. Title and live price for the headline item."""
        if offer is None:
            return None
        headline = None
        if offer.get("headline_variant_id"):
            rows = self.store.rows(
                "SELECT v.id, COALESCE(v.title, p.title) AS title FROM catalog_variants v "
                "JOIN catalog_products p ON p.id=v.product_id WHERE v.id=?",
                (offer["headline_variant_id"],))
            if rows:
                headline = {"variant_id": rows[0]["id"], "title": rows[0]["title"],
                            "price_minor": self.price_of(rows[0]["id"])}
        return {
            "offer_id": offer["id"], "kind": offer["kind"], "code": offer["code"],
            "discount_percentage": offer["discount_percentage"],
            "max_discount_minor": offer["max_discount_minor"],
            "expires_at": str(offer["expires_at"]), "headline": headline,
        }

    # -- redemption (called from checkout staging) --------------------------------

    def terms_for_stage(self, customer_id: str, subtotal_minor: int) -> tuple[int, str | None]:
        """The discount and promotion a stage should carry, from the stored offer."""
        offer = self.live_offer(customer_id)
        if offer is None or offer["kind"] != "coupon":
            return 0, None
        policy = self.store.rows("SELECT min_cart_minor FROM recovery_policies WHERE id=?",
                                 (offer["policy_id"],))
        if policy and subtotal_minor < int(policy[0]["min_cart_minor"]):
            return 0, None
        return discount_for(offer, subtotal_minor), offer["promotion_id"]

    def mark_redeemed(self, order_id: str) -> None:
        rows = self.store.rows("SELECT customer_id, promotion_id FROM commerce_orders WHERE id=?",
                               (order_id,))
        if not rows or not rows[0]["promotion_id"]:
            return
        self.store.execute(
            "UPDATE recovery_offers SET status='redeemed', redeemed_order_id=?, redeemed_at=? "
            "WHERE promotion_id=? AND customer_id=? AND status='issued'",
            (order_id, now(), rows[0]["promotion_id"], rows[0]["customer_id"]))
        self.store.execute("UPDATE promotions SET status='ended' WHERE id=?",
                           (rows[0]["promotion_id"],))

    def expire(self, *, at: datetime | None = None) -> int:
        at = at or datetime.now(UTC)
        stale = self.store.rows("SELECT id, promotion_id FROM recovery_offers WHERE status='issued' "
                                "AND expires_at<=?", (at.isoformat(),))
        for row in stale:
            self.store.execute("UPDATE recovery_offers SET status='expired' WHERE id=?", (row["id"],))
            if row["promotion_id"]:
                self.store.execute("UPDATE promotions SET status='ended' WHERE id=?", (row["promotion_id"],))
        return len(stale)

    # -- the merchant's numbers ---------------------------------------------------

    def stats(self, days: int = 30) -> dict:
        since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        offers = self.store.rows(
            "SELECT kind, status, COUNT(*) AS n FROM recovery_offers WHERE created_at>=? "
            "GROUP BY kind, status", (since,))
        recovered = self.store.rows(
            "SELECT COALESCE(SUM(o.total_minor),0) AS revenue, COALESCE(SUM(o.discount_minor),0) AS discount, "
            "COUNT(*) AS orders FROM recovery_offers r JOIN commerce_orders o ON o.id=r.redeemed_order_id "
            "WHERE r.created_at>=? AND o.status='paid'", (since,))[0]
        counts: dict[str, int] = {}
        for row in offers:
            counts[f"{row['kind']}_{row['status']}"] = int(row["n"])
        issued = sum(counts.values())
        redeemed = sum(v for k, v in counts.items() if k.endswith("_redeemed"))
        return {
            "window_days": days, "offers_issued": issued, "offers_redeemed": redeemed,
            "coupons": sum(v for k, v in counts.items() if k.startswith("coupon_")),
            "reminders": sum(v for k, v in counts.items() if k.startswith("reminder_")),
            "recovered_revenue_minor": int(recovered["revenue"]),
            "discount_given_minor": int(recovered["discount"]),
            "paid_orders": int(recovered["orders"]),
            "redemption_rate": round(redeemed / issued, 3) if issued else 0.0,
        }


# ---------------------------------------------------------------------------
# Consent
# ---------------------------------------------------------------------------


class MarketingConsent:
    def __init__(self, store: Store) -> None:
        self.store = store

    def get(self, customer_id: str) -> dict:
        rows = self.store.rows("SELECT * FROM marketing_consents WHERE customer_id=?", (customer_id,))
        return {"email_opt_in": bool(rows[0]["email_opt_in"]) if rows else False}

    def set(self, customer_id: str, opt_in: bool) -> dict:
        self.store.execute(
            "INSERT INTO marketing_consents (customer_id,email_opt_in,unsubscribe_token,updated_at) "
            "VALUES (?,?,?,?) ON CONFLICT(customer_id) DO UPDATE SET email_opt_in=excluded.email_opt_in, "
            "updated_at=excluded.updated_at",
            (customer_id, opt_in, secrets.token_urlsafe(24), now()))
        return self.get(customer_id)

    def token(self, customer_id: str) -> str | None:
        rows = self.store.rows("SELECT unsubscribe_token FROM marketing_consents WHERE customer_id=?",
                               (customer_id,))
        return rows[0]["unsubscribe_token"] if rows else None

    def unsubscribe(self, token: str) -> bool:
        rows = self.store.rows("SELECT customer_id FROM marketing_consents WHERE unsubscribe_token=?",
                               (token,))
        if not rows:
            return False
        self.store.execute("UPDATE marketing_consents SET email_opt_in=?, updated_at=? WHERE customer_id=?",
                           (False, now(), rows[0]["customer_id"]))
        return True


# ---------------------------------------------------------------------------
# Email delivery
# ---------------------------------------------------------------------------


class EmailSender(Protocol):
    enabled: bool

    async def send(self, *, to: str, subject: str, html_body: str, headers: dict[str, str]) -> None: ...


class ResendSender:
    """Resend's REST API. Off unless RESEND_API_KEY and RESEND_FROM are set."""

    def __init__(self, api_key: str, sender: str,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.api_key, self.sender, self._transport = api_key, sender, transport
        self.enabled = bool(api_key and sender)

    @classmethod
    def from_env(cls) -> ResendSender:
        return cls(os.getenv("RESEND_API_KEY", ""), os.getenv("RESEND_FROM", ""))

    async def send(self, *, to: str, subject: str, html_body: str, headers: dict[str, str]) -> None:
        async with httpx.AsyncClient(timeout=15, transport=self._transport) as client:
            response = await client.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"from": self.sender, "to": [to], "subject": subject, "html": html_body,
                      "headers": headers})
        if response.status_code >= 400:
            raise RuntimeError(f"Resend refused the email: {response.status_code}")


class RecoveryEmailWorker:
    """Sends an issued offer by email when the customer consented, and records what
    happened on the offer either way. The on-site banner never depends on this."""

    def __init__(self, store: Store, outbox: Outbox, offers: RecoveryOffers,
                 consent: MarketingConsent, sender: EmailSender, *, site_url: str,
                 api_url: str) -> None:
        self.store, self.outbox, self.offers = store, outbox, offers
        self.consent, self.sender = consent, sender
        self.site_url, self.api_url = site_url.rstrip("/"), api_url.rstrip("/")

    async def drain(self, limit: int = 20) -> list[dict]:
        results = []
        for message in self.outbox.claim(limit=limit, topic=EMAIL_TOPIC):
            offer = self.offers.get(message["payload"]["offer_id"])
            try:
                status = await self._deliver(offer) if offer else "skipped"
            except Exception as exc:
                logger.warning("recovery email failed", exc_info=True)
                outcome = self.outbox.failed(message["id"], str(exc)[:300], retry_in_seconds=600)
                if outcome == "dead_letter" and offer:
                    self._mark(offer["id"], "failed")
                results.append({"offer_id": offer and offer["id"], "status": "retry"})
                continue
            self.outbox.delivered(message["id"])
            if offer:
                self._mark(offer["id"], status)
            results.append({"offer_id": offer and offer["id"], "status": status})
        return results

    async def _deliver(self, offer: dict) -> str:
        if offer["status"] != "issued" or not self.sender.enabled:
            return "skipped"
        if not self.consent.get(offer["customer_id"])["email_opt_in"]:
            return "skipped"
        rows = self.store.rows("SELECT email, display_name FROM customers WHERE id=?",
                               (offer["customer_id"],))
        token = self.consent.token(offer["customer_id"])
        if not rows or not rows[0]["email"] or not token:
            return "skipped"
        view = self.offers.view(offer) or {}
        unsubscribe = f"{self.api_url}/marketing/unsubscribe?token={token}"
        subject, body = render_email(view, rows[0].get("display_name"), self.site_url, unsubscribe)
        await self.sender.send(
            to=rows[0]["email"], subject=subject, html_body=body,
            headers={"List-Unsubscribe": f"<{unsubscribe}>",
                     "List-Unsubscribe-Post": "List-Unsubscribe=One-Click"})
        return "sent"

    def _mark(self, offer_id: str, status: str) -> None:
        self.store.execute("UPDATE recovery_offers SET email_status=? WHERE id=?", (status, offer_id))


def render_email(view: dict, name: str | None, site_url: str, unsubscribe_url: str) -> tuple[str, str]:
    esc = html.escape
    headline = view.get("headline") or {}
    item = esc(headline.get("title") or "the items in your cart")
    greeting = f"Hi {esc(name)}," if name else "Hi,"
    expires = as_datetime(view.get("expires_at"))
    until = expires.strftime("%d %b, %H:%M UTC") if expires else "soon"
    if view.get("kind") == "coupon":
        cap = (view.get("max_discount_minor") or 0) // 100
        subject = f"{view['discount_percentage']}% off {headline.get('title') or 'your cart'} — just for you"
        offer_html = (f"<p>Here's <strong>{view['discount_percentage']}% off</strong> (up to ₹{cap:,}) "
                      f"to finish your order. It applies automatically at checkout — code "
                      f"<strong>{esc(view['code'] or '')}</strong> — until {until}.</p>")
    else:
        subject = f"Still thinking about {headline.get('title') or 'your cart'}?"
        offer_html = "<p>Your cart is saved and ready whenever you are.</p>"
    body = (f"<div style=\"font-family:system-ui,sans-serif;max-width:520px\">"
            f"<p>{greeting}</p><p>You left <strong>{item}</strong> in your Cartisan cart.</p>"
            f"{offer_html}<p><a href=\"{esc(site_url)}/storefront\" "
            f"style=\"background:#0f5c4f;color:#fff;padding:10px 16px;border-radius:8px;"
            f"text-decoration:none\">Return to your cart</a></p>"
            f"<p style=\"color:#8a8a84;font-size:12px\">You're getting this because you opted in to "
            f"offers from Cartisan. <a href=\"{esc(unsubscribe_url)}\">Unsubscribe</a>.</p></div>")
    return subject, body

