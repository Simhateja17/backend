"""The simulated Paytm payment gateway.

Stands in for a real provider behind the same two seams Razorpay used:

  `SimulatedPaytmGateway`  satisfies `PaymentLinkGateway`. It issues a link to the
                           hosted checkout page (`/pay?link=<link id>` on the frontend)
                           without any network call.
  `SimulatedCheckout`      is what that page talks to. It reads the attempt behind a
                           link, and turns the customer's choice on the page into a
                           provider event handed to `WebhookProcessor` — so an order
                           still becomes `paid` only through the one verified path,
                           with the amount and reference taken from our own records.

No real money moves. The link id is an HMAC of the attempt's reference, which
keeps a redelivered outbox message idempotent (same attempt, same link) and keeps
link ids unguessable from order ids alone.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import uuid

from .store import Store
from .payments import PROVIDER, WebhookProcessor

LINK_PREFIX = "simlink_"
METHODS = frozenset({"upi", "wallet", "credit", "debit", "net"})


def _secret() -> bytes:
    return os.getenv("PAYMENT_SIM_SECRET", "cartisan-payment-simulator").encode()


def _frontend_url() -> str:
    return os.getenv("FRONTEND_URL", "http://localhost:3000").rstrip("/")


def link_id_for(reference_id: str) -> str:
    digest = hmac.new(_secret(), reference_id.encode(), hashlib.sha256).hexdigest()
    return LINK_PREFIX + digest[:24]


class SimulatedPaytmGateway:
    async def create_payment_link(self, *, amount: int, reference_id: str, description: str) -> dict:
        link_id = link_id_for(reference_id)
        return {
            "id": link_id,
            "short_url": f"{_frontend_url()}/pay?link={link_id}",
            "amount": amount,
            "currency": "INR",
            "reference_id": reference_id,
            "description": description,
            "status": "created",
            "simulated": True,
        }


class SimulatedCheckoutError(LookupError):
    pass


class SimulatedCheckout:
    def __init__(self, store: Store, webhooks: WebhookProcessor) -> None:
        self.store, self.webhooks = store, webhooks

    def _attempt(self, link_id: str) -> dict:
        if not link_id.startswith(LINK_PREFIX):
            raise SimulatedCheckoutError("Unknown payment link")
        rows = self.store.rows(
            "SELECT * FROM payment_attempts WHERE provider=? AND provider_reference=? "
            "ORDER BY created_at DESC LIMIT 1",
            (PROVIDER, link_id),
        )
        if not rows:
            raise SimulatedCheckoutError("Unknown payment link")
        return rows[0]

    def summary(self, link_id: str) -> dict:
        """What the hosted page shows: who is paying whom, and how much."""
        attempt = self._attempt(link_id)
        orders = self.store.rows(
            "SELECT id, customer_id FROM commerce_orders WHERE id=?", (attempt["order_id"],))
        order = orders[0] if orders else {"id": attempt["order_id"], "customer_id": None}
        return {
            "link_id": link_id,
            "order_id": order["id"],
            "customer": order.get("customer_id"),
            "merchant": os.getenv("PAYMENT_SIM_MERCHANT", "Cartisan"),
            "amount_minor": int(attempt["amount_minor"]),
            "currency": attempt["currency"],
            "status": attempt["status"],
        }

    def complete(self, link_id: str, *, method: str, succeed: bool) -> dict:
        """The customer pressed pay (or cancelled) on the hosted page.

        The event carries the amount and currency from our own attempt row, the way
        a real provider would echo back what it was asked to collect. A second press
        on an attempt that has already settled is answered from the record, not sent
        as a second event.
        """
        if method not in METHODS:
            raise ValueError(f"Unsupported payment method {method!r}")
        attempt = self._attempt(link_id)
        if attempt["status"] not in {"created", "pending"}:
            return {"result": "already_settled", "order_id": attempt["order_id"],
                    "attempt_status": attempt["status"]}
        event = {
            "id": f"simevt_{uuid.uuid4().hex}",
            "event": "payment_link.paid" if succeed else "payment.failed",
            "payload": {"payment_link": {"entity": {
                "id": link_id,
                "amount": int(attempt["amount_minor"]),
                "currency": attempt["currency"],
                "method": method,
                "simulated": True,
            }}},
        }
        return self.webhooks.process(event)


__all__ = ["SimulatedPaytmGateway", "SimulatedCheckout", "SimulatedCheckoutError", "link_id_for"]
