"""The Razorpay handoff and the verified return path.

Two halves, deliberately separate:

  `PaymentLinkDispatcher`  drains `razorpay.payment_link.create` outbox messages,
                           calls the provider through the replaceable interface
                           (ADR 0011), and attaches the link to the attempt. The
                           host runs it; no model-reachable path enters here.

  `WebhookProcessor`       takes a signed provider callback, stores it once in the
                           deduplicating inbox, and applies it only when the exact
                           order, amount, currency and provider reference agree.
                           Everything else is quarantined with a reason.

The asymmetry between them is the point of ADR 0013. Creating a link is a request;
only a verified provider outcome is evidence. A browser redirect proves that a
person came back, so it can reach `payment_verification_pending` and no further.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any, Protocol

from .checkout import CheckoutRepository, PaymentVerificationMismatch
from .evidence import Actor, Correlation, EvidenceLedger, Inbox, Outbox
from .state_machines import TransitionError
from .store import Store

PROVIDER = "razorpay"

# The provider events that say something about money. Anything else is stored for
# the audit trail and ignored, rather than guessed at.
PAID_EVENTS = frozenset({"payment_link.paid", "payment.captured"})
FAILED_EVENTS = frozenset(
    {"payment_link.cancelled", "payment_link.expired", "payment.failed"}
)


class PaymentLinkGateway(Protocol):
    """What the dispatcher needs from a provider. `RazorpayMCPClient` satisfies it,
    and so does a test double — which is the whole reason it is stated here."""

    async def create_payment_link(
        self, *, amount: int, reference_id: str, description: str
    ) -> dict: ...


class PaymentLinkDispatcher:
    """Delivers the outbox's link requests. Safe to run repeatedly and concurrently:
    `Outbox.claim` hands one message to one worker, and the provider call carries the
    payment attempt's own id as its `reference_id`, which Razorpay enforces as unique
    per attempt — so a retry after a decline gets a genuinely new link rather than
    recovering one the provider may have already cancelled or expired.

    That per-attempt uniqueness is enforced by *rejecting* a second create for the
    same reference, not by returning the first link, so
    `RazorpayMCPClient.create_payment_link` reads the existing link back on a
    collision. Either way one attempt has one link, which is the property that
    matters for a redelivered outbox message."""

    topic = "razorpay.payment_link.create"

    def __init__(
        self,
        store: Store,
        checkout: CheckoutRepository,
        outbox: Outbox,
        gateway: PaymentLinkGateway,
        ledger: EvidenceLedger,
    ) -> None:
        self.store, self.checkout = store, checkout
        self.outbox, self.gateway, self.ledger = outbox, gateway, ledger

    async def drain(self, limit: int = 10) -> list[dict]:
        """Deliver every due message, and report what happened to each."""
        results = []
        for message in self.outbox.claim(limit=limit, topic=self.topic):
            if message["topic"] != self.topic:
                # Not ours. Put it back rather than burning an attempt on it.
                self.outbox.failed(message["id"], "no handler for this topic")
                continue
            results.append(await self._deliver(message))
        return results

    def _correlation_for(self, attempt_id: str, fallback: str | None) -> Correlation:
        """The journey this delivery belongs to.

        The attempt is the authority — it was written inside the transaction that
        enqueued this message — and the message's own `correlation_id` is what an
        attempt predating the lineage columns still has.
        """
        rows = self.store.rows(
            "SELECT correlation_id, demo_run_id FROM payment_attempts WHERE id=?", (attempt_id,))
        row = rows[0] if rows else {}
        return Correlation(
            correlation_id=row.get("correlation_id") or fallback or Correlation().correlation_id,
            demo_run_id=row.get("demo_run_id"))

    def _already_attached(self, attempt_id: str, provider_reference: str) -> bool:
        rows = self.store.rows(
            "SELECT provider_reference FROM payment_attempts WHERE id=?", (attempt_id,))
        return bool(rows) and rows[0]["provider_reference"] == provider_reference

    async def _deliver(self, message: dict) -> dict:
        payload = message["payload"]
        attempt_id, order_id = payload["attempt_id"], payload["order_id"]
        correlation = self._correlation_for(attempt_id, message["correlation_id"])
        try:
            link = await self.gateway.create_payment_link(
                amount=int(payload["amount_minor"]),
                reference_id=payload["idempotency_key"],
                description=f"Cartisan order {order_id}",
            )
            reference, url = link.get("id"), link.get("short_url") or link.get("url")
            if not reference or not url:
                raise ValueError(f"provider returned no usable link: {link!r}")
            # A redelivery whose link is already recorded is finished work, not a
            # failure: attaching again would fail the attempt's state check
            # (`pending` does not transition to `pending`) and the message would
            # retry until it dead-lettered, leaving a usable link unusable.
            if not self._already_attached(attempt_id, reference):
                self.checkout.attach_provider_link(
                    attempt_id, provider_reference=reference, link_url=url, snapshot=link
                )
        except Exception as exc:  # noqa: BLE001 — every failure is recorded, then retried
            disposition = self.outbox.failed(message["id"], str(exc))
            self.ledger.record(
                actor=Actor("system", None, "shopping"),
                action="create_payment_link",
                reason="Provider call for a confirmed order failed",
                outcome="failed",
                target_type="payment_attempt",
                target_id=attempt_id,
                state_ref={"order_id": order_id, "disposition": disposition},
                policy_checks={"error": str(exc)},
                data_origin="razorpay_test",
                correlation=correlation,
            )
            return {"attempt_id": attempt_id, "status": disposition, "error": str(exc)}

        self.outbox.delivered(message["id"])
        self.ledger.record(
            actor=Actor("system", None, "shopping"),
            action="create_payment_link",
            reason="Confirmed order needs a provider payment link",
            outcome="applied",
            target_type="payment_attempt",
            target_id=attempt_id,
            state_ref={"order_id": order_id, "provider_reference": reference},
            data_origin="razorpay_test",
            correlation=correlation,
        )
        return {"attempt_id": attempt_id, "status": "delivered", "link_url": url}


def verify_signature(raw_body: bytes, signature: str, secret: str) -> bool:
    """Razorpay signs the exact bytes with HMAC-SHA256. An unset secret verifies
    nothing, so it fails closed rather than accepting every caller."""
    if not secret or not signature:
        return False
    expected = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


class WebhookProcessor:
    """The only path from a provider event to `paid`."""

    def __init__(
        self,
        store: Store,
        checkout: CheckoutRepository,
        inbox: Inbox,
        ledger: EvidenceLedger,
    ) -> None:
        self.store, self.checkout, self.inbox, self.ledger = store, checkout, inbox, ledger

    def process(self, event: dict, *, correlation: Correlation | None = None) -> dict:
        """Store the event once, then apply it if — and only if — it matches.

        The return value says which of the four things happened: `duplicate` (seen
        before, nothing re-applied), `applied`, `quarantined` (it did not match the
        order it names), or `ignored` (a real event that says nothing about money).
        """
        correlation = correlation or Correlation()
        event_type = str(event.get("event") or "")
        provider_event_id = _event_id(event)
        row, is_new = self.inbox.receive(
            provider=PROVIDER,
            provider_event_id=provider_event_id,
            event_type=event_type,
            payload=event,
        )
        if not is_new:
            # A redelivery. The first copy already decided the outcome; re-applying
            # would be exactly the duplication this phase has to rule out.
            return {"result": "duplicate", "inbox_id": row["id"], "status": row["status"]}

        if event_type not in PAID_EVENTS | FAILED_EVENTS:
            self.inbox.mark(row["id"], "ignored")
            return {"result": "ignored", "inbox_id": row["id"], "event": event_type}

        entity = _entity(event)
        reference = entity.get("id")
        attempt = self._attempt_for(reference)
        if attempt is None:
            return self._quarantine(
                row["id"],
                f"no payment attempt holds provider reference {reference!r}",
                correlation,
            )

        # From here the event has a home. It adopts the journey that asked for the
        # link rather than the one this HTTP request invented, which is what makes the
        # provider's answer the last step of the customer's story instead of a
        # free-standing row (ADR 0032). An unmatched event above keeps its own id —
        # there is genuinely nothing to join it to, and saying so is the honest view.
        if attempt.get("correlation_id"):
            correlation = Correlation(
                correlation_id=attempt["correlation_id"],
                demo_run_id=attempt.get("demo_run_id"))
        self.store.execute(
            "UPDATE inbox_events SET correlation_id=? WHERE id=?",
            (correlation.correlation_id, row["id"]))

        try:
            order = self.checkout.settle_from_provider(
                attempt_id=attempt["id"],
                provider_reference=reference,
                amount_minor=int(entity.get("amount") or 0),
                currency=str(entity.get("currency") or ""),
                succeeded=event_type in PAID_EVENTS,
                snapshot=entity,
                failure_reason=None if event_type in PAID_EVENTS else event_type,
                correlation=correlation,
            )
        except PaymentVerificationMismatch as exc:
            return self._quarantine(row["id"], str(exc), correlation)
        except TransitionError as exc:
            # The attempt already reached a terminal state — a second event for the
            # same attempt, arriving under a different event id. Not applied.
            return self._quarantine(row["id"], f"attempt is not settleable: {exc}", correlation)

        # A failed attempt settles the attempt and nothing else: the order keeps its
        # reservation so the customer can retry against the same internal order while
        # the hold is valid (ADR 0030). Expiry, not this event, releases stock.
        self.inbox.mark(row["id"], "processed")
        return {
            "result": "applied",
            "inbox_id": row["id"],
            "order_id": order["id"],
            "order_status": order["status"],
        }

    def _quarantine(self, inbox_id: str, reason: str, correlation: Correlation) -> dict:
        self.inbox.mark(inbox_id, "quarantined", quarantine_reason=reason)
        self.ledger.record(
            actor=Actor("provider", PROVIDER, "shopping"),
            action="quarantine_provider_event",
            reason="Provider event did not match the order it claims",
            outcome="blocked",
            target_type="inbox_event",
            target_id=inbox_id,
            policy_checks={"error": reason},
            data_origin="razorpay_test",
            correlation=correlation,
        )
        return {"result": "quarantined", "inbox_id": inbox_id, "reason": reason}

    def _attempt_for(self, provider_reference: str | None) -> dict | None:
        if not provider_reference:
            return None
        rows = self.store.rows(
            "SELECT * FROM payment_attempts WHERE provider=? AND provider_reference=? "
            "ORDER BY created_at DESC LIMIT 1",
            (PROVIDER, provider_reference),
        )
        return rows[0] if rows else None


def _entity(event: dict) -> dict:
    """The payload entity this event is about. Razorpay nests one entity per kind;
    the link is what Cartisan created, so it is preferred where both are present."""
    payload = event.get("payload") or {}
    for key in ("payment_link", "payment", "order"):
        entity = (payload.get(key) or {}).get("entity")
        if isinstance(entity, dict) and entity:
            return entity
    return {}


def _event_id(event: dict) -> str:
    """The provider's own id for this delivery, which is what deduplication keys on.
    Razorpay sends `x-razorpay-event-id`; when the caller has folded it into the body
    we use it, and otherwise we derive a stable id from the event's content so a
    redelivery of the identical body still collapses to one row."""
    for key in ("id", "event_id", "x-razorpay-event-id"):
        value = event.get(key)
        if isinstance(value, str) and value:
            return value
    digest = hashlib.sha256(
        json.dumps(event, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return f"derived_{digest[:32]}"


__all__ = [
    "PaymentLinkDispatcher",
    "PaymentLinkGateway",
    "WebhookProcessor",
    "verify_signature",
    "PROVIDER",
    "PAID_EVENTS",
    "FAILED_EVENTS",
]
