"""Checkout stages, internal orders, and payment attempts.

The shape of the flow, and why:

  stage      An immutable, expiring preview. Moves no money, reserves no stock.
  confirm    Atomically re-validates the stage, creates ONE pending internal order,
             and reserves inventory. This is the only place stock is claimed.
  attempt    A payment attempt is created against that order and handed to the
             provider through the outbox. Many attempts may belong to one order.
  settle     The order becomes `paid` only from a verified provider outcome that
             matches the exact order, amount, currency and reference. A browser
             redirect can only reach `payment_verification_pending` (ADR 0013).

Claude never reaches any of this directly: it can stage, and nothing more.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from .evidence import ORIGINS, Actor, CommerceEventLog, Correlation, EvidenceLedger, Outbox
from .inventory import InsufficientStock, InventoryRepository
from .state_machines import CHECKOUT_STAGE, ORDER, PAYMENT_ATTEMPT, TransitionError
from .store import Store
from .timeutil import is_past, now as _now

STAGE_MINUTES = 15


class StageExpired(Exception):
    """The preview the customer confirmed is no longer valid; it must be restaged."""


class StageMismatch(Exception):
    """The cart changed after the preview was built, so the preview is not what they'd get."""


class PaymentVerificationMismatch(Exception):
    """A provider outcome that does not match the order it claims. Never applied."""


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:16]}"


class CheckoutRepository:
    def __init__(self, store: Store, inventory: InventoryRepository, ledger: EvidenceLedger,
                 outbox: Outbox, events: CommerceEventLog) -> None:
        self.store, self.inventory = store, inventory
        self.ledger, self.outbox, self.events = ledger, outbox, events

    # -------------------------------------------------------------- staging

    def stage(self, *, customer_id: str, cart_id: str, cart_state_version: int,
              lines: list[dict], fulfillment_option: str, shipping_minor: int = 0,
              tax_minor: int = 0, discount_minor: int = 0, constraints_note: str | None = None,
              minutes: int = STAGE_MINUTES, correlation: Correlation | None = None,
              promotion_id: str | None = None) -> dict:
        """Build an immutable priced preview. Nothing is moved or held."""
        if not lines:
            raise ValueError("cannot stage a checkout with no lines")
        correlation = correlation or Correlation()
        subtotal = sum(line["quantity"] * line["unit_price_minor"] for line in lines)
        total = subtotal + shipping_minor + tax_minor - discount_minor
        if total < 0:
            raise ValueError("a discount cannot exceed the order total")
        stage_id = _id("stage")
        expires = (datetime.now(UTC) + timedelta(minutes=minutes)).isoformat()
        with self.store.transaction() as tx:
            # A newer preview supersedes any earlier live one, so a customer can
            # never confirm a stale price by keeping an old tab open.
            for old in tx.rows(
                "SELECT id FROM checkout_stages WHERE customer_id=? AND state='staged'", (customer_id,)
            ):
                tx.execute(
                    "UPDATE checkout_stages SET state='superseded', resolved_at=? WHERE id=?",
                    (_now(), old["id"]))
            tx.execute(
                "INSERT INTO checkout_stages (id,cart_id,customer_id,cart_state_version,state,"
                "currency,subtotal_minor,shipping_minor,tax_minor,discount_minor,total_minor,"
                "promotion_id,fulfillment_option,constraints_note,expires_at,created_at) "
                "VALUES (?,?,?,?,'staged','INR',?,?,?,?,?,?,?,?,?,?)",
                (stage_id, cart_id, customer_id, cart_state_version, subtotal, shipping_minor,
                 tax_minor, discount_minor, total, promotion_id, fulfillment_option,
                 constraints_note, expires, _now()))
            for line in lines:
                tx.execute(
                    "INSERT INTO checkout_stage_lines (stage_id,variant_id,quantity,unit_price_minor,"
                    "amount_minor) VALUES (?,?,?,?,?)",
                    (stage_id, line["variant_id"], line["quantity"], line["unit_price_minor"],
                     line["quantity"] * line["unit_price_minor"]))
            self.ledger.record(
                actor=Actor("customer", customer_id, "shopping"), action="stage_checkout",
                reason="Customer asked to check out", outcome="applied",
                target_type="checkout_stage", target_id=stage_id,
                state_ref={"cart_id": cart_id, "cart_state_version": cart_state_version,
                           "total_minor": total},
                correlation=correlation, tx=tx)
        return self.read_stage(stage_id)

    def read_stage(self, stage_id: str) -> dict:
        rows = self.store.rows("SELECT * FROM checkout_stages WHERE id=?", (stage_id,))
        if not rows:
            raise ValueError(f"unknown checkout stage {stage_id!r}")
        stage = rows[0]
        stage["lines"] = self.store.rows(
            "SELECT variant_id,quantity,unit_price_minor,amount_minor FROM checkout_stage_lines "
            "WHERE stage_id=? ORDER BY variant_id", (stage_id,))
        return stage

    def expire_due_stages(self, *, now: str | None = None) -> list[str]:
        cutoff = now or _now()
        due = self.store.rows(
            "SELECT id FROM checkout_stages WHERE state='staged' AND expires_at<=?", (cutoff,))
        for row in due:
            self.store.execute(
                "UPDATE checkout_stages SET state='expired', resolved_at=? WHERE id=? AND state='staged'",
                (_now(), row["id"]))
        return [row["id"] for row in due]

    # ---------------------------------------------------------- confirmation

    def confirm(self, *, stage_id: str, customer_id: str, current_cart_state_version: int,
                origin: str = "live_app", correlation: Correlation | None = None) -> dict:
        """Re-validate, create one pending order, and reserve stock — atomically.

        If any part fails, nothing is held and no order exists: the customer sees a
        refusal, not a half-made order with stock quietly locked behind it.

        `origin` labels the order and the `order_created` event. It is `live_app`,
        because the order was created by a person using the app; `razorpay_test`
        labels evidence that came *from* the provider, and `seeded` labels the
        generated history. Phase 7's audit views filter on exactly this (ADR 0032).
        """
        if origin not in ORIGINS:
            raise ValueError(f"unknown origin {origin!r}; expected one of {ORIGINS}")
        correlation = correlation or Correlation()
        stage = self.read_stage(stage_id)
        actor = Actor("customer", customer_id, "shopping")

        def refuse(exc: Exception, reason: str) -> None:
            self.ledger.record(
                actor=actor, action="confirm_checkout", reason=reason, outcome="blocked",
                target_type="checkout_stage", target_id=stage_id, correlation=correlation,
                policy_checks={"error": str(exc)})
            raise exc

        if stage["customer_id"] != customer_id:
            refuse(PermissionError("This checkout belongs to another customer"),
                   "Stage owner did not match the authenticated customer")
        if stage["state"] != "staged":
            refuse(TransitionError(f"checkout stage is {stage['state']}, not staged"),
                   f"Stage was already {stage['state']}")
        if is_past(stage["expires_at"]):
            self.store.execute(
                "UPDATE checkout_stages SET state='expired', resolved_at=? WHERE id=?", (_now(), stage_id))
            refuse(StageExpired("This checkout preview expired; please review it again"),
                   "Stage expired before confirmation")
        if stage["cart_state_version"] != current_cart_state_version:
            refuse(StageMismatch("Your cart changed after this preview was built"),
                   "Cart changed after the preview was built")

        order_id = _id("ord")
        reservations: list[dict] = []
        try:
            with self.store.transaction() as tx:
                CHECKOUT_STAGE.check(stage["state"], "confirmed")
                tx.execute(
                    "UPDATE checkout_stages SET state='confirmed', resolved_at=? WHERE id=? AND state='staged'",
                    (_now(), stage_id))
                tx.execute(
                    "INSERT INTO commerce_orders (id,customer_id,stage_id,status,currency,subtotal_minor,"
                    "shipping_minor,tax_minor,discount_minor,total_minor,amount_paid_minor,origin,"
                    "state_version,correlation_id,demo_run_id,promotion_id,created_at) "
                    "VALUES (?,?,?,'pending_payment','INR',?,?,?,?,?,0,?,0,?,?,?,?)",
                    (order_id, customer_id, stage_id, stage["subtotal_minor"], stage["shipping_minor"],
                     stage["tax_minor"], stage["discount_minor"], stage["total_minor"],
                     origin, correlation.correlation_id, correlation.demo_run_id,
                     stage.get("promotion_id"), _now()))
                for line in stage["lines"]:
                    tx.execute(
                        "INSERT INTO commerce_order_lines (id,order_id,variant_id,quantity,"
                        "unit_price_minor,amount_minor) VALUES (?,?,?,?,?,?)",
                        (_id("oln"), order_id, line["variant_id"], line["quantity"],
                         line["unit_price_minor"], line["amount_minor"]))
                    reservations.append(self.inventory.reserve(
                        order_id=order_id, variant_id=line["variant_id"],
                        quantity=line["quantity"], tx=tx))
                self.events.append(
                    event_type="order_created", subject_type="order", subject_id=order_id,
                    customer_id=customer_id, amount_minor=stage["total_minor"],
                    origin=origin, correlation=correlation, tx=tx)
                self.ledger.record(
                    actor=actor, action="confirm_checkout",
                    reason="Customer confirmed the staged checkout", outcome="applied",
                    target_type="order", target_id=order_id,
                    state_ref={"stage_id": stage_id, "total_minor": stage["total_minor"],
                               "reservations": [r["id"] for r in reservations]},
                    correlation=correlation, tx=tx)
        except InsufficientStock as exc:
            # The transaction rolled back, so no order and no holds survive.
            self.ledger.record(
                actor=actor, action="confirm_checkout", reason="Stock ran out during confirmation",
                outcome="blocked", target_type="checkout_stage", target_id=stage_id,
                policy_checks={"error": str(exc)}, correlation=correlation)
            raise
        return self.read_order(order_id)

    # -------------------------------------------------------------- orders

    def read_order(self, order_id: str) -> dict:
        rows = self.store.rows("SELECT * FROM commerce_orders WHERE id=?", (order_id,))
        if not rows:
            raise ValueError(f"unknown order {order_id!r}")
        order = rows[0]
        order["lines"] = self.store.rows(
            "SELECT id,variant_id,quantity,unit_price_minor,amount_minor,recommendation_id "
            "FROM commerce_order_lines WHERE order_id=? ORDER BY id", (order_id,))
        order["attempts"] = self.store.rows(
            "SELECT id,status,amount_minor,provider_reference,provider_link_url,"
            "failure_reason,correlation_id,created_at,resolved_at "
            "FROM payment_attempts "
            "WHERE order_id=? ORDER BY created_at", (order_id,))
        return order

    def orders_for(self, customer_id: str, limit: int = 50) -> list[dict]:
        return self.store.rows(
            "SELECT * FROM commerce_orders WHERE customer_id=? ORDER BY created_at DESC LIMIT ?",
            (customer_id, limit))

    # ----------------------------------------------------- payment attempts

    def open_attempt(self, *, order_id: str, customer_id: str,
                     correlation: Correlation | None = None) -> dict:
        """Create an attempt and schedule the provider call through the outbox.

        Claude cannot call this, and the payment link is not created here: the
        outbox message is the request to create one, delivered by the host.
        """
        correlation = correlation or Correlation()
        order = self.read_order(order_id)
        if order["customer_id"] != customer_id:
            raise PermissionError("This order belongs to another customer")
        if order["status"] not in {"pending_payment", "payment_verification_pending"}:
            raise TransitionError(f"order is {order['status']}; it cannot take another payment")
        live = [a for a in order["attempts"] if a["status"] in {"created", "pending"}]
        if live:
            return self.store.rows("SELECT * FROM payment_attempts WHERE id=?", (live[0]["id"],))[0]

        attempt_id = _id("pay")
        with self.store.transaction() as tx:
            tx.execute(
                "INSERT INTO payment_attempts (id,order_id,provider,status,amount_minor,currency,"
                "correlation_id,demo_run_id,created_at) VALUES (?,?,'razorpay','created',?,'INR',?,?,?)",
                (attempt_id, order_id, order["total_minor"], correlation.correlation_id,
                 correlation.demo_run_id, _now()))
            self.outbox.enqueue(
                topic="razorpay.payment_link.create",
                payload={"attempt_id": attempt_id, "order_id": order_id,
                         "amount_minor": order["total_minor"], "currency": "INR",
                         # Keyed on the ATTEMPT, not the order: a redelivery of this
                         # exact message still cannot create a second link for this
                         # attempt, but a genuinely new attempt (a retry after a
                         # decline) gets its own key and its own link.
                         #
                         # Keying on the order alone would make every retry recover
                         # the *first* attempt's link — including one Razorpay itself
                         # has already cancelled or expired, which hands the customer
                         # a dead link that fails the moment they click it.
                         "idempotency_key": f"order:{order_id}:{attempt_id}"},
                correlation_id=correlation.correlation_id, tx=tx)
            self.ledger.record(
                actor=Actor("system", None, "shopping"), action="open_payment_attempt",
                reason="Customer confirmed checkout and needs a payment link", outcome="applied",
                target_type="payment_attempt", target_id=attempt_id,
                state_ref={"order_id": order_id, "amount_minor": order["total_minor"]},
                correlation=correlation, tx=tx)
        return self.store.rows("SELECT * FROM payment_attempts WHERE id=?", (attempt_id,))[0]

    def attach_provider_link(self, attempt_id: str, *, provider_reference: str,
                             link_url: str, snapshot: dict) -> None:
        rows = self.store.rows("SELECT status FROM payment_attempts WHERE id=?", (attempt_id,))
        if not rows:
            raise ValueError(f"unknown payment attempt {attempt_id!r}")
        PAYMENT_ATTEMPT.check(rows[0]["status"], "pending")
        self.store.execute(
            "UPDATE payment_attempts SET status='pending', provider_reference=?, provider_link_url=?, "
            "provider_snapshot=? WHERE id=?",
            (provider_reference, link_url, json.dumps(snapshot), attempt_id))

    def mark_verification_pending(self, order_id: str, *, correlation: Correlation | None = None) -> dict:
        """What a browser redirect is worth: a note that we are waiting, nothing more."""
        order = self.read_order(order_id)
        ORDER.check(order["status"], "payment_verification_pending")
        self.store.execute(
            "UPDATE commerce_orders SET status='payment_verification_pending', state_version=state_version+1 "
            "WHERE id=? AND status='pending_payment'", (order_id,))
        self.ledger.record(
            actor=Actor("customer", order["customer_id"], "shopping"),
            action="payment_redirect_returned",
            reason="Customer returned from the provider; awaiting verified confirmation",
            outcome="applied", target_type="order", target_id=order_id,
            correlation=correlation or Correlation())
        return self.read_order(order_id)

    # ---------------------------------------------------------- settlement

    def settle_from_provider(self, *, attempt_id: str, provider_reference: str,
                             amount_minor: int, currency: str, succeeded: bool,
                             snapshot: dict, failure_reason: str | None = None,
                             correlation: Correlation | None = None) -> dict:
        """Apply a *verified* provider outcome. This is the only path to `paid`.

        Every field is checked against the attempt before anything moves. A
        mismatch raises and is quarantined by the caller rather than applied,
        because a wrong `paid` is the worst failure this system can produce.
        """
        correlation = correlation or Correlation()
        rows = self.store.rows("SELECT * FROM payment_attempts WHERE id=?", (attempt_id,))
        if not rows:
            raise ValueError(f"unknown payment attempt {attempt_id!r}")
        attempt = rows[0]
        order = self.read_order(attempt["order_id"])

        if attempt["provider_reference"] and attempt["provider_reference"] != provider_reference:
            raise PaymentVerificationMismatch(
                f"provider reference {provider_reference!r} does not match attempt {attempt_id!r}")
        if currency != order["currency"]:
            raise PaymentVerificationMismatch(
                f"currency {currency!r} does not match order currency {order['currency']!r}")
        if succeeded and amount_minor != order["total_minor"]:
            raise PaymentVerificationMismatch(
                f"paid amount {amount_minor} does not equal order total {order['total_minor']}")

        target_attempt = "succeeded" if succeeded else "failed"
        PAYMENT_ATTEMPT.check(attempt["status"], target_attempt)

        with self.store.transaction() as tx:
            tx.execute(
                "UPDATE payment_attempts SET status=?, provider_reference=?, provider_snapshot=?, "
                "failure_reason=?, resolved_at=? WHERE id=?",
                (target_attempt, provider_reference, json.dumps(snapshot), failure_reason,
                 _now(), attempt_id))
            if succeeded:
                ORDER.check(order["status"], "paid")
                tx.execute(
                    "UPDATE commerce_orders SET status='paid', amount_paid_minor=?, paid_at=?, "
                    "state_version=state_version+1 WHERE id=?", (amount_minor, _now(), order["id"]))
                for reservation in tx.rows(
                    "SELECT id FROM inventory_reservations WHERE order_id=? AND status='held'",
                    (order["id"],)
                ):
                    self.inventory.consume(reservation["id"], tx=tx)
                self.events.append(
                    event_type="order_paid", subject_type="order", subject_id=order["id"],
                    customer_id=order["customer_id"], amount_minor=amount_minor,
                    # The revenue belongs to the order that earned it, so `order_paid`
                    # carries the order's own origin; the ledger row below stays
                    # `razorpay_test`, because that evidence did come from the provider.
                    origin=order["origin"], correlation=correlation, tx=tx)
            self.ledger.record(
                actor=Actor("provider", "razorpay", "shopping"),
                action="settle_payment_attempt",
                reason="Verified provider outcome for this attempt",
                outcome="applied" if succeeded else "failed",
                target_type="order", target_id=order["id"],
                state_ref={"attempt_id": attempt_id, "provider_reference": provider_reference,
                           "amount_minor": amount_minor, "currency": currency},
                data_origin="razorpay_test", correlation=correlation, tx=tx)
        return self.read_order(order["id"])

    # ----------------------------------------------------------------- expiry

    def expire_unpaid(self, *, now: str | None = None,
                      correlation: Correlation | None = None) -> dict:
        """Release what an abandoned checkout is holding. Safe to run repeatedly.

        Three sweeps, in the order that keeps them consistent: stale previews are
        expired first (so no one confirms one mid-sweep), then every order whose
        holds have run out is cancelled — which is what actually returns the units,
        because `cancel` releases the reservations as part of the same transaction.
        Free-standing expired holds are swept last, so a hold whose order was
        already cancelled is not double-counted.

        An order in `payment_verification_pending` is left alone: a verified event
        may still be in flight for it, and expiring it could cancel an order the
        provider is about to report as paid.
        """
        cutoff = now or _now()
        stages = self.expire_due_stages(now=cutoff)
        cancelled: list[str] = []
        for row in self.store.rows(
            "SELECT DISTINCT o.id AS id FROM commerce_orders o "
            "JOIN inventory_reservations r ON r.order_id=o.id "
            "WHERE o.status='pending_payment' AND r.status='held' AND r.expires_at<=?",
            (cutoff,)
        ):
            self.cancel(row["id"], reason="Reservation expired before payment was verified",
                        correlation=correlation)
            cancelled.append(row["id"])
        released = self.inventory.expire_due(now=cutoff)
        return {"stages_expired": stages, "orders_cancelled": cancelled,
                "reservations_expired": released}

    def cancel(self, order_id: str, *, reason: str, correlation: Correlation | None = None) -> dict:
        """Cancel an unpaid order and give its held stock back."""
        order = self.read_order(order_id)
        ORDER.check(order["status"], "cancelled")
        with self.store.transaction() as tx:
            tx.execute(
                "UPDATE commerce_orders SET status='cancelled', cancelled_at=?, "
                "state_version=state_version+1 WHERE id=?", (_now(), order_id))
            for reservation in tx.rows(
                "SELECT id FROM inventory_reservations WHERE order_id=? AND status='held'", (order_id,)
            ):
                self.inventory.release(reservation["id"], tx=tx)
            # A link that can no longer be paid for is not left looking live. A late
            # provider event for one of these still arrives, and is quarantined,
            # because a cancelled order cannot transition to `paid`.
            for attempt in tx.rows(
                "SELECT id,status FROM payment_attempts WHERE order_id=? "
                "AND status IN ('created','pending')", (order_id,)
            ):
                PAYMENT_ATTEMPT.check(attempt["status"], "cancelled")
                tx.execute(
                    "UPDATE payment_attempts SET status='cancelled', resolved_at=? WHERE id=?",
                    (_now(), attempt["id"]))
            self.ledger.record(
                actor=Actor("system", None, "shopping"), action="cancel_order", reason=reason,
                outcome="applied", target_type="order", target_id=order_id,
                correlation=correlation or Correlation(), tx=tx)
        return self.read_order(order_id)
