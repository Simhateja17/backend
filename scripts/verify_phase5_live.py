"""Run the Phase 5 checkout paths against the live Supabase Postgres.

The pytest suite runs on SQLite, and a green SQLite suite is not evidence: dialect
differences (timestamptz coming back as `datetime`, real booleans, `%` in LIKE
patterns, alias rewriting) have shipped broken code from this repo before. So the
golden purchase and the failure modes that matter run here against the real
database, on real seeded rows.

Everything this creates is removable: probe rows carry the `ph5_` prefix in their
correlation ids and the run deletes its own orders, stages, attempts, reservations,
movements, events, evidence and inbox rows on the way out — including on failure.

    PYTHONPATH=. .venv/bin/python scripts/verify_phase5_live.py
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from marketplace_backend.checkout import CheckoutRepository  # noqa: E402
from marketplace_backend.evidence import (  # noqa: E402
    CommerceEventLog,
    EvidenceLedger,
    Inbox,
    Outbox,
)
from marketplace_backend.inventory import InventoryRepository  # noqa: E402
from marketplace_backend.payments import PaymentLinkDispatcher, WebhookProcessor  # noqa: E402
from marketplace_backend.shopping import CheckoutRefused, ShoppingService  # noqa: E402
from marketplace_backend.store import Store  # noqa: E402

from cartisan_agent.config import CartisanAgentConfig  # noqa: E402
from cartisan_agent.core_port import CoreCommercePort  # noqa: E402

PROBE_PREFIX = "ph5"


class StubGateway:
    """Deterministic links, so the database paths can be proven without spending a
    provider round trip on every run."""

    def __init__(self) -> None:
        self.links: dict[str, dict] = {}

    async def create_payment_link(self, *, amount: int, reference_id: str, description: str) -> dict:
        if reference_id not in self.links:
            index = len(self.links) + 1
            self.links[reference_id] = {
                "id": f"plink_{PROBE_PREFIX}_{index}",
                "short_url": f"https://rzp.io/{PROBE_PREFIX}/{index}",
                "amount": amount,
                "currency": "INR",
            }
        return self.links[reference_id]


def paid_event(reference: str, amount_minor: int, *, currency: str = "INR",
               event_id: str, event: str = "payment_link.paid") -> dict:
    return {
        "id": event_id,
        "event": event,
        "payload": {"payment_link": {"entity": {
            "id": reference, "amount": amount_minor, "currency": currency}}},
    }


class Probe:
    def __init__(self, store: Store) -> None:
        self.store = store
        self.ledger = EvidenceLedger(store)
        self.outbox, self.inbox = Outbox(store), Inbox(store)
        self.inventory = InventoryRepository(store)
        self.checkout = CheckoutRepository(
            store, self.inventory, self.ledger, self.outbox, CommerceEventLog(store))
        self.port = CoreCommercePort(store, checkout=self.checkout, config=CartisanAgentConfig())
        # The matrix always runs on the stub: it exercises database and state-machine
        # paths, and firing twenty real provider calls to prove them would only buy a
        # rate limit (ADR 0011).
        self.gateway = StubGateway()
        self.dispatcher = PaymentLinkDispatcher(
            store, self.checkout, self.outbox, self.gateway, self.ledger)
        self.service = ShoppingService(store, self.port, self.checkout, self.dispatcher)
        self.webhooks = WebhookProcessor(store, self.checkout, self.inbox, self.ledger)
        self.customer = self.pick_customer()
        self.order_ids: list[str] = []
        self.passed: list[str] = []

    # -- setup ---------------------------------------------------------------

    def pick_customer(self) -> str:
        """An existing verified principal, never a fabricated one.

        `cartisan.customers` is uuid-keyed with a foreign key to `auth.users`: a
        customer row cannot exist without a real Supabase account behind it. That
        constraint is the Phase 1 authority boundary doing its job, so the probe
        borrows a demo principal rather than inventing one — and leaves it in place.
        """
        rows = self.store.rows("SELECT id FROM customers ORDER BY created_at LIMIT 1")
        if not rows:
            raise SystemExit(
                "no verified customer exists; sign in once through the app first")
        return str(rows[0]["id"])

    def pick_variant(self) -> tuple[str, int]:
        """A real seeded variant with stock to spare, and its live price."""
        rows = self.store.rows(
            "SELECT l.variant_id AS variant_id, SUM(l.on_hand - l.reserved) AS free "
            "FROM inventory_levels l JOIN catalog_variants v ON v.id = l.variant_id "
            "WHERE v.status = 'active' GROUP BY l.variant_id "
            "HAVING SUM(l.on_hand - l.reserved) >= 3 ORDER BY l.variant_id LIMIT 1")
        if not rows:
            raise SystemExit("no seeded variant has 3 sellable units; reseed first")
        variant_id = rows[0]["variant_id"]
        return variant_id, self.port.current_price(variant_id)

    # -- the checks ----------------------------------------------------------

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        if not condition:
            raise AssertionError(f"{name}: {detail or 'failed'}")
        self.passed.append(name)
        print(f"  ok  {name}")

    async def cart_and_confirm(self, variant_id: str, quantity: int = 1) -> dict:
        await self.service.add(self.customer, variant_id, quantity)
        stage = await self.service.stage(self.customer)
        result = await self.service.confirm(self.customer, stage["stage_id"])
        self.order_ids.append(result["order"]["order_id"])
        return result

    async def run(self) -> None:
        variant_id, price = self.pick_variant()
        print(f"\nprobe variant {variant_id} at {price} paise\n")

        # 1. Golden purchase.
        before = self.inventory.sellable(variant_id)
        result = await self.cart_and_confirm(variant_id)
        order_id = result["order"]["order_id"]
        self.check("confirmation creates one pending order",
                   result["order"]["status"] == "pending_payment")
        self.check("confirmation is not paid", result["order"]["paid"] is False)
        self.check("confirmation reserves stock",
                   self.inventory.sellable(variant_id) == before - 1,
                   f"{self.inventory.sellable(variant_id)} vs {before - 1}")
        self.check("order is labelled live_app", result["order"]["origin"] == "live_app")
        self.check("a payment link was attached", bool(result["payment"]["pay_url"]))

        reference = result["payment"]["provider_reference"]
        outcome = self.webhooks.process(
            paid_event(reference, price, event_id=f"{PROBE_PREFIX}_paid_{order_id[-8:]}"))
        self.check("a verified webhook pays the order", outcome["result"] == "applied")
        order = self.service.order(self.customer, order_id)
        self.check("the order reads as paid", order["paid"] is True)
        self.check("the hold became a sale",
                   self.inventory.sellable(variant_id) == before - 1)
        self.check("inventory reconciles", self.inventory.reconcile(variant_id)["balanced"],
                   str(self.inventory.reconcile(variant_id)["problems"]))

        # 2. Redelivery.
        again = self.webhooks.process(
            paid_event(reference, price, event_id=f"{PROBE_PREFIX}_paid_{order_id[-8:]}"))
        self.check("a redelivered webhook is a duplicate", again["result"] == "duplicate")
        paid_events = self.store.rows(
            "SELECT id FROM commerce_events WHERE event_type='order_paid' AND subject_id=?",
            (order_id,))
        self.check("the redelivery produced no second payment event", len(paid_events) == 1,
                   f"{len(paid_events)} order_paid events")

        # 3. Mismatched amount, currency and reference.
        second = await self.cart_and_confirm(variant_id)
        second_id = second["order"]["order_id"]
        second_ref = second["payment"]["provider_reference"]
        for label, event in (
            ("short payment", paid_event(second_ref, second["order"]["total_minor"] - 100,
                                         event_id=f"{PROBE_PREFIX}_short_{second_id[-8:]}")),
            ("wrong currency", paid_event(second_ref, second["order"]["total_minor"],
                                          currency="USD",
                                          event_id=f"{PROBE_PREFIX}_curr_{second_id[-8:]}")),
            ("unknown reference", paid_event("plink_not_ours", second["order"]["total_minor"],
                                             event_id=f"{PROBE_PREFIX}_unk_{second_id[-8:]}")),
        ):
            quarantined = self.webhooks.process(event)
            self.check(f"{label} is quarantined", quarantined["result"] == "quarantined",
                       str(quarantined))
        self.check("no false paid state",
                   self.service.order(self.customer, second_id)["paid"] is False)

        # 4. Redirect without a verified event.
        returned = self.service.redirect_returned(self.customer, second_id)
        self.check("a redirect reaches only verification-pending",
                   returned["status"] == "payment_verification_pending")
        self.check("a redirect is not payment", returned["paid"] is False)

        # 5. Decline, then a retry on the same internal order.
        third = await self.cart_and_confirm(variant_id)
        third_id = third["order"]["order_id"]
        declined = self.webhooks.process(paid_event(
            third["payment"]["provider_reference"], third["order"]["total_minor"],
            event_id=f"{PROBE_PREFIX}_fail_{third_id[-8:]}", event="payment_link.cancelled"))
        self.check("a decline settles the attempt", declined["result"] == "applied",
                   str(declined))
        after_decline = self.service.order(self.customer, third_id)
        self.check("a decline leaves the order unpaid and retryable",
                   after_decline["paid"] is False
                   and after_decline["status"] == "pending_payment")
        retry = await self.service.open_payment(self.customer, third_id)
        self.check("a retry is a new attempt, not a new order",
                   retry["attempt_id"] != third["payment"]["attempt_id"]
                   and len(self.service.order(self.customer, third_id)["attempts"]) == 2)

        # 6. Expiry releases the hold.
        free_before = self.inventory.sellable(variant_id)
        self.store.execute(
            "UPDATE inventory_reservations SET expires_at=now() - interval '1 hour' "
            "WHERE order_id=? AND status='held'", (third_id,))
        swept = self.checkout.expire_unpaid()
        self.check("expiry cancels the abandoned order", third_id in swept["orders_cancelled"],
                   str(swept))
        self.check("expiry returns the units",
                   self.inventory.sellable(variant_id) == free_before + 1)
        self.check("inventory still reconciles after expiry",
                   self.inventory.reconcile(variant_id)["balanced"])

        # 7. A paid event for a cancelled order is refused.
        late = self.webhooks.process(paid_event(
            retry["provider_reference"], third["order"]["total_minor"],
            event_id=f"{PROBE_PREFIX}_late_{third_id[-8:]}"))
        self.check("a late paid event for a cancelled order is quarantined",
                   late["result"] == "quarantined", str(late))
        self.check("the cancelled order is still not paid",
                   self.service.order(self.customer, third_id)["paid"] is False)

        # 8. Confirming an expired preview creates nothing.
        await self.service.add(self.customer, variant_id, 1)
        stale = await self.service.stage(self.customer)
        self.store.execute(
            "UPDATE checkout_stages SET expires_at=now() - interval '1 hour' WHERE id=?",
            (stale["stage_id"],))
        count_before = len(self.store.rows(
            "SELECT id FROM commerce_orders WHERE customer_id=?", (self.customer,)))
        try:
            await self.service.confirm(self.customer, stale["stage_id"])
            raise AssertionError("an expired preview was confirmed")
        except CheckoutRefused:
            pass
        self.check("an expired preview creates no order",
                   len(self.store.rows("SELECT id FROM commerce_orders WHERE customer_id=?",
                                       (self.customer,))) == count_before)

    # -- cleanup -------------------------------------------------------------

    def cleanup(self) -> None:
        """Remove every row this run produced, children before parents."""
        stages = [row["id"] for row in self.store.rows(
            "SELECT id FROM checkout_stages WHERE customer_id=?", (self.customer,))]
        orders = [row["id"] for row in self.store.rows(
            "SELECT id FROM commerce_orders WHERE customer_id=?", (self.customer,))]

        for order_id in orders:
            # Release anything still held, so the probe cannot leave stock locked.
            for row in self.store.rows(
                "SELECT id FROM inventory_reservations WHERE order_id=? AND status='held'",
                (order_id,)
            ):
                try:
                    self.inventory.release(row["id"])
                except Exception:  # noqa: BLE001 - cleanup is best-effort past this point
                    pass
            # A *consumed* reservation was a sale: it permanently lowered `on_hand`
            # and wrote a movement explaining it. Deleting the movement without
            # putting the unit back would leave `on_hand` below the sum of its
            # movements — the exact drift `reconcile` exists to catch. So undo the
            # sale before the rows that explain it are removed.
            for row in self.store.rows(
                "SELECT variant_id,location_id,quantity FROM inventory_reservations "
                "WHERE order_id=? AND status='consumed'", (order_id,)
            ):
                try:
                    self.store.execute(
                        "UPDATE inventory_levels SET on_hand=on_hand+?, updated_at=now() "
                        "WHERE variant_id=? AND location_id=?",
                        (row["quantity"], row["variant_id"], row["location_id"]))
                except Exception as exc:  # noqa: BLE001
                    print(f"  !!  could not restore consumed stock: {exc}")

        def wipe(sql: str, params: tuple) -> None:
            try:
                self.store.execute(sql, params)
            except Exception as exc:  # noqa: BLE001
                print(f"  !!  cleanup step failed: {exc}")

        for order_id in orders:
            wipe("DELETE FROM inventory_movements WHERE reference_id IN "
                 "(SELECT id FROM inventory_reservations WHERE order_id=?)", (order_id,))
            wipe("DELETE FROM inventory_reservations WHERE order_id=?", (order_id,))
            wipe("DELETE FROM payment_attempts WHERE order_id=?", (order_id,))
            wipe("DELETE FROM commerce_order_lines WHERE order_id=?", (order_id,))
            wipe("DELETE FROM commerce_events WHERE subject_id=?", (order_id,))
            wipe("DELETE FROM evidence_records WHERE target_id=?", (order_id,))
            # The outbox payload names the order, not the customer, so it has to be
            # swept per order — otherwise a message survives whose attempt is gone,
            # and the next drain fails on an attempt id that no longer exists.
            wipe("DELETE FROM outbox_messages WHERE payload LIKE ?", (f"%{order_id}%",))
        wipe("DELETE FROM commerce_orders WHERE customer_id=?", (self.customer,))
        for stage_id in stages:
            wipe("DELETE FROM checkout_stage_lines WHERE stage_id=?", (stage_id,))
            wipe("DELETE FROM evidence_records WHERE target_id=?", (stage_id,))
        wipe("DELETE FROM checkout_stages WHERE customer_id=?", (self.customer,))
        wipe("DELETE FROM cart_lines WHERE cart_id IN "
             "(SELECT id FROM customer_carts WHERE customer_id=?)", (self.customer,))
        wipe("DELETE FROM customer_carts WHERE customer_id=?", (self.customer,))
        wipe("DELETE FROM idempotency_records WHERE principal_id=?", (self.customer,))
        # `_` is a LIKE wildcard, so the underscore in the `ph5_` prefix is escaped —
        # and an escape character only means anything when ESCAPE declares it.
        wipe("DELETE FROM inbox_events WHERE provider_event_id LIKE ? ESCAPE '!'",
             (f"{PROBE_PREFIX}!_%",))
        wipe("DELETE FROM evidence_records WHERE actor_id=?", (self.customer,))
        # The customer row is NOT deleted: it belongs to a real account, and the
        # probe only borrowed it.
        print("\ncleanup complete")


def main() -> int:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    url = os.getenv("SUPABASE_DATABASE_URL")
    if not url:
        raise SystemExit("SUPABASE_DATABASE_URL is not set; this script only runs live")
    store = Store(database_url=url)
    print(f"connected to {store.backend}")
    if store.backend not in {"postgres", "supabase"}:
        raise SystemExit("refusing to run: this must exercise Postgres, not SQLite")

    probe = Probe(store)
    failed = None
    try:
        asyncio.run(probe.run())
    except Exception:  # noqa: BLE001 - reported below, after cleanup runs
        failed = traceback.format_exc()
    finally:
        probe.cleanup()

    if failed:
        print("\nFAILED\n")
        print(failed)
        return 1
    print(f"\n{len(probe.passed)} live checks passed against Postgres")
    return 0


if __name__ == "__main__":
    sys.exit(main())
