"""Phase 5 acceptance: the golden purchase, and every way it can fail safely.

The acceptance sentence is "without duplication or false paid state", and a happy
path proves neither. So the golden purchase is one test here and the other twenty
are the ways a purchase goes wrong: a webhook delivered twice, a webhook whose
amount or currency or reference does not match, a browser redirect with nothing
behind it, an expired hold, a decline followed by a retry, a double-tapped
confirmation, a provider that times out and is retried, and a paid event arriving
after the order was already cancelled.

Every one of them asserts two things: how many orders exist, and whether anything
reads as paid.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from marketplace_backend.checkout import PaymentVerificationMismatch
from marketplace_backend.identity import Principal
from marketplace_backend.payments import PaymentLinkDispatcher, verify_signature
from marketplace_backend.shopping import CheckoutRefused

from conftest_runtime import (
    CUSTOMER,
    GOOD_CHARGER,
    LAPTOP,
    FakeGateway,
    build_shopping,
    build_store,
    paid_event,
    signed_event,
)

LAPTOP_PRICE = 8_499_00
SECRET = "whsec-test"


@pytest.fixture
def world(tmp_path):
    return build_shopping(build_store(tmp_path))


def run(coro):
    return asyncio.run(coro)


def buy(world, variant_id: str = LAPTOP, quantity: int = 1) -> dict:
    """Cart, stage and confirm — the point at which one order exists and stock is held."""
    run(world.service.add(CUSTOMER, variant_id, quantity))
    stage = run(world.service.stage(CUSTOMER))
    return run(world.service.confirm(CUSTOMER, stage["stage_id"]))


def orders(world) -> list[dict]:
    return world.store.rows("SELECT * FROM commerce_orders ORDER BY created_at")


def paid_orders(world) -> list[dict]:
    return [o for o in orders(world) if o["status"] == "paid"]


def sellable(world, variant_id: str = LAPTOP) -> int:
    return world.inventory.sellable(variant_id)


# ------------------------------------------------------------ golden purchase


def test_golden_purchase_reserves_then_consumes_exactly_once(world):
    before = sellable(world)

    result = buy(world)
    order_id = result["order"]["order_id"]

    # Confirmation, and only confirmation, holds the stock (ADR 0012).
    assert result["order"]["status"] == "pending_payment"
    assert result["order"]["paid"] is False
    assert sellable(world) == before - 1
    assert result["payment"]["pay_url"] == "https://rzp.io/test/1"
    assert world.gateway.calls == [
        {"amount": LAPTOP_PRICE,
         "reference_id": f"order:{order_id}:{result['payment']['attempt_id']}"}
    ]

    outcome = world.webhooks.process(
        paid_event(result["payment"]["provider_reference"], LAPTOP_PRICE))

    assert outcome["result"] == "applied"
    order = world.service.order(CUSTOMER, order_id)
    assert order["status"] == "paid" and order["paid"] is True
    assert order["amount_paid_minor"] == LAPTOP_PRICE
    # The hold became a sale: reserved fell and so did on_hand, exactly once.
    assert sellable(world) == before - 1
    assert world.inventory.reconcile(LAPTOP)["balanced"]
    assert len(orders(world)) == 1


def test_a_live_purchase_is_labelled_live_app_not_test_mode(world):
    """Phase 7's audit views filter on `origin`. An order a person placed in the app
    is `live_app`; `razorpay_test` labels evidence that came from the provider."""
    result = buy(world)

    assert result["order"]["origin"] == "live_app"
    events = world.store.rows(
        "SELECT origin FROM commerce_events WHERE event_type='order_created'")
    assert [row["origin"] for row in events] == ["live_app"]


def test_cart_is_retired_only_after_the_order_exists(world):
    run(world.service.add(CUSTOMER, LAPTOP, 1))
    first_cart = run(world.service.cart(CUSTOMER))["cart_id"]

    buy_result = buy(world)

    assert buy_result["order"]["order_id"]
    assert world.store.rows(
        "SELECT status FROM customer_carts WHERE id=?", (first_cart,))[0]["status"] == "checked_out"


# -------------------------------------------------------------- duplication


def test_a_redelivered_webhook_pays_the_order_once(world):
    result = buy(world)
    reference = result["payment"]["provider_reference"]
    event = paid_event(reference, LAPTOP_PRICE, event_id="evt_dup")

    first = world.webhooks.process(event)
    second = world.webhooks.process(event)
    third = world.webhooks.process(event)

    assert first["result"] == "applied"
    assert second["result"] == third["result"] == "duplicate"
    assert len(paid_orders(world)) == 1
    # One payment, one consumption, one revenue event — not three.
    assert len(world.store.rows(
        "SELECT id FROM commerce_events WHERE event_type='order_paid'")) == 1
    assert len(world.store.rows(
        "SELECT id FROM inventory_reservations WHERE status='consumed'")) == 1
    assert len(world.store.rows("SELECT id FROM inbox_events")) == 1
    assert world.inventory.reconcile(LAPTOP)["balanced"]


def test_a_second_event_id_for_a_settled_attempt_is_quarantined(world):
    """Deduplication keys on the provider's event id, so a *different* id for the
    same attempt gets past the inbox. The attempt's state machine is what stops it."""
    result = buy(world)
    reference = result["payment"]["provider_reference"]

    world.webhooks.process(paid_event(reference, LAPTOP_PRICE, event_id="evt_a"))
    replay = world.webhooks.process(paid_event(reference, LAPTOP_PRICE, event_id="evt_b"))

    assert replay["result"] == "quarantined"
    assert len(paid_orders(world)) == 1
    assert world.store.rows("SELECT amount_paid_minor FROM commerce_orders"
                            )[0]["amount_paid_minor"] == LAPTOP_PRICE


def test_double_confirmation_of_one_stage_creates_one_order(world):
    run(world.service.add(CUSTOMER, LAPTOP, 1))
    stage = run(world.service.stage(CUSTOMER))
    before = sellable(world)

    run(world.service.confirm(CUSTOMER, stage["stage_id"]))
    with pytest.raises(CheckoutRefused):
        run(world.service.confirm(CUSTOMER, stage["stage_id"]))

    assert len(orders(world)) == 1
    # The refused second confirmation held nothing extra.
    assert sellable(world) == before - 1


def test_a_replayed_confirmation_returns_the_first_order(world):
    run(world.service.add(CUSTOMER, LAPTOP, 1))
    stage = run(world.service.stage(CUSTOMER))

    first = run(world.service.confirm(CUSTOMER, stage["stage_id"], idempotency_key="c1"))
    second = run(world.service.confirm(CUSTOMER, stage["stage_id"], idempotency_key="c1"))

    assert first == second
    assert len(orders(world)) == 1
    assert len(world.gateway.calls) == 1


def test_one_order_never_gets_two_payment_links(world):
    result = buy(world)
    order_id = result["order"]["order_id"]

    again = run(world.service.open_payment(CUSTOMER, order_id))

    # The live attempt is reused rather than replaced: a second call while the first
    # is still `created`/`pending` returns it untouched, so a customer clicking
    # "checkout" twice in a row cannot end up with two attempts or two links.
    assert again["attempt_id"] == result["payment"]["attempt_id"]
    assert again["pay_url"] == result["payment"]["pay_url"]
    assert len(world.store.rows("SELECT id FROM payment_attempts")) == 1


# ---------------------------------------------------------- false paid state


def test_a_browser_redirect_alone_never_reads_as_paid(world):
    """The customer came back from Razorpay. That is not money (ADR 0013)."""
    result = buy(world)
    order_id = result["order"]["order_id"]

    returned = world.service.redirect_returned(CUSTOMER, order_id)

    assert returned["status"] == "payment_verification_pending"
    assert returned["paid"] is False
    assert returned["amount_paid_minor"] == 0
    assert paid_orders(world) == []
    # And no fulfilment-shaped evidence was produced from browser state.
    assert world.store.rows(
        "SELECT id FROM commerce_events WHERE event_type='order_paid'") == []


def test_a_short_payment_is_refused_rather_than_applied(world):
    result = buy(world)

    outcome = world.webhooks.process(
        paid_event(result["payment"]["provider_reference"], LAPTOP_PRICE - 100))

    assert outcome["result"] == "quarantined"
    assert "does not equal order total" in outcome["reason"]
    assert paid_orders(world) == []
    assert world.store.rows("SELECT status FROM inbox_events")[0]["status"] == "quarantined"


def test_an_overpayment_is_also_refused(world):
    result = buy(world)

    outcome = world.webhooks.process(
        paid_event(result["payment"]["provider_reference"], LAPTOP_PRICE + 1))

    assert outcome["result"] == "quarantined"
    assert paid_orders(world) == []


def test_a_currency_mismatch_is_refused(world):
    result = buy(world)

    outcome = world.webhooks.process(
        paid_event(result["payment"]["provider_reference"], LAPTOP_PRICE, currency="USD"))

    assert outcome["result"] == "quarantined"
    assert "currency" in outcome["reason"]
    assert paid_orders(world) == []


def test_an_unknown_provider_reference_is_refused(world):
    buy(world)

    outcome = world.webhooks.process(paid_event("plink_not_ours", LAPTOP_PRICE))

    assert outcome["result"] == "quarantined"
    assert "no payment attempt" in outcome["reason"]
    assert paid_orders(world) == []


def test_a_reference_belonging_to_another_order_cannot_pay_this_one(world):
    """The classic cross-order confusion: a real link id, the wrong order's amount."""
    first = buy(world)
    world.store.execute(
        "UPDATE customer_carts SET status='active' WHERE customer_id=?", (CUSTOMER,))
    run(world.service.add(CUSTOMER, GOOD_CHARGER, 1))
    stage = run(world.service.stage(CUSTOMER))
    second = run(world.service.confirm(CUSTOMER, stage["stage_id"]))

    # The first order's link, carrying the second order's total.
    outcome = world.webhooks.process(
        paid_event(first["payment"]["provider_reference"],
                   second["order"]["total_minor"], event_id="evt_cross"))

    assert outcome["result"] == "quarantined"
    assert paid_orders(world) == []


def test_a_paid_event_after_cancellation_is_refused(world):
    result = buy(world)
    order_id = result["order"]["order_id"]
    world.checkout.cancel(order_id, reason="Customer abandoned the checkout")

    outcome = world.webhooks.process(
        paid_event(result["payment"]["provider_reference"], LAPTOP_PRICE))

    assert outcome["result"] == "quarantined"
    assert world.service.order(CUSTOMER, order_id)["paid"] is False
    assert world.store.rows("SELECT status FROM commerce_orders")[0]["status"] == "cancelled"


def test_signature_verification_fails_closed_without_a_secret():
    body, signature = signed_event({"event": "payment_link.paid"}, SECRET)

    assert verify_signature(body, signature, SECRET) is True
    assert verify_signature(body, signature, "") is False
    assert verify_signature(body, "", SECRET) is False
    assert verify_signature(b"tampered", signature, SECRET) is False


# ------------------------------------------------- decline, retry, expiry


def test_a_decline_leaves_one_order_that_can_be_retried(world):
    before = sellable(world)
    result = buy(world)
    order_id = result["order"]["order_id"]

    declined = world.webhooks.process(paid_event(
        result["payment"]["provider_reference"], LAPTOP_PRICE,
        event_id="evt_fail", event="payment_link.cancelled"))
    assert declined["result"] == "applied"

    after_decline = world.service.order(CUSTOMER, order_id)
    assert after_decline["paid"] is False
    assert after_decline["status"] == "pending_payment"
    # The hold survives the decline, so the retry needs no restaging (ADR 0030).
    assert sellable(world) == before - 1

    retry = run(world.service.open_payment(CUSTOMER, order_id))
    assert retry["attempt_id"] != result["payment"]["attempt_id"]
    assert len(orders(world)) == 1

    paid = world.webhooks.process(
        paid_event(retry["provider_reference"], LAPTOP_PRICE, event_id="evt_retry_ok"))

    assert paid["result"] == "applied"
    assert world.service.order(CUSTOMER, order_id)["paid"] is True
    assert len(orders(world)) == 1
    assert len(world.store.rows("SELECT id FROM payment_attempts")) == 2
    assert sellable(world) == before - 1
    assert world.inventory.reconcile(LAPTOP)["balanced"]


def test_a_retry_after_decline_gets_its_own_link_not_the_dead_one(world):
    """The provider is keyed per ATTEMPT, not per order: a retry must get a genuinely
    new link, because the first one may already be dead at the provider (Razorpay
    cancels or expires the link itself, not just the payment, for some declines).
    Reusing the order as the whole key would recover that dead link and hand the
    customer something that fails the instant they click it."""
    result = buy(world)
    order_id = result["order"]["order_id"]
    world.webhooks.process(paid_event(
        result["payment"]["provider_reference"], LAPTOP_PRICE,
        event_id="evt_f", event="payment_link.cancelled"))
    assert world.service.order(CUSTOMER, order_id)["attempts"][-1]["status"] == "failed"

    retry = run(world.service.open_payment(CUSTOMER, order_id))

    assert retry["attempt_id"] != result["payment"]["attempt_id"]
    assert retry["pay_url"] != result["payment"]["pay_url"]
    assert {call["reference_id"] for call in world.gateway.calls} == {
        f"order:{order_id}:{result['payment']['attempt_id']}",
        f"order:{order_id}:{retry['attempt_id']}",
    }


def test_an_expired_reservation_releases_stock_and_cancels_the_order(world):
    before = sellable(world)
    result = buy(world)
    order_id = result["order"]["order_id"]
    assert sellable(world) == before - 1

    swept = world.checkout.expire_unpaid(now="2999-01-01T00:00:00+00:00")

    assert order_id in swept["orders_cancelled"]
    assert sellable(world) == before
    order = world.service.order(CUSTOMER, order_id)
    assert order["status"] == "cancelled" and order["paid"] is False
    assert world.inventory.reconcile(LAPTOP)["balanced"]

    # And running it again changes nothing.
    assert world.checkout.expire_unpaid(now="2999-01-01T00:00:00+00:00")["orders_cancelled"] == []
    assert sellable(world) == before


def test_expiry_leaves_a_verification_pending_order_alone(world):
    """A verified event may still be in flight; expiring here would cancel an order
    the provider is about to report as paid."""
    result = buy(world)
    order_id = result["order"]["order_id"]
    world.service.redirect_returned(CUSTOMER, order_id)

    swept = world.checkout.expire_unpaid(now="2999-01-01T00:00:00+00:00")

    assert swept["orders_cancelled"] == []
    assert world.service.order(CUSTOMER, order_id)["status"] == "payment_verification_pending"


def test_confirming_an_expired_preview_creates_nothing(world):
    run(world.service.add(CUSTOMER, LAPTOP, 1))
    stage = run(world.service.stage(CUSTOMER))
    before = sellable(world)
    world.store.execute(
        "UPDATE checkout_stages SET expires_at='2000-01-01T00:00:00+00:00' WHERE id=?",
        (stage["stage_id"],))

    with pytest.raises(CheckoutRefused, match="expired"):
        run(world.service.confirm(CUSTOMER, stage["stage_id"]))

    assert orders(world) == []
    assert sellable(world) == before


def test_confirming_after_the_cart_changed_creates_nothing(world):
    run(world.service.add(CUSTOMER, LAPTOP, 1))
    stage = run(world.service.stage(CUSTOMER))
    run(world.service.add(CUSTOMER, GOOD_CHARGER, 1))  # the cart moved on
    before = sellable(world)

    with pytest.raises(CheckoutRefused, match="changed"):
        run(world.service.confirm(CUSTOMER, stage["stage_id"]))

    assert orders(world) == []
    assert sellable(world) == before


def test_confirming_when_stock_ran_out_holds_nothing(world):
    run(world.service.add(CUSTOMER, LAPTOP, 1))
    stage = run(world.service.stage(CUSTOMER))
    world.store.execute(
        "UPDATE inventory_levels SET on_hand=0 WHERE variant_id=?", (LAPTOP,))

    with pytest.raises(CheckoutRefused, match="sold out"):
        run(world.service.confirm(CUSTOMER, stage["stage_id"]))

    # The transaction rolled back whole: no order, and no orphaned hold behind it.
    assert orders(world) == []
    assert world.store.rows("SELECT id FROM inventory_reservations") == []


# ---------------------------------------------------- the provider handoff


def test_a_provider_failure_is_retried_without_a_second_order(tmp_path):
    world = build_shopping(build_store(tmp_path), gateway=FakeGateway(fail_times=1))

    result = buy(world)
    order_id = result["order"]["order_id"]

    # The order exists and the stock is held; only the link is missing.
    assert result["payment"]["pay_url"] is None
    assert result["order"]["status"] == "pending_payment"
    assert world.store.rows(
        "SELECT status FROM outbox_messages")[0]["status"] == "pending"

    world.store.execute("UPDATE outbox_messages SET available_at='2000-01-01T00:00:00+00:00'")
    run(world.dispatcher.drain())

    assert len(orders(world)) == 1
    attempt = world.store.rows("SELECT * FROM payment_attempts")[0]
    assert attempt["status"] == "pending"
    assert attempt["provider_link_url"] == "https://rzp.io/test/1"
    assert world.store.rows(
        "SELECT status FROM outbox_messages")[0]["status"] == "delivered"


def test_the_link_request_only_leaves_if_the_order_committed(world):
    """The outbox message is written inside the confirming transaction, so a refused
    confirmation cannot leave a request to charge someone behind it (ADR 0024)."""
    run(world.service.add(CUSTOMER, LAPTOP, 1))
    stage = run(world.service.stage(CUSTOMER))
    world.store.execute("UPDATE inventory_levels SET on_hand=0 WHERE variant_id=?", (LAPTOP,))

    with pytest.raises(CheckoutRefused):
        run(world.service.confirm(CUSTOMER, stage["stage_id"]))

    assert world.store.rows("SELECT id FROM outbox_messages") == []
    assert world.gateway.calls == []


def test_settlement_checks_every_field_before_it_moves_anything(world):
    """`settle_from_provider` is the only door to `paid`, and it is the one that
    refuses — the webhook processor's quarantine is a consequence, not the check."""
    result = buy(world)
    attempt_id = result["payment"]["attempt_id"]
    reference = result["payment"]["provider_reference"]

    for kwargs in (
        {"provider_reference": "plink_other"},
        {"amount_minor": LAPTOP_PRICE + 1},
        {"currency": "USD"},
    ):
        with pytest.raises(PaymentVerificationMismatch):
            world.checkout.settle_from_provider(
                attempt_id=attempt_id,
                **{"provider_reference": reference, "amount_minor": LAPTOP_PRICE,
                   "currency": "INR", "succeeded": True, "snapshot": {}, **kwargs})

    assert paid_orders(world) == []


def test_an_unrelated_provider_event_is_recorded_and_ignored(world):
    outcome = world.webhooks.process(
        {"id": "evt_noise", "event": "payment_link.partially_paid", "payload": {}})

    assert outcome["result"] == "ignored"
    assert world.store.rows("SELECT status FROM inbox_events")[0]["status"] == "ignored"
    assert paid_orders(world) == []


# ----------------------------------------------- the agent's reach, unchanged


def test_no_shopping_tool_can_pay_confirm_or_release(world):
    """Phase 5 added confirmation, links, capture and release to the *host*. None of
    them became a tool, and the shopping surface still ends at `stage_checkout`."""
    from cartisan_agent.config import FORBIDDEN_TOOLS
    from cartisan_agent.contracts import (
        SHOPPING_MUTATIONS,
        SHOPPING_PRESENTATION,
        SHOPPING_READS,
    )

    names = set(SHOPPING_READS) | set(SHOPPING_MUTATIONS) | set(SHOPPING_PRESENTATION)
    assert "stage_checkout" in names
    assert names.isdisjoint(FORBIDDEN_TOOLS)
    for forbidden in ("confirm_checkout", "create_payment_link", "capture_payment",
                      "mark_order_paid", "release_stock", "refund"):
        assert forbidden not in names

    # And the port the tools run on exposes no such method either.
    from cartisan_agent.ports import CommercePort
    assert not {"confirm", "open_attempt", "settle_from_provider", "cancel"} & set(
        vars(CommercePort))


def test_staging_holds_no_stock(world):
    """The agent may stage. Staging must move nothing (ADR 0012)."""
    before = sellable(world)
    run(world.service.add(CUSTOMER, LAPTOP, 1))

    run(world.service.stage(CUSTOMER))

    assert sellable(world) == before
    assert world.store.rows("SELECT id FROM inventory_reservations") == []
    assert orders(world) == []


def test_staging_an_empty_cart_is_refused(world):
    with pytest.raises(CheckoutRefused, match="empty"):
        run(world.service.stage(CUSTOMER))


def test_one_customer_cannot_confirm_anothers_stage(world):
    run(world.service.add(CUSTOMER, LAPTOP, 1))
    stage = run(world.service.stage(CUSTOMER))
    intruder = "33333333-3333-3333-3333-333333333333"

    with pytest.raises(CheckoutRefused, match="another account"):
        run(world.service.confirm(intruder, stage["stage_id"]))

    assert orders(world) == []


def test_a_redelivered_link_request_recovers_the_existing_link(tmp_path):
    """Razorpay rejects a repeated `reference_id` rather than returning the link it
    already made, so a redelivered outbox message would dead-letter an order that
    has a live link at the provider. The gateway reads it back instead."""

    class CollidingGateway:
        """Behaves the way Razorpay actually behaves."""

        def __init__(self) -> None:
            self.created: dict[str, dict] = {}
            self.creates = 0

        async def create_payment_link(self, *, amount, reference_id, description):
            self.creates += 1
            if reference_id in self.created:
                raise RuntimeError(
                    f"payment link with given reference_id: {reference_id} already exists")
            link = {"id": f"plink_{len(self.created) + 1}",
                    "short_url": f"https://rzp.io/x/{len(self.created) + 1}",
                    "reference_id": reference_id}
            self.created[reference_id] = link
            return link

    class RecoveringGateway(CollidingGateway):
        """What `RazorpayMCPClient` now does: fall back to the existing link."""

        async def create_payment_link(self, *, amount, reference_id, description):
            try:
                return await super().create_payment_link(
                    amount=amount, reference_id=reference_id, description=description)
            except RuntimeError as exc:
                if "already exists" not in str(exc):
                    raise
                return self.created[reference_id]

    world = build_shopping(build_store(tmp_path), gateway=RecoveringGateway())
    result = buy(world)
    order_id = result["order"]["order_id"]

    # Re-enqueue the same request, as a redelivery after a lost response would —
    # same attempt, same key `open_attempt` would actually have generated for it.
    world.outbox.enqueue(
        topic=PaymentLinkDispatcher.topic,
        payload={"attempt_id": result["payment"]["attempt_id"], "order_id": order_id,
                 "amount_minor": result["order"]["total_minor"], "currency": "INR",
                 "idempotency_key": f"order:{order_id}:{result['payment']['attempt_id']}"})
    redelivered = run(world.dispatcher.drain())

    assert [entry["status"] for entry in redelivered] == ["delivered"]
    assert redelivered[0]["link_url"] == result["payment"]["pay_url"]
    assert world.gateway.creates == 2  # it tried twice
    assert len(world.gateway.created) == 1  # and one link exists
    assert len(orders(world)) == 1
    assert paid_orders(world) == []
