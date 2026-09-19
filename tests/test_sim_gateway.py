"""The simulated Paytm gateway: its links resolve to our own attempt, and the hosted
page's answer settles an order only through the webhook processor."""

from __future__ import annotations

import asyncio

import pytest

from marketplace_backend.sim_gateway import (
    SimulatedCheckout,
    SimulatedCheckoutError,
    SimulatedPaytmGateway,
    link_id_for,
)

from conftest_runtime import CUSTOMER, LAPTOP, build_shopping, build_store

LAPTOP_PRICE = 8_499_00


@pytest.fixture
def world(tmp_path):
    w = build_shopping(build_store(tmp_path), gateway=SimulatedPaytmGateway())
    w.sim = SimulatedCheckout(w.store, w.webhooks)
    return w


def buy(world) -> dict:
    asyncio.run(world.service.add(CUSTOMER, LAPTOP, 1))
    stage = asyncio.run(world.service.stage(CUSTOMER))
    return asyncio.run(world.service.confirm(CUSTOMER, stage["stage_id"]))


def order_status(world, order_id: str) -> str:
    return world.store.rows("SELECT status FROM commerce_orders WHERE id=?", (order_id,))[0]["status"]


def test_link_is_stable_per_reference_and_points_at_the_hosted_page(monkeypatch):
    monkeypatch.setenv("FRONTEND_URL", "https://shop.example/")
    link = asyncio.run(SimulatedPaytmGateway().create_payment_link(
        amount=100, reference_id="pay_1", description="x"))
    assert link["id"] == link_id_for("pay_1") != link_id_for("pay_2")
    assert link["short_url"] == f"https://shop.example/pay?link={link['id']}"


def test_paying_on_the_hosted_page_marks_the_order_paid(world):
    result = buy(world)
    link_id = result["payment"]["provider_reference"]
    summary = world.sim.summary(link_id)
    assert summary["amount_minor"] == LAPTOP_PRICE and summary["customer"] == CUSTOMER

    outcome = world.sim.complete(link_id, method="upi", succeed=True)
    assert outcome["result"] == "applied"
    assert order_status(world, outcome["order_id"]) == "paid"

    again = world.sim.complete(link_id, method="upi", succeed=True)
    assert again["result"] == "already_settled"
    assert world.store.rows("SELECT id FROM inbox_events WHERE status='quarantined'") == []


def test_a_failed_payment_keeps_the_order_unpaid(world):
    link_id = buy(world)["payment"]["provider_reference"]
    outcome = world.sim.complete(link_id, method="credit", succeed=False)
    assert outcome["result"] == "applied"
    assert order_status(world, outcome["order_id"]) != "paid"


def test_unknown_links_and_methods_are_refused(world):
    link_id = buy(world)["payment"]["provider_reference"]
    with pytest.raises(SimulatedCheckoutError):
        world.sim.summary("simlink_nope")
    with pytest.raises(ValueError):
        world.sim.complete(link_id, method="crypto", succeed=True)
