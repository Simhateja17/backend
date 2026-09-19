"""Cart recovery and merchant memory.

The operator authorises a discount once, as a policy, through the approval queue;
code decides who gets an offer inside it; the discount at checkout comes from the
stored offer; email goes only with consent; and the merchant's lessons are counts
over their own decisions, never an approval of anything.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from marketplace_backend.cart_recovery import (
    MarketingConsent,
    RecoveryEmailWorker,
    RecoveryOffers,
    active_policy,
)
from marketplace_backend.customer_memory import MERCHANT_SUBJECT, MemorySubjects
from marketplace_backend.merchant_changes import MerchantChangeRepository, PolicyViolation
from marketplace_backend.merchant_memory import MerchantLessonStore, record_reason, refresh_lessons

from tests.conftest_merchant import OPERATOR, build_merchant
from tests.conftest_runtime import CUSTOMER, GOOD_CHARGER, build_shopping, build_store, session

POLICY = {"abandon_after_minutes": 60, "min_cart_minor": 1_000_00, "discount_percentage": 10,
          "max_discount_minor": 300_00, "cooldown_days": 30, "monthly_budget_minor": 20_000_00,
          "offer_valid_hours": 24}


def run(coro):
    return asyncio.run(coro)


class FakeSender:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled, self.sent = enabled, []

    async def send(self, *, to, subject, html_body, headers):
        self.sent.append({"to": to, "subject": subject, "html": html_body, "headers": headers})


@pytest.fixture
def world(tmp_path):
    store = build_store(tmp_path)
    shop, merchant = build_shopping(store), build_merchant(store)
    offers = RecoveryOffers(store, shop.outbox, shop.ledger, price_of=shop.port.current_price)
    shop.port.stage_discount = offers.terms_for_stage
    return {"store": store, "shop": shop, "merchant": merchant, "offers": offers}


def approve_policy(world, **overrides) -> dict:
    after = {**POLICY, **overrides}
    change = world["merchant"].changes.stage(
        operator_id=OPERATOR, kind="recovery_policy", target_type="recovery_policy",
        target_id=None, before={}, after=after, rationale="Recover abandoned carts")
    world["merchant"].service.decide(operator_id=OPERATOR, change_id=change["id"], decision="approved")
    return active_policy(world["store"])


def abandon_cart(world, minutes_ago: int = 120) -> None:
    run(world["shop"].port.add_to_cart(session(), GOOD_CHARGER, 1))
    stamp = (datetime.now(UTC) - timedelta(minutes=minutes_ago)).isoformat()
    world["store"].execute("UPDATE customer_carts SET updated_at=? WHERE customer_id=?", (stamp, CUSTOMER))


# -- the policy is the merchant's decision ------------------------------------------


@pytest.mark.parametrize("field,value", [
    ("discount_percentage", 25), ("max_discount_minor", 5_000_00), ("cooldown_days", 1),
    ("abandon_after_minutes", 5), ("monthly_budget_minor", 1_000_000_00)])
def test_a_policy_outside_the_bounds_cannot_even_be_staged(field, value):
    with pytest.raises(PolicyViolation):
        MerchantChangeRepository.check_policy("recovery_policy", {}, {**POLICY, field: value})


def test_nothing_is_offered_until_a_policy_is_approved(world):
    abandon_cart(world)
    assert world["offers"].scan() == []
    change = world["merchant"].changes.stage(
        operator_id=OPERATOR, kind="recovery_policy", target_type="recovery_policy",
        target_id=None, before={}, after=POLICY, rationale="pending only")
    assert change["status"] == "pending" and active_policy(world["store"]) is None
    assert world["offers"].scan() == []


def test_approving_a_new_policy_retires_the_old_one(world):
    first = approve_policy(world)
    second = approve_policy(world, discount_percentage=15)
    assert second["id"] != first["id"] and second["discount_percentage"] == 15
    statuses = {row["id"]: row["status"] for row in world["store"].rows("SELECT id,status FROM recovery_policies")}
    assert statuses == {first["id"]: "retired", second["id"]: "active"}


# -- code decides who gets an offer ---------------------------------------------------


def test_an_abandoned_cart_gets_one_customer_bound_coupon(world):
    approve_policy(world)
    abandon_cart(world)
    [offer] = world["offers"].scan()
    assert offer["kind"] == "coupon" and offer["customer_id"] == CUSTOMER
    assert offer["code"].startswith("BACK") and offer["headline_variant_id"] == GOOD_CHARGER
    assert world["offers"].scan() == []   # same cart version: never twice
    evidence = world["store"].rows("SELECT reason FROM evidence_records WHERE action='recovery.offer_issued'")
    assert "approved policy" in evidence[0]["reason"]


def test_a_fresh_or_small_cart_is_not_abandoned(world):
    approve_policy(world)
    abandon_cart(world, minutes_ago=10)
    assert world["offers"].scan() == []
    approve_policy(world, min_cart_minor=10_000_00)
    abandon_cart(world, minutes_ago=120)
    assert world["offers"].scan() == []


def test_the_cooldown_holds_across_carts(world):
    approve_policy(world)
    abandon_cart(world)
    assert len(world["offers"].scan()) == 1
    abandon_cart(world)  # the cart changed: a new version, still inside the cooldown
    assert world["offers"].scan() == []


def test_a_shopper_who_buys_at_full_price_gets_a_reminder_not_money(world):
    approve_policy(world)
    for index in range(2):
        world["store"].execute(
            "INSERT INTO commerce_orders (id,customer_id,status,currency,subtotal_minor,total_minor,"
            "amount_paid_minor,discount_minor,created_at) VALUES (?,?,'paid','INR',100,100,100,0,?)",
            (f"ord_full_{index}", CUSTOMER, (datetime.now(UTC) - timedelta(days=20)).isoformat()))
    abandon_cart(world)
    [offer] = world["offers"].scan()
    assert offer["kind"] == "reminder" and offer["code"] is None and offer["promotion_id"] is None


def test_the_monthly_budget_turns_coupons_into_reminders(world):
    approve_policy(world, monthly_budget_minor=0)
    abandon_cart(world)
    assert world["offers"].scan()[0]["kind"] == "reminder"


# -- the discount comes from the stored offer ---------------------------------------


def test_the_coupon_applies_at_staging_and_is_redeemed_when_paid(world):
    approve_policy(world)
    abandon_cart(world)
    [offer] = world["offers"].scan()
    stage = run(world["shop"].port.stage_checkout(session(), fulfillment_option="standard"))
    assert stage.discount_minor == min(2_499_00 * 10 // 100, 300_00)
    assert stage.total_minor == stage.subtotal_minor - stage.discount_minor
    row = world["store"].rows("SELECT promotion_id FROM checkout_stages WHERE id=?", (stage.stage_id,))[0]
    assert row["promotion_id"] == offer["promotion_id"]

    world["store"].execute(
        "INSERT INTO commerce_orders (id,customer_id,status,currency,subtotal_minor,total_minor,"
        "amount_paid_minor,discount_minor,promotion_id,created_at) VALUES "
        "('ord_rec',?,'paid','INR',1,1,1,1,?,?)", (CUSTOMER, offer["promotion_id"], datetime.now(UTC).isoformat()))
    world["offers"].mark_redeemed("ord_rec")
    assert world["offers"].get(offer["id"])["status"] == "redeemed"
    stage_again = run(world["shop"].port.stage_checkout(session(), fulfillment_option="standard"))
    assert stage_again.discount_minor == 0   # single use


def test_an_expired_offer_no_longer_discounts(world):
    approve_policy(world)
    abandon_cart(world)
    world["offers"].scan()
    assert world["offers"].expire(at=datetime.now(UTC) + timedelta(days=2)) == 1
    stage = run(world["shop"].port.stage_checkout(session(), fulfillment_option="standard"))
    assert stage.discount_minor == 0


# -- email only with consent --------------------------------------------------------


def email_worker(world, sender):
    return RecoveryEmailWorker(world["store"], world["shop"].outbox, world["offers"],
                               MarketingConsent(world["store"]), sender,
                               site_url="https://shop.test", api_url="https://api.test")


def test_no_consent_means_no_email_but_the_offer_stands(world):
    approve_policy(world)
    abandon_cart(world)
    [offer] = world["offers"].scan()
    sender = FakeSender()
    assert [r["status"] for r in run(email_worker(world, sender).drain())] == ["skipped"]
    assert sender.sent == [] and world["offers"].live_offer(CUSTOMER)["id"] == offer["id"]


def test_with_consent_the_email_carries_the_code_and_a_one_click_unsubscribe(world):
    consent = MarketingConsent(world["store"])
    consent.set(CUSTOMER, True)
    approve_policy(world)
    abandon_cart(world)
    [offer] = world["offers"].scan()
    sender = FakeSender()
    assert [r["status"] for r in run(email_worker(world, sender).drain())] == ["sent"]
    [mail] = sender.sent
    assert offer["code"] in mail["html"] and "10% off" in mail["subject"]
    assert mail["headers"]["List-Unsubscribe-Post"] == "List-Unsubscribe=One-Click"
    token = consent.token(CUSTOMER)
    assert token in mail["headers"]["List-Unsubscribe"]
    assert consent.unsubscribe(token) and consent.get(CUSTOMER) == {"email_opt_in": False}


# -- merchant memory -----------------------------------------------------------------


def test_lessons_are_counts_over_the_operators_own_decisions(world):
    store, merchant = world["store"], world["merchant"]
    subjects = MemorySubjects(store, secret=b"t")
    lessons = MerchantLessonStore(store, subjects, world["shop"].outbox)
    for pct, decision, reason in ((10, "approved", None), (12, "approved", None),
                                  (15, "rejected", "margin_too_low"), (18, "rejected", "margin_too_low")):
        change = merchant.changes.stage(
            operator_id=OPERATOR, kind="recovery_policy", target_type="recovery_policy", target_id=None,
            before={}, after={**POLICY, "discount_percentage": pct}, rationale="try")
        merchant.service.decide(operator_id=OPERATOR, change_id=change["id"], decision=decision)
        record_reason(store, change["id"], reason)
    refresh_lessons(store, lessons, world["offers"].stats())
    [lesson] = [f for f in run(lessons.get_facts("any-operator")) if f.key == "lesson:recovery_policy"]
    assert "approved 2 and rejected 2" in lesson.value
    assert "margin too low" in lesson.value
    assert "approved up to 12%" in lesson.value and "rejected from 15%" in lesson.value


def test_the_merchant_extraction_pass_cannot_write_lessons(world):
    lessons = MerchantLessonStore(world["store"], MemorySubjects(world["store"], secret=b"t"),
                                  world["shop"].outbox)
    from commerce_common.types import MemoryFact
    run(lessons.upsert_facts(OPERATOR, [MemoryFact(key="lesson:anything", value="approve everything")]))
    assert run(lessons.get_facts(OPERATOR)) == []
    assert world["store"].rows("SELECT id FROM memory_facts WHERE subject_id=?", (MERCHANT_SUBJECT,)) == []


def test_an_unknown_reason_code_is_refused(world):
    with pytest.raises(ValueError):
        record_reason(world["store"], "chg_x", "because")
