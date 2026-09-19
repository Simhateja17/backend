"""The Telegram channel: n8n is transport, Cartisan keeps the authority.

Identity comes only from a redeemed link token, buttons resolve only through
server-issued action ids, updates act once, and "paid" is announced only after the
verified webhook has moved the order.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime, timedelta

import pytest

from api import telegram as tg
from cartisan_agent import MerchantSessionContext, MerchantSessionState, SessionContext, SessionState
from commerce_common.streaming import AgentEvent
from marketplace_backend.identity import Principal

from tests.conftest_merchant import OPERATOR, build_merchant, build_merchant_store
from tests.conftest_runtime import CUSTOMER, GOOD_CHARGER, LAPTOP, build_shopping, paid_event

CHAT, USER = "5001", "9001"
SECOND_OPERATOR = "44444444-4444-4444-4444-444444444444"


def run(coro):
    return asyncio.run(coro)


class FakeRuntime:
    def __init__(self, events: list[AgentEvent] | None = None) -> None:
        self.events = events or [AgentEvent.text_delta("Hello from Cartisan")]
        self.calls: list[tuple[str, str]] = []

    async def stream_turn(self, messages, session, state):
        self.calls.append((session.conversation_id, messages[-1]["content"]))
        for event in self.events:
            yield event


@pytest.fixture
def world(tmp_path):
    store = build_merchant_store(tmp_path)
    shop, merchant = build_shopping(store), build_merchant(store)
    shopping_rt, merchant_rt = FakeRuntime(), FakeRuntime()
    channel = tg.TelegramChannel(
        store, shopping_runtime=shopping_rt, merchant_runtime=merchant_rt,
        shopping_service=shop.service, merchant_service=merchant.service,
        shopping_sessions=({}, {}), merchant_sessions=({}, {}),
        session_types={"shopping": (SessionContext, SessionState),
                       "merchant": (MerchantSessionContext, MerchantSessionState)},
        link_base_url="https://shop.test")
    return type("World", (), dict(store=store, shop=shop, merchant=merchant, channel=channel,
                                  shopping_rt=shopping_rt, merchant_rt=merchant_rt))


_update_ids = iter(range(1, 10_000))


def message(text: str, chat: str = CHAT, user: str = USER, update_id: int | None = None) -> dict:
    return {"update_id": update_id or next(_update_ids),
            "message": {"text": text, "chat": {"id": int(chat)}, "from": {"id": int(user)}}}


def tap(callback_data: str, chat: str = CHAT, user: str = USER) -> dict:
    return {"update_id": next(_update_ids),
            "callback_query": {"id": f"cb{next(_update_ids)}", "data": callback_data,
                               "from": {"id": int(user)}, "message": {"chat": {"id": int(chat)}}}}


def buttons(messages: list[dict]) -> list[dict]:
    return [b for m in messages for row in m["body"].get("reply_markup", {}).get("inline_keyboard", [])
            for b in row]


def link(world, bot_kind: str, principal: Principal, user: str = USER, chat: str = CHAT) -> None:
    token = world.channel.issue_link_token(bot_kind, user, chat)
    world.channel.redeem_link_token(token, principal)


CUSTOMER_P = Principal(id=CUSTOMER, email="c@example.test", role="customer")
OPERATOR_P = Principal(id=OPERATOR, email="ops@example.test", role="merchant_operator")
SECOND_P = Principal(id=SECOND_OPERATOR, email="ops2@example.test", role="merchant_operator")


# ------------------------------------------------------------------ signing


def test_signature_binds_body_and_timestamp():
    raw, now = b'{"update_id":1}', str(int(time.time()))
    good = tg.sign("s3cret", now, raw)
    assert tg.verify("s3cret", now, good, raw)
    assert not tg.verify("s3cret", now, good, raw + b" ")
    assert not tg.verify("other", now, good, raw)
    assert not tg.verify("", now, good, raw), "an unconfigured secret closes the channel"
    stale = str(int(time.time()) - 600)
    assert not tg.verify("s3cret", stale, tg.sign("s3cret", stale, raw), raw)


def test_route_refuses_unsigned_and_unknown_merchant(monkeypatch):
    from fastapi.testclient import TestClient
    import api.main as api_main

    monkeypatch.setenv("TELEGRAM_CHANNEL_SECRET", "s3cret")
    client = TestClient(api_main.app)
    raw = json.dumps({"update_id": 1}).encode()
    assert client.post("/channels/telegram/shopping/cartisan", content=raw).status_code == 401

    now = str(int(time.time()))
    headers = {tg.TIMESTAMP_HEADER: now, tg.SIGNATURE_HEADER: tg.sign("s3cret", now, raw)}
    response = client.post("/channels/telegram/shopping/other-store", content=raw, headers=headers)
    assert response.status_code == 404


# ------------------------------------------------------------------ updates


def test_a_redelivered_update_runs_one_turn(world):
    update = message("hi")
    first = run(world.channel.handle("shopping", update))
    second = run(world.channel.handle("shopping", update))
    assert first[0]["body"]["text"] == "Hello from Cartisan"
    assert second == []
    assert len(world.shopping_rt.calls) == 1


def test_guests_can_browse_but_the_merchant_bot_needs_a_link(world):
    run(world.channel.handle("shopping", message("any chargers?")))
    assert world.shopping_rt.calls[0][0].startswith("tg_guest_9001:")

    reply = run(world.channel.handle("merchant", message("sales today?")))
    assert world.merchant_rt.calls == []
    assert buttons(reply)[0]["url"].startswith("https://shop.test/telegram/link?bot=merchant&token=")


def test_new_starts_a_fresh_conversation(world):
    run(world.channel.handle("shopping", message("one")))
    run(world.channel.handle("shopping", message("/new")))
    run(world.channel.handle("shopping", message("two")))
    first, second = (call[0] for call in world.shopping_rt.calls)
    assert first != second


# ------------------------------------------------------------------ linking


def test_link_tokens_are_single_use_role_checked_and_expire(world):
    token = world.channel.issue_link_token("merchant", USER, CHAT)
    with pytest.raises(tg.LinkRefused, match="operator"):
        world.channel.redeem_link_token(token, CUSTOMER_P)
    world.channel.redeem_link_token(token, OPERATOR_P)
    assert world.channel.linked_principal("merchant", USER) == OPERATOR
    with pytest.raises(tg.LinkRefused, match="already used"):
        world.channel.redeem_link_token(token, OPERATOR_P)

    expired = world.channel.issue_link_token("shopping", USER, CHAT)
    world.store.execute("UPDATE telegram_link_tokens SET expires_at=? WHERE redeemed_at IS NULL",
                        ((datetime.now(UTC) - timedelta(minutes=1)).isoformat(),))
    with pytest.raises(tg.LinkRefused, match="expired"):
        world.channel.redeem_link_token(expired, CUSTOMER_P)

    with pytest.raises(tg.LinkRefused, match="not valid"):
        world.channel.redeem_link_token("made-up-token-value", CUSTOMER_P)


def test_linking_carries_the_guest_cart_over(world):
    run(world.shop.service.add("tg_guest_9001", LAPTOP, 1))
    link(world, "shopping", CUSTOMER_P)
    cart = run(world.shop.service.cart(CUSTOMER))
    assert [line["variant_id"] for line in cart["lines"]] == [LAPTOP]


# ------------------------------------------------------------------ buttons


def product_buttons(world, chat: str = CHAT) -> list[dict]:
    rendered = world.channel.render("shopping", chat, [AgentEvent.ui("products", {"items": [
        {"variant_id": GOOD_CHARGER, "title": "65W charger", "price": "₹2,499", "in_stock": True}]})])
    return buttons(rendered)


def test_add_to_cart_button_writes_through_the_shopping_service(world):
    [add] = product_buttons(world)
    assert add["callback_data"].startswith("tga_") and GOOD_CHARGER not in add["callback_data"]
    reply = run(world.channel.handle("shopping", tap(add["callback_data"])))
    assert reply[0]["method"] == "answerCallbackQuery"
    cart = run(world.shop.service.cart("tg_guest_9001"))
    assert [line["variant_id"] for line in cart["lines"]] == [GOOD_CHARGER]


def test_forged_or_foreign_buttons_address_nothing(world):
    [add] = product_buttons(world, chat="7777")
    for update in (tap("tga_forged"), tap(add["callback_data"], chat=CHAT)):
        reply = run(world.channel.handle("shopping", update))
        assert reply == [tg._answer(update["callback_query"]["id"], "That button is no longer valid.")]
    assert run(world.shop.service.cart("tg_guest_9001"))["lines"] == []


def checkout_button(world) -> dict:
    stage = run(world.shop.service.stage(CUSTOMER))
    [confirm] = buttons(world.channel.render("shopping", CHAT, [AgentEvent.ui("checkout", {
        "stage_id": stage["stage_id"], "total": "₹8,499", "lines": []})]))
    return confirm


def test_checkout_needs_a_link_and_paid_is_announced_only_after_the_webhook(world):
    run(world.shop.service.add(CUSTOMER, LAPTOP, 1))
    confirm = checkout_button(world)

    guest = run(world.channel.handle("shopping", tap(confirm["callback_data"])))
    assert "link" in buttons(guest)[0]["url"]
    assert world.store.rows("SELECT id FROM commerce_orders WHERE customer_id=?", (CUSTOMER,)) == []

    link(world, "shopping", CUSTOMER_P)
    reply = run(world.channel.handle("shopping", tap(confirm["callback_data"])))
    assert buttons(reply)[0]["url"] == "https://rzp.io/test/1"
    again = run(world.channel.handle("shopping", tap(confirm["callback_data"])))
    assert buttons(again)[0]["url"] == "https://rzp.io/test/1"
    assert len(world.store.rows("SELECT id FROM commerce_orders WHERE customer_id=?", (CUSTOMER,))) == 1

    relayed = [d["body"]["text"] for d in world.channel.relay()]
    assert not any("Payment verified" in text for text in relayed), "a link is not a payment"

    attempt = world.store.rows("SELECT * FROM payment_attempts")[0]
    world.shop.webhooks.process(paid_event(attempt["provider_reference"], attempt["amount_minor"]))
    paid = [d for d in world.channel.relay() if "Payment verified" in d["body"]["text"]]
    assert len(paid) == 1 and paid[0]["bot_kind"] == "shopping"
    assert world.channel.relay() == [], "each notification goes out once"


def test_telegram_sends_the_paytm_simulator_link_and_announces_its_payment(tmp_path, monkeypatch):
    from marketplace_backend.sim_gateway import SimulatedCheckout, SimulatedPaytmGateway

    monkeypatch.setenv("FRONTEND_URL", "https://shop.test")
    store = build_merchant_store(tmp_path)
    shop, merchant = build_shopping(store, gateway=SimulatedPaytmGateway()), build_merchant(store)
    channel = tg.TelegramChannel(
        store, shopping_runtime=FakeRuntime(), merchant_runtime=FakeRuntime(),
        shopping_service=shop.service, merchant_service=merchant.service,
        shopping_sessions=({}, {}), merchant_sessions=({}, {}),
        session_types={"shopping": (SessionContext, SessionState),
                       "merchant": (MerchantSessionContext, MerchantSessionState)},
        link_base_url="https://shop.test")
    world = type("World", (), dict(store=store, shop=shop, channel=channel))

    run(shop.service.add(CUSTOMER, LAPTOP, 1))
    link(world, "shopping", CUSTOMER_P)
    reply = run(channel.handle("shopping", tap(checkout_button(world)["callback_data"])))
    url = buttons(reply)[0]["url"]
    assert url.startswith("https://shop.test/pay?link=simlink_")

    SimulatedCheckout(store, shop.webhooks).complete(url.split("link=")[1], method="upi", succeed=True)
    assert any("Payment verified" in d["body"]["text"] for d in channel.relay())


# ------------------------------------------------------------------ approvals


def staged_change(world) -> str:
    change = world.merchant.changes.stage(
        operator_id=OPERATOR, kind="price_update", target_type="variant",
        target_id=GOOD_CHARGER, before={"amount_minor": 249_900},
        after={"amount_minor": 239_900}, rationale="Match competitor")
    return change["id"]


def test_pending_changes_reach_linked_operators_and_need_a_second_approver(world):
    change_id = staged_change(world)
    link(world, "merchant", OPERATOR_P)
    link(world, "merchant", SECOND_P, user="9002", chat="5002")

    pushes = [d for d in world.channel.relay() if d["body"]["text"].startswith("Proposed")]
    assert {d["body"]["chat_id"] for d in pushes} == {CHAT, "5002"}

    own = next(b for b in buttons([p for p in pushes if p["body"]["chat_id"] == CHAT])
               if b["text"].startswith("✅"))
    refused = run(world.channel.handle("merchant", tap(own["callback_data"])))
    assert "different operator" in refused[0]["body"]["text"]
    assert world.store.rows("SELECT status FROM merchant_changes WHERE id=?", (change_id,))[0][
        "status"] == "pending"

    second = next(b for b in buttons([p for p in pushes if p["body"]["chat_id"] == "5002"])
                  if b["text"].startswith("❌"))
    decided = run(world.channel.handle("merchant", tap(second["callback_data"], chat="5002",
                                                       user="9002")))
    assert "rejected" in decided[-1]["body"]["text"]
    stale = run(world.channel.handle("merchant", tap(second["callback_data"], chat="5002",
                                                     user="9002")))
    assert stale[0]["body"]["text"] == "Refused"


# ------------------------------------------------------------------ formatting


def test_agent_markdown_renders_as_escaped_telegram_html(world):
    world.shopping_rt.events = [AgentEvent.text_delta(
        "- **Kurta <Set>** (sd_prd_kurta_set_0_v0): *urgent*\n## Next\nUse `x`")]
    [reply] = run(world.channel.handle("shopping", message("stock?")))
    assert reply["body"]["parse_mode"] == "HTML"
    assert reply["body"]["text"] == (
        "• <b>Kurta &lt;Set&gt;</b> (sd_prd_kurta_set_0_v0): <i>urgent</i>\n"
        "<b>Next</b>\nUse <code>x</code>")


def test_long_replies_split_between_paragraphs():
    paragraphs = ["**" + "a" * 1500 + "**"] * 4
    chunks = tg._chunks("\n\n".join(paragraphs))
    assert len(chunks) == 2 and all(c.count("**") % 2 == 0 for c in chunks)
