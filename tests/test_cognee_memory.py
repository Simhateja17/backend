"""Cognee memory: advisory memory about people, never authority over commerce.

What these pin down: the host decides what counts as a signal and what it weighs;
only allowlisted fact types are kept and a superseded fact is closed, not lost;
suggestions come from live catalogue rows the brief merely steers; the sync worker
is idempotent, capped and survives Cognee being off; a guest merges one way; and a
payment worker never claims a memory message.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cartisan_agent.executor import build_memory
from cartisan_agent.config import CartisanAgentConfig
from cartisan_agent.prompts import build_dynamic_context
from commerce_common.types import MemoryCategory, MemoryFact
from marketplace_backend.cognee_client import CogneeUnavailable
from marketplace_backend.customer_memory import (
    SYNC_TOPIC,
    BehaviorLog,
    EventRefused,
    MemoryBriefs,
    MemorySubjects,
    MemorySyncWorker,
    SyncCaps,
    TypedFactStore,
    forget_locally,
    merge_guest,
    render_brief,
)
from marketplace_backend.evidence import EvidenceLedger, Outbox
from marketplace_backend.personalization import Personalizer

from tests.conftest_runtime import CUSTOMER, GOOD_CHARGER, LAPTOP, WEAK_CHARGER, build_services, build_store

BANK = "sd_var_bank10k"
SOLD_OUT = "sd_var_bank20k"
FOREIGN = "sd_var_zapcharger"


def run(coro):
    return asyncio.run(coro)


class FakeCognee:
    def __init__(self, enabled: bool = True, fail: bool = False) -> None:
        self.enabled, self.fail = enabled, fail
        self.remembered: list[tuple[str, list[str]]] = []
        self.forgotten: list[str] = []

    async def ensure_dataset(self, name: str) -> str:
        return "ds-" + name

    async def remember(self, dataset_name, texts, node_set):
        if self.fail:
            raise CogneeUnavailable("down")
        self.remembered.append((dataset_name, list(texts)))
        return {"status": "ok"}

    async def recall(self, dataset_name, query, top_k=8, answer=False, system_prompt=None):
        return ["Owns an Aster 14 laptop; shopping for chargers"]

    async def improve(self, dataset_name):
        return {}

    async def forget(self, dataset_name):
        self.forgotten.append(dataset_name)


@pytest.fixture
def mem(tmp_path):
    store = build_store(tmp_path)
    store.execute("INSERT INTO catalog_products (id,sku_root,title,brand,category_id,description,"
                  "status,origin) VALUES ('sd_prod_bank','BNK-1','Voltix power bank','Voltix',"
                  "'cat-power','A power bank.','active','seeded')")
    store.execute("INSERT INTO catalog_products (id,sku_root,title,brand,category_id,description,"
                  "status,origin) VALUES ('sd_prod_zap','ZAP-1','Zap wall charger','Zap',"
                  "'cat-power','A charger.','active','seeded')")
    for variant_id, product, sku, price, on_hand in (
            (BANK, "sd_prod_bank", "BNK-1-10", 1_999_00, 4),
            (SOLD_OUT, "sd_prod_bank", "BNK-1-20", 2_999_00, 0),
            (FOREIGN, "sd_prod_zap", "ZAP-1-20", 999_00, 3)):
        store.execute("INSERT INTO catalog_variants (id,product_id,sku,title,options,status) "
                      "VALUES (?,?,?,?,'{}','active')", (variant_id, product, sku, sku))
        store.execute("INSERT INTO variant_prices (id,variant_id,currency,amount_minor,price_kind,"
                      "valid_from) VALUES (?,?,'INR',?,'list','2020-01-01T00:00:00+00:00')",
                      (f"price_{variant_id}", variant_id, price))
        store.execute("INSERT INTO inventory_levels (variant_id,location_id,on_hand,reserved) "
                      "VALUES (?,'loc-blr',?,0)", (variant_id, on_hand))
    port = build_services(store).port
    outbox = Outbox(store)
    subjects = MemorySubjects(store, secret=b"test-secret")
    facts = TypedFactStore(store, subjects, outbox)
    briefs = MemoryBriefs(store, facts, price_of=port.current_price)
    return {
        "store": store, "port": port, "outbox": outbox, "subjects": subjects, "facts": facts,
        "briefs": briefs, "behavior": BehaviorLog(store, subjects, outbox),
        "personalizer": Personalizer(store, port), "ledger": EvidenceLedger(store),
    }


def worker(mem, client, caps=None):
    return MemorySyncWorker(mem["store"], mem["subjects"], mem["facts"], mem["briefs"],
                            mem["outbox"], mem["ledger"], client, caps)


# -- signals ---------------------------------------------------------------------


def test_a_glance_is_not_a_signal_but_a_second_look_or_a_long_one_is(mem):
    log = mem["behavior"]
    assert log.record(CUSTOMER, "product_view", GOOD_CHARGER, dwell_ms=3_000)["counted"] is False
    assert log.record(CUSTOMER, "product_view", GOOD_CHARGER, dwell_ms=3_000)["counted"] is True
    assert log.record(CUSTOMER, "product_view", LAPTOP, dwell_ms=25_000)["counted"] is True


def test_the_host_sets_the_weight_and_refuses_what_the_catalogue_lacks(mem):
    assert mem["behavior"].record(CUSTOMER, "add_to_cart", GOOD_CHARGER)["weight"] == 2.0
    with pytest.raises(EventRefused):
        mem["behavior"].record(CUSTOMER, "add_to_cart", "sd_var_invented")
    with pytest.raises(EventRefused):
        mem["behavior"].record(CUSTOMER, "free_money", GOOD_CHARGER)


def test_a_burst_of_signals_queues_one_sync(mem):
    for _ in range(3):
        mem["behavior"].record(CUSTOMER, "add_to_cart", GOOD_CHARGER)
    pending = mem["store"].rows("SELECT id FROM outbox_messages WHERE topic=? AND status='pending'",
                                (SYNC_TOPIC,))
    assert len(pending) == 1


# -- typed facts -----------------------------------------------------------------


def fact(key, value):
    return MemoryFact(key=key, value=value, category=MemoryCategory.CONTEXT)


def test_only_allowlisted_fact_types_are_kept(mem):
    run(mem["facts"].upsert_facts(CUSTOMER, [
        fact("device_owned:laptop", "Aster 14 laptop"),
        fact("home_address", "somewhere"),
        fact("favourite_colour", "blue"),
    ]))
    assert [f.key for f in run(mem["facts"].get_facts(CUSTOMER))] == ["device_owned:laptop"]


def test_a_new_value_closes_the_old_fact_and_none_ends_it(mem):
    facts = mem["facts"]
    run(facts.upsert_facts(CUSTOMER, [fact("device_owned:phone", "Pixel 8")]))
    run(facts.upsert_facts(CUSTOMER, [fact("device_owned:phone", "iPhone 16")]))
    assert [f.value for f in run(facts.get_facts(CUSTOMER))] == ["iPhone 16"]
    history = mem["store"].rows("SELECT value,valid_to FROM memory_facts WHERE fact_key='device_owned:phone'")
    assert {row["value"]: row["valid_to"] is None for row in history} == {"Pixel 8": False, "iPhone 16": True}

    run(facts.upsert_facts(CUSTOMER, [fact("device_owned:phone", "none")]))
    assert run(facts.get_facts(CUSTOMER)) == []


def test_deleting_from_the_panel_removes_the_fact_and_its_history(mem):
    facts = mem["facts"]
    run(facts.upsert_facts(CUSTOMER, [fact("device_owned:phone", "Pixel 8")]))
    run(facts.upsert_facts(CUSTOMER, [fact("device_owned:phone", "iPhone 16")]))
    live = facts.live_rows(CUSTOMER)[0]
    assert facts.delete_by_id(CUSTOMER, live["id"]) is True
    assert mem["store"].rows("SELECT id FROM memory_facts WHERE subject_id=?", (CUSTOMER,)) == []
    assert facts.delete_by_id("someone-else", live["id"]) is False


def test_forgetting_clears_everything_and_advances_the_purge_generation(mem):
    run(mem["facts"].upsert_facts(CUSTOMER, [fact("budget:chargers", "under 3000 rupees")]))
    mem["behavior"].record(CUSTOMER, "add_to_cart", GOOD_CHARGER)
    before = run(mem["facts"].purge_generation(CUSTOMER))
    forget_locally(mem["store"], CUSTOMER)
    assert run(mem["facts"].get_facts(CUSTOMER)) == []
    assert mem["store"].rows("SELECT id FROM behavior_events WHERE subject_id=?", (CUSTOMER,)) == []
    assert run(mem["facts"].purge_generation(CUSTOMER)) == before + 1


def test_the_agent_runtime_reads_typed_facts_through_the_memory_contract(mem):
    runtime = build_memory(CartisanAgentConfig(), mem["facts"])
    run(mem["facts"].upsert_facts(CUSTOMER, [fact("device_owned:laptop", "Aster 14 laptop")]))
    assert runtime.enabled
    assert [f.value for f in run(runtime.tier_one(CUSTOMER))] == ["Aster 14 laptop"]


# -- the brief and suggestions ---------------------------------------------------


def test_the_brief_is_built_from_counted_signals_and_leaves_out_what_was_bought(mem):
    log = mem["behavior"]
    log.record(CUSTOMER, "add_to_cart", GOOD_CHARGER)
    log.record(CUSTOMER, "purchase", LAPTOP)
    log.record(CUSTOMER, "product_view", WEAK_CHARGER, dwell_ms=1)  # a glance: not counted
    brief = mem["briefs"].refresh(CUSTOMER)
    assert [v["variant_id"] for v in brief["explored"]] == [GOOD_CHARGER]
    assert brief["purchased_variant_ids"] == [LAPTOP]
    assert brief["price_band_minor"] == [2_499_00, 2_499_00]
    assert "Nimbus" in brief["liked_brands"]
    assert "Nimbus travel charger" in render_brief(brief)


def test_suggestions_are_live_in_stock_catalogue_rows_with_a_reason(mem):
    mem["behavior"].record(CUSTOMER, "add_to_cart", GOOD_CHARGER)
    result = mem["personalizer"].suggestions(mem["briefs"].refresh(CUSTOMER))
    ids = [item["variant_id"] for item in result["items"]]
    assert BANK in ids and FOREIGN in ids
    assert SOLD_OUT not in ids                      # out of stock
    assert GOOD_CHARGER not in ids and WEAK_CHARGER not in ids  # the explored product itself
    bank = next(item for item in result["items"] if item["variant_id"] == BANK)
    assert bank["reason"] == "Because you explored Nimbus travel charger"
    assert bank["price_minor"] == 1_999_00


def test_a_brand_moved_away_from_is_not_suggested(mem):
    mem["behavior"].record(CUSTOMER, "add_to_cart", GOOD_CHARGER)
    mem["behavior"].record(CUSTOMER, "rejected_recommendation", FOREIGN)
    ids = [i["variant_id"] for i in mem["personalizer"].suggestions(mem["briefs"].refresh(CUSTOMER))["items"]]
    assert FOREIGN not in ids and BANK in ids


def test_no_history_means_no_suggestions_rather_than_guesses(mem):
    assert mem["personalizer"].suggestions(mem["briefs"].refresh(CUSTOMER)) == {
        "items": [], "basis": "no_history"}
    assert mem["personalizer"].welcome(mem["briefs"].refresh(CUSTOMER)) is None


def test_the_agent_reads_the_brief_as_fenced_advisory_context_without_ids(mem):
    mem["behavior"].record(CUSTOMER, "add_to_cart", GOOD_CHARGER)
    context = build_dynamic_context(preferences=None, memory_facts=[], cart=None, page=None,
                                    memory_brief=mem["briefs"].refresh(CUSTOMER))
    assert "memory_brief" in context and "Nimbus travel charger" in context
    assert GOOD_CHARGER not in context


# -- the sync worker -------------------------------------------------------------


def test_sync_sends_distilled_text_once_and_refreshes_the_brief(mem):
    client = FakeCognee()
    mem["behavior"].record(CUSTOMER, "add_to_cart", GOOD_CHARGER)
    run(mem["facts"].upsert_facts(CUSTOMER, [fact("device_owned:laptop", "Aster 14 laptop")]))
    results = run(worker(mem, client).drain())
    assert [r["status"] for r in results] == ["synced"]
    dataset, texts = client.remembered[0]
    assert CUSTOMER not in dataset and dataset.startswith("cartisan_cust_")
    assert any("added to cart: Nimbus travel charger" in t for t in texts)
    assert mem["briefs"].cached(CUSTOMER)["cognee_notes"]
    # A notes follow-up is queued for once Cognee has built the graph, not due yet.
    follow_up = mem["store"].rows("SELECT available_at FROM outbox_messages WHERE topic=? "
                                  "AND status='pending'", (SYNC_TOPIC,))
    assert len(follow_up) == 1 and run(worker(mem, client).drain()) == []
    # Nothing new: a second drain sends nothing.
    mem["outbox"].enqueue(topic=SYNC_TOPIC, payload={"subject_id": CUSTOMER})
    assert [r["status"] for r in run(worker(mem, client).drain())] == ["nothing_new"]
    assert len(client.remembered) == 1


def test_notes_come_only_from_retrieved_context(mem):
    class Empty(FakeCognee):
        async def recall(self, dataset_name, query, top_k=8, answer=False, system_prompt=None):
            return [] if not answer else ["- A shopper who loves robot vacuums"]
    mem["behavior"].record(CUSTOMER, "add_to_cart", GOOD_CHARGER)
    run(worker(mem, Empty()).drain())
    assert mem["briefs"].cached(CUSTOMER)["cognee_notes"] == []


def test_a_replayed_batch_is_a_no_op(mem):
    client = FakeCognee()
    mem["behavior"].record(CUSTOMER, "add_to_cart", GOOD_CHARGER)
    run(worker(mem, client).drain())
    mem["store"].execute("UPDATE behavior_events SET synced_at=NULL")
    mem["outbox"].enqueue(topic=SYNC_TOPIC, payload={"subject_id": CUSTOMER})
    assert [r["status"] for r in run(worker(mem, client).drain())] == ["duplicate"]
    assert len(client.remembered) == 1


def test_memory_works_locally_with_cognee_off_and_retries_when_it_is_down(mem):
    mem["behavior"].record(CUSTOMER, "add_to_cart", GOOD_CHARGER)
    assert [r["status"] for r in run(worker(mem, FakeCognee(enabled=False)).drain())] == ["local_only"]
    assert mem["briefs"].cached(CUSTOMER)["explored"]

    mem["behavior"].record(CUSTOMER, "add_to_cart", BANK)
    assert [r["status"] for r in run(worker(mem, FakeCognee(fail=True)).drain())] == ["retry"]
    row = mem["store"].rows("SELECT status FROM outbox_messages WHERE topic=? ORDER BY created_at DESC",
                            (SYNC_TOPIC,))[0]
    assert row["status"] == "pending"


def test_the_daily_cap_defers_instead_of_spending(mem):
    client = FakeCognee()
    mem["behavior"].record(CUSTOMER, "add_to_cart", GOOD_CHARGER)
    results = run(worker(mem, client, SyncCaps(daily_total=0, daily_per_subject=0)).drain())
    assert [r["status"] for r in results] == ["deferred"] and client.remembered == []


def test_a_payment_worker_never_claims_a_memory_message(mem):
    mem["behavior"].record(CUSTOMER, "add_to_cart", GOOD_CHARGER)
    assert mem["outbox"].claim(topic="razorpay.payment_link.create") == []
    assert len(mem["outbox"].claim(topic=SYNC_TOPIC)) == 1


# -- guests ----------------------------------------------------------------------


def test_the_guest_cookie_is_signed(mem):
    subjects = mem["subjects"]
    cookie = subjects.new_guest_cookie()
    anon_id = subjects.verify_cookie(cookie)
    assert anon_id and subjects.verify_cookie(cookie[:-1] + ("0" if cookie[-1] != "0" else "1")) is None
    assert subjects.verify_cookie("forged.value") is None


def test_a_guest_merges_into_the_account_once_and_one_way(mem):
    client = FakeCognee()
    guest = mem["subjects"].guest_subject("anon123")
    mem["behavior"].record(guest, "add_to_cart", GOOD_CHARGER)
    mem["behavior"].record(guest, "product_view", BANK, dwell_ms=1)  # not counted, not moved
    first = merge_guest(mem["store"], mem["subjects"], mem["facts"], mem["outbox"], "anon123", CUSTOMER)
    assert first == {"merged": True, "events": 1}
    assert merge_guest(mem["store"], mem["subjects"], mem["facts"], mem["outbox"], "anon123",
                       CUSTOMER)["merged"] is False
    assert [v["variant_id"] for v in mem["briefs"].refresh(CUSTOMER)["explored"]] == [GOOD_CHARGER]
    assert mem["briefs"].refresh(guest)["explored"] == []
    run(worker(mem, client).drain())
    assert client.forgotten == [mem["subjects"].dataset_name(guest, "guest")]


# -- the API, end to end on SQLite -----------------------------------------------


def test_guest_memory_api_round_trip(tmp_path):
    """A subprocess, because `api.main` binds its Store at import time."""
    script = f"""
import os, json
os.environ["CARTISAN_DB_PATH"] = {str(tmp_path / 'api.db')!r}
os.environ["SUPABASE_DATABASE_URL"] = ""
os.environ["COGNEE_ENABLED"] = "0"
from tests.conftest_runtime import build_store
from pathlib import Path
import shutil
seeded = build_store(Path({str(tmp_path)!r}) / "seed")
seeded.close()
shutil.copy(Path({str(tmp_path)!r}) / "seed" / "runtime.db", os.environ["CARTISAN_DB_PATH"])
from fastapi.testclient import TestClient
import api.main as main
client = TestClient(main.app)
r = client.post("/memory/events", json={{"event_type": "add_to_cart", "variant_id": "{GOOD_CHARGER}"}})
assert r.status_code == 200 and r.json()["counted"], r.text
assert "cartisan_anon_id" in r.cookies or client.cookies.get("cartisan_anon_id")
assert client.post("/memory/events", json={{"event_type": "purchase", "variant_id": "{GOOD_CHARGER}"}}).status_code == 422
panel = client.get("/memory/me").json()
assert panel["subject_kind"] == "guest" and panel["explored"][0]["variant_id"] == "{GOOD_CHARGER}"
sugg = client.get("/memory/suggestions").json()
assert sugg["basis"] == "memory"
assert client.get("/memory/welcome").json()["card"]["still_interested"]
assert client.delete("/memory/me").json() == {{"forgotten": True}}
assert client.get("/memory/me").json()["explored"] == []
print("ok")
"""
    (tmp_path / "seed").mkdir()
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1], timeout=120)
    assert "ok" in result.stdout, result.stderr[-3000:]
