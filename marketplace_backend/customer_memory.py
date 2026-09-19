"""Customer memory: subjects, typed facts, the behaviour log, the brief, and the
worker that feeds Cognee.

The split is the one the rest of Cartisan uses. Memory is *advisory*: it may shape
what the agent asks, which products it ranks first and how it phrases things, but a
price, a stock level, a compatibility verdict or a cart change still comes from the
commerce core. So nothing here is authority over commerce state, and nothing here is
reachable from a model except through the fenced memory block and `recall_memories`.

Postgres is the system of record: the panel lists these rows, deletion removes them,
and the brief is built from them. Cognee receives distilled text through the outbox
(`MemorySyncWorker`) and answers semantic recall; when it is off or down the brief
still works.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Callable
from uuid import uuid4

from commerce_common.types import MemoryCategory, MemoryFact

from .cognee_client import CogneeUnavailable, MemoryBackend
from .evidence import Actor, Correlation, EvidenceLedger, Outbox
from .store import Store
from .timeutil import as_datetime, now

logger = logging.getLogger(__name__)

SYNC_TOPIC = "cognee.sync"
NOTES_REFRESH_DELAY_SECONDS = 120
GROUNDED_NOTES_PROMPT = (
    "Answer only from the retrieved context. Every bullet must be stated in it; add no "
    "detail, guess or generalisation. If the context holds nothing relevant, reply "
    "exactly NONE.")
MERCHANT_SUBJECT = "merchant_ops"

# Host-set weights (Q2). The client names the event; it never names the weight.
EVENT_WEIGHTS: dict[str, float] = {
    "purchase": 3.0,
    "add_to_cart": 2.0,
    "suggestion_click": 1.0,
    "product_view": 0.5,
    "remove_from_cart": -1.5,
    "rejected_recommendation": -2.0,
}
VIEW_DWELL_MS = 20_000
VIEW_REPEAT_COUNT = 2
INTEREST_WINDOW_DAYS = 90
GUEST_IDLE_DAYS = 30

# The typed facts a customer's memory may hold (Q10). A key is `<type>` or
# `<type>:<qualifier>` — `device_owned:phone`, `brand_affinity:boat`.
CUSTOMER_FACT_TYPES = frozenset({
    "device_owned", "brand_affinity", "budget", "use_case", "gift_context",
    "rejected_item", "current_project",
})
MERCHANT_FACT_TYPES = frozenset({"lesson", "outcome"})
# A value that ends a fact without replacing it: "I sold my Pixel".
CLOSING_VALUES = frozenset({"none", "no longer", "not anymore", "sold", "removed"})


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:16]}"


def _secret() -> bytes:
    value = os.getenv("CARTISAN_MEMORY_SECRET") or os.getenv("CARTISAN_OPS_TOKEN") or ""
    if not value:
        # A dev default keeps local runs working; the hash is still one-way.
        value = "cartisan-dev-memory-secret"
    return value.encode()


def fact_type_of(key: str) -> str:
    return key.split(":", 1)[0]


# ---------------------------------------------------------------------------
# Subjects
# ---------------------------------------------------------------------------


class MemorySubjects:
    """Whose memory a row is, and which Cognee dataset holds it (Q6, Q9).

    Dataset names are a keyed hash of the subject id: Cognee never sees a Supabase
    user id, and the name cannot be walked back to one without the secret.
    """

    def __init__(self, store: Store, secret: bytes | None = None) -> None:
        self.store = store
        self._secret = secret or _secret()

    def _digest(self, value: str) -> str:
        return hmac.new(self._secret, value.encode(), hashlib.sha256).hexdigest()

    def dataset_name(self, subject_id: str, kind: str) -> str:
        if kind == "merchant":
            return "cartisan_merchant_ops"
        prefix = "guest" if kind == "guest" else "cust"
        return f"cartisan_{prefix}_{self._digest(subject_id)[:24]}"

    @staticmethod
    def guest_subject(anon_id: str) -> str:
        return f"guest:{anon_id}"

    @staticmethod
    def kind_of(subject_id: str) -> str:
        if subject_id == MERCHANT_SUBJECT:
            return "merchant"
        return "guest" if subject_id.startswith("guest:") else "customer"

    def ensure(self, subject_id: str) -> dict:
        kind = self.kind_of(subject_id)
        stamp = now()
        self.store.execute(
            "INSERT INTO memory_subjects (id,kind,dataset_name,last_active_at,created_at) "
            "VALUES (?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET last_active_at=excluded.last_active_at",
            (subject_id, kind, self.dataset_name(subject_id, kind), stamp, stamp))
        return self.get(subject_id) or {}

    def get(self, subject_id: str) -> dict | None:
        rows = self.store.rows("SELECT * FROM memory_subjects WHERE id=?", (subject_id,))
        return rows[0] if rows else None

    def generation(self, subject_id: str) -> int:
        row = self.get(subject_id)
        return int(row["purge_generation"]) if row else 0

    # -- the guest cookie ----------------------------------------------------

    def new_guest_cookie(self) -> str:
        return self.sign(secrets.token_urlsafe(18))

    def sign(self, anon_id: str) -> str:
        return f"{anon_id}.{self._digest('anon:' + anon_id)[:32]}"

    def verify_cookie(self, cookie: str | None) -> str | None:
        """The anon id a cookie carries, or None when it was not issued here."""
        if not cookie or "." not in cookie:
            return None
        anon_id, _, signature = cookie.rpartition(".")
        if not anon_id or len(anon_id) > 64:
            return None
        expected = self._digest("anon:" + anon_id)[:32]
        return anon_id if hmac.compare_digest(expected, signature) else None


# ---------------------------------------------------------------------------
# Scheduling a sync
# ---------------------------------------------------------------------------


def schedule_sync(store: Store, outbox: Outbox, subject_id: str) -> str | None:
    """Queue one Cognee sync for the subject unless one is already waiting, so a
    burst of clicks becomes one batch rather than one call each (Q15)."""
    waiting = store.rows(
        "SELECT id FROM outbox_messages WHERE topic=? AND status='pending' AND payload=?",
        (SYNC_TOPIC, json.dumps({"subject_id": subject_id})))
    if waiting:
        return None
    return outbox.enqueue(topic=SYNC_TOPIC, payload={"subject_id": subject_id})


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


class EventRefused(ValueError):
    """The event names something the catalogue does not have, or an unknown type."""


class BehaviorLog:
    """High-intent browsing events (Q2). Views count only once they show intent:
    a second look at the same variant, or a long enough dwell."""

    def __init__(self, store: Store, subjects: MemorySubjects, outbox: Outbox) -> None:
        self.store, self.subjects, self.outbox = store, subjects, outbox

    def record(self, subject_id: str, event_type: str, variant_id: str, *,
               dwell_ms: int | None = None, correlation_id: str | None = None) -> dict:
        if event_type not in EVENT_WEIGHTS:
            raise EventRefused(f"Unknown event type {event_type!r}")
        rows = self.store.rows(
            "SELECT v.product_id FROM catalog_variants v WHERE v.id=?", (variant_id,))
        if not rows:
            raise EventRefused("That product is not in the catalogue")
        weight = EVENT_WEIGHTS[event_type]
        if event_type == "product_view" and not self._view_shows_intent(
                subject_id, variant_id, dwell_ms):
            weight = 0.0
        self.subjects.ensure(subject_id)
        event_id = _id("bev")
        self.store.execute(
            "INSERT INTO behavior_events (id,subject_id,event_type,variant_id,product_id,weight,"
            "dwell_ms,correlation_id,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (event_id, subject_id, event_type, variant_id, rows[0]["product_id"], weight,
             dwell_ms, correlation_id, now()))
        counted = weight != 0
        if counted:
            schedule_sync(self.store, self.outbox, subject_id)
        return {"event_id": event_id, "counted": counted, "weight": weight}

    def _view_shows_intent(self, subject_id: str, variant_id: str, dwell_ms: int | None) -> bool:
        if dwell_ms is not None and dwell_ms >= VIEW_DWELL_MS:
            return True
        since = (datetime.now(UTC) - timedelta(days=INTEREST_WINDOW_DAYS)).isoformat()
        earlier = self.store.rows(
            "SELECT COUNT(*) AS n FROM behavior_events WHERE subject_id=? AND variant_id=? "
            "AND event_type='product_view' AND created_at>=?", (subject_id, variant_id, since))
        return int(earlier[0]["n"]) + 1 >= VIEW_REPEAT_COUNT


# ---------------------------------------------------------------------------
# Typed facts — the MemoryStore the agent runtime writes through
# ---------------------------------------------------------------------------


class TypedFactStore:
    """`commerce_common.memory.MemoryStore` over `memory_facts`.

    Only allowlisted fact types are kept; anything else the extraction pass proposes
    is dropped here, after the runtime's PII write filter has already run. A new
    value under a held key closes the old row instead of overwriting it (Q10).
    """

    def __init__(self, store: Store, subjects: MemorySubjects, outbox: Outbox, *,
                 allowed: frozenset[str] = CUSTOMER_FACT_TYPES) -> None:
        self.store, self.subjects, self.outbox, self.allowed = store, subjects, outbox, allowed

    # -- the MemoryStore contract ---------------------------------------------

    async def get_facts(self, subject_id: str) -> list[MemoryFact]:
        return [self._fact(row) for row in self.live_rows(subject_id)]

    async def upsert_facts(self, subject_id: str, facts: list[MemoryFact]) -> None:
        self.write(subject_id, [
            (fact.key, fact.value, fact.category.value, fact.source_session_id) for fact in facts])

    async def search_facts(self, subject_id: str, query: str) -> list[MemoryFact]:
        from commerce_common.memory import match_facts

        return match_facts(await self.get_facts(subject_id), query)

    async def delete_fact(self, subject_id: str, key: str) -> bool:
        cursor = self.store.execute(
            "DELETE FROM memory_facts WHERE subject_id=? AND fact_key=?", (subject_id, key))
        return bool(getattr(cursor, "rowcount", 0))

    async def clear(self, subject_id: str) -> None:
        forget_locally(self.store, subject_id)

    async def purge_generation(self, subject_id: str) -> int:
        return self.subjects.generation(subject_id)

    # -- host-side reads and writes -------------------------------------------

    def write(self, subject_id: str, facts: list[tuple[str, str, str, str | None]]) -> list[str]:
        """Apply facts; returns the keys that changed."""
        changed: list[str] = []
        self.subjects.ensure(subject_id)
        for key, value, category, source in facts:
            key, value = key.strip().lower(), value.strip()
            if fact_type_of(key) not in self.allowed or not value:
                continue
            closing = value.lower() in CLOSING_VALUES
            stamp = now()
            with self.store.transaction() as tx:
                live = tx.rows(
                    "SELECT id,value FROM memory_facts WHERE subject_id=? AND fact_key=? "
                    "AND valid_to IS NULL", (subject_id, key))
                if live and not closing and live[0]["value"].lower() == value.lower():
                    continue
                for row in live:
                    tx.execute("UPDATE memory_facts SET valid_to=?, updated_at=? WHERE id=?",
                               (stamp, stamp, row["id"]))
                if not closing:
                    tx.execute(
                        "INSERT INTO memory_facts (id,subject_id,fact_key,fact_type,value,category,"
                        "source_session,valid_from,updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                        (_id("mf"), subject_id, key, fact_type_of(key), value[:200],
                         category if category in {c.value for c in MemoryCategory} else "preference",
                         source, stamp, stamp))
                elif not live:
                    continue
            changed.append(key)
        if changed:
            schedule_sync(self.store, self.outbox, subject_id)
        return changed

    def live_rows(self, subject_id: str) -> list[dict]:
        return self.store.rows(
            "SELECT * FROM memory_facts WHERE subject_id=? AND valid_to IS NULL "
            "ORDER BY updated_at DESC", (subject_id,))

    def delete_by_id(self, subject_id: str, fact_id: str) -> bool:
        """The panel's delete: the fact and its closed history both go."""
        rows = self.store.rows("SELECT fact_key FROM memory_facts WHERE id=? AND subject_id=?",
                               (fact_id, subject_id))
        if not rows:
            return False
        self.store.execute("DELETE FROM memory_facts WHERE subject_id=? AND fact_key=?",
                           (subject_id, rows[0]["fact_key"]))
        schedule_sync(self.store, self.outbox, subject_id)
        return True

    @staticmethod
    def _fact(row: dict) -> MemoryFact:
        return MemoryFact(
            key=row["fact_key"][:64], value=row["value"][:200],
            category=MemoryCategory(row["category"]),
            updated_at=as_datetime(row["updated_at"]),
            source_session_id=(row.get("source_session") or None),
        )


# ---------------------------------------------------------------------------
# Forgetting and merging
# ---------------------------------------------------------------------------


def forget_locally(store: Store, subject_id: str) -> None:
    """Drop every local trace of the subject's memory and advance its purge
    generation, so an extraction pass already running cannot write it back."""
    with store.transaction() as tx:
        for table in ("memory_facts", "behavior_events", "memory_feedback"):
            tx.execute(f"DELETE FROM {table} WHERE subject_id=?", (subject_id,))
        tx.execute("DELETE FROM memory_briefs WHERE subject_id=?", (subject_id,))
        tx.execute("UPDATE memory_subjects SET purge_generation=purge_generation+1 WHERE id=?",
                   (subject_id,))


# ---------------------------------------------------------------------------
# The brief
# ---------------------------------------------------------------------------


class MemoryBriefs:
    """The precomputed, advisory profile (Q7). Built from local rows only, so it is
    cheap and needs no Cognee call; the sync worker adds Cognee's recall notes when
    it has them. The agent reads it as a claim about the past, never as catalogue
    truth."""

    def __init__(self, store: Store, facts: TypedFactStore,
                 price_of: Callable[[str], int] | None = None) -> None:
        self.store, self.facts, self.price_of = store, facts, price_of

    def build(self, subject_id: str, cognee_notes: list[str] | None = None) -> dict:
        since = (datetime.now(UTC) - timedelta(days=INTEREST_WINDOW_DAYS)).isoformat()
        events = self.store.rows(
            "SELECT e.event_type, e.variant_id, e.weight, e.created_at, p.id AS product_id, "
            "p.title, p.brand, c.name AS category FROM behavior_events e "
            "JOIN catalog_variants v ON v.id=e.variant_id "
            "JOIN catalog_products p ON p.id=v.product_id "
            "LEFT JOIN catalog_categories c ON c.id=p.category_id "
            "WHERE e.subject_id=? AND e.created_at>=? AND e.weight<>0 ORDER BY e.created_at DESC",
            (subject_id, since))
        by_variant: dict[str, dict] = {}
        brands: dict[str, float] = defaultdict(float)
        categories: dict[str, float] = defaultdict(float)
        purchased: set[str] = set()
        in_cart_then_left: set[str] = set()
        for event in events:
            weight = float(event["weight"])
            brands[event["brand"]] += weight
            if event["category"]:
                categories[event["category"]] += weight
            entry = by_variant.setdefault(event["variant_id"], {
                "variant_id": event["variant_id"], "product_id": event["product_id"],
                "title": event["title"], "brand": event["brand"],
                "category": event["category"], "score": 0.0, "last_seen": event["created_at"],
                "signals": []})
            entry["score"] += weight
            entry["signals"].append(event["event_type"])
            if event["event_type"] == "purchase":
                purchased.add(event["variant_id"])
            if event["event_type"] == "remove_from_cart":
                in_cart_then_left.add(event["variant_id"])
        explored = sorted((v for v in by_variant.values() if v["score"] > 0
                           and v["variant_id"] not in purchased),
                          key=lambda v: (-v["score"], str(v["last_seen"])))[:6]
        prices = [self._price(v["variant_id"]) for v in explored]
        prices = [p for p in prices if p]
        facts = [
            {"id": row["id"], "type": row["fact_type"], "key": row["fact_key"],
             "value": row["value"], "source": row.get("source_session"),
             "since": str(row["valid_from"])}
            for row in self.facts.live_rows(subject_id)
        ]
        rejected = {f["value"].lower() for f in facts if f["type"] == "rejected_item"}
        brief = {
            "subject_kind": MemorySubjects.kind_of(subject_id),
            "facts": facts,
            "top_categories": _top(categories),
            "liked_brands": _top(brands),
            "avoided_brands": sorted(b for b, w in brands.items() if w < 0),
            "explored": [{k: v[k] for k in ("variant_id", "product_id", "title", "brand",
                                           "category", "signals")} for v in explored],
            "purchased_variant_ids": sorted(purchased),
            "removed_from_cart_variant_ids": sorted(in_cart_then_left - purchased),
            "rejected": sorted(rejected),
            "price_band_minor": [min(prices), max(prices)] if prices else None,
            "cognee_notes": (cognee_notes or [])[:5],
            "built_at": now(),
        }
        return brief

    def refresh(self, subject_id: str, cognee_notes: list[str] | None = None) -> dict:
        if cognee_notes is None:
            cognee_notes = (self.cached(subject_id) or {}).get("cognee_notes")
        brief = self.build(subject_id, cognee_notes)
        self.store.execute(
            "INSERT INTO memory_briefs (subject_id,brief,refreshed_at) VALUES (?,?,?) "
            "ON CONFLICT(subject_id) DO UPDATE SET brief=excluded.brief, refreshed_at=excluded.refreshed_at",
            (subject_id, json.dumps(brief), now()))
        return brief

    def cached(self, subject_id: str) -> dict | None:
        rows = self.store.rows("SELECT brief FROM memory_briefs WHERE subject_id=?", (subject_id,))
        return json.loads(rows[0]["brief"]) if rows else None

    def get(self, subject_id: str) -> dict:
        return self.cached(subject_id) or self.refresh(subject_id)

    def _price(self, variant_id: str) -> int | None:
        if self.price_of is None:
            return None
        try:
            return int(self.price_of(variant_id))
        except Exception:
            return None


def _top(weights: dict[str, float], n: int = 3) -> list[str]:
    return [name for name, w in sorted(weights.items(), key=lambda kv: -kv[1]) if w > 0][:n]


def render_brief(brief: dict | None) -> str:
    """The advisory block the shopping agent reads after its cached prefix."""
    if not brief:
        return ""
    lines = []
    for fact in brief.get("facts", []):
        lines.append(f"- {fact['key']}: {fact['value']}")
    if brief.get("top_categories"):
        lines.append(f"- browsing interest: {', '.join(brief['top_categories'])}")
    if brief.get("liked_brands"):
        lines.append(f"- brands engaged with: {', '.join(brief['liked_brands'])}")
    if brief.get("avoided_brands"):
        lines.append(f"- brands moved away from: {', '.join(brief['avoided_brands'])}")
    if brief.get("explored"):
        lines.append("- recently explored: " + "; ".join(
            f"{v['title']} ({v['variant_id']})" for v in brief["explored"][:4]))
    band = brief.get("price_band_minor")
    if band:
        lines.append(f"- explored price band: ₹{band[0] // 100:,}–₹{band[1] // 100:,}")
    for note in brief.get("cognee_notes", [])[:3]:
        lines.append(f"- memory note: {note[:200]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Merging a guest into a customer (Q9)
# ---------------------------------------------------------------------------


def merge_guest(store: Store, subjects: MemorySubjects, facts: TypedFactStore, outbox: Outbox,
                anon_id: str, customer_id: str) -> dict:
    """One-way and idempotent: the guest's counted events move to the customer, the
    guest subject is marked merged and its local memory dropped. The customer's
    memory never flows back to the anonymous side."""
    guest = subjects.guest_subject(anon_id)
    row = subjects.get(guest)
    if row is None or row.get("merged_into"):
        return {"merged": False, "events": 0}
    subjects.ensure(customer_id)
    moved = store.rows(
        "SELECT COUNT(*) AS n FROM behavior_events WHERE subject_id=? AND weight<>0", (guest,))
    with store.transaction() as tx:
        tx.execute("UPDATE behavior_events SET subject_id=?, synced_at=NULL "
                   "WHERE subject_id=? AND weight<>0", (customer_id, guest))
        tx.execute("UPDATE memory_subjects SET merged_into=? WHERE id=?", (customer_id, guest))
    forget_locally(store, guest)
    outbox.enqueue(topic=SYNC_TOPIC, payload={"subject_id": guest, "forget": True})
    schedule_sync(store, outbox, customer_id)
    return {"merged": True, "events": int(moved[0]["n"])}


# ---------------------------------------------------------------------------
# The sync worker (Q15)
# ---------------------------------------------------------------------------


@dataclass
class SyncCaps:
    daily_total: int = int(os.getenv("COGNEE_DAILY_CALL_CAP", "2000"))
    daily_per_subject: int = int(os.getenv("COGNEE_SUBJECT_DAILY_CAP", "40"))


class MemorySyncWorker:
    """Moves distilled memory to Cognee. Each message is one subject; the batch is
    what changed since the last sync, keyed so a replay is a no-op; the daily caps
    leave work queued instead of spending credits; the brief is refreshed either
    way, so the storefront never waits on Cognee."""

    def __init__(self, store: Store, subjects: MemorySubjects, facts: TypedFactStore,
                 briefs: MemoryBriefs, outbox: Outbox, ledger: EvidenceLedger,
                 client: MemoryBackend, caps: SyncCaps | None = None) -> None:
        self.store, self.subjects, self.facts, self.briefs = store, subjects, facts, briefs
        self.outbox, self.ledger, self.client = outbox, ledger, client
        self.caps = caps or SyncCaps()

    async def drain(self, limit: int = 20) -> list[dict]:
        results = []
        for message in self.outbox.claim(limit=limit, topic=SYNC_TOPIC):
            try:
                result = await self._handle(message["payload"])
            except CogneeUnavailable as exc:
                self.outbox.failed(message["id"], str(exc)[:300], retry_in_seconds=300)
                results.append({"message_id": message["id"], "status": "retry", "error": str(exc)})
                continue
            except Exception as exc:  # a bug must park the message, not lose it
                logger.exception("memory sync failed")
                self.outbox.failed(message["id"], type(exc).__name__, retry_in_seconds=300)
                results.append({"message_id": message["id"], "status": "failed"})
                continue
            if result.get("status") == "deferred":
                self.outbox.failed(message["id"], "daily Cognee cap reached", retry_in_seconds=3600)
            else:
                self.outbox.delivered(message["id"])
            results.append({"message_id": message["id"], **result})
        return results

    async def _handle(self, payload: dict) -> dict:
        subject_id = str(payload["subject_id"])
        subject = self.subjects.get(subject_id)
        if subject is None:
            return {"status": "skipped", "reason": "unknown subject"}
        dataset = subject["dataset_name"]
        if payload.get("forget"):
            if self.client.enabled:
                await self.client.forget(dataset)
            self._record(subject_id, "memory.cognee_forget", "Subject memory deleted in Cognee")
            return {"status": "forgotten"}

        texts, fact_ids, event_ids = self._batch(subject_id)
        if not texts:
            # Nothing to send, but the graph from the last sync may be ready by now.
            notes = None
            if self.client.enabled and self._within_caps(subject_id):
                notes = await self.grounded_notes(dataset, subject["kind"])
                self._count(subject_id, calls=2)
            self.briefs.refresh(subject_id, cognee_notes=notes)
            return {"status": "nothing_new"}
        key = f"{dataset}:{hashlib.sha256(json.dumps(texts).encode()).hexdigest()[:32]}"
        if self.store.rows("SELECT 1 AS seen FROM memory_sync_batches WHERE idempotency_key=?", (key,)):
            self._mark_synced(fact_ids, event_ids)
            return {"status": "duplicate"}
        if not self.client.enabled:
            # Memory still works locally; nothing is sent, nothing is lost.
            self.briefs.refresh(subject_id)
            return {"status": "local_only", "items": len(texts)}
        if not self._within_caps(subject_id):
            self.briefs.refresh(subject_id)
            return {"status": "deferred"}

        await self.client.ensure_dataset(dataset)
        # Notes come from what earlier syncs already put in the graph: `remember` builds
        # it in the background, so asking right after would query an empty dataset —
        # and an ungrounded completion invents a shopper. See `grounded_notes`.
        notes = await self.grounded_notes(dataset, subject["kind"])
        await self.client.remember(dataset, texts, node_set=[subject["kind"]])
        self._count(subject_id, calls=3)
        self.store.execute(
            "INSERT INTO memory_sync_batches (idempotency_key,subject_id,item_count,cognee_status,created_at) "
            "VALUES (?,?,?,?,?)", (key, subject_id, len(texts), "remembered", now()))
        self._mark_synced(fact_ids, event_ids)
        self.briefs.refresh(subject_id, cognee_notes=notes)
        self._schedule_notes_refresh(subject_id)
        self._record(subject_id, "memory.cognee_sync",
                     f"Sent {len(texts)} distilled memory item(s) to Cognee",
                     state_ref={"batch": key, "facts": len(fact_ids), "events": len(event_ids)})
        return {"status": "synced", "items": len(texts)}

    def _schedule_notes_refresh(self, subject_id: str) -> None:
        """Come back once Cognee has built the graph for what was just sent, so the
        brief's notes reflect it. One pending follow-up per subject at most."""
        payload = {"subject_id": subject_id, "refresh_notes": True}
        if self.store.rows("SELECT id FROM outbox_messages WHERE topic=? AND status='pending' AND payload=?",
                           (SYNC_TOPIC, json.dumps(payload))):
            return
        message_id = self.outbox.enqueue(topic=SYNC_TOPIC, payload=payload)
        later = (datetime.now(UTC) + timedelta(seconds=NOTES_REFRESH_DELAY_SECONDS)).isoformat()
        self.store.execute("UPDATE outbox_messages SET available_at=? WHERE id=?", (later, message_id))

    async def grounded_notes(self, dataset: str, kind: str) -> list[str] | None:
        """A short summary only when Cognee actually retrieved something for this
        dataset, and only from that. None means "keep the notes we had"."""
        question = (
            "What does this store's operator consistently approve and reject, and what did "
            "recent promotions and recovery offers achieve?" if kind == "merchant" else
            "What are this shopper's lasting preferences, devices, budget, and what are they "
            "currently shopping for?")
        try:
            context = await self.client.recall(dataset, question, top_k=5)
            if not any(text.strip() for text in context):
                return None
            answer = await self.client.recall(
                dataset, question + " Answer in at most five short bullet points.", top_k=5,
                answer=True, system_prompt=GROUNDED_NOTES_PROMPT)
        except CogneeUnavailable:
            return None
        notes = [note for note in _bullets(answer) if note.upper().strip(". ") != "NONE"]
        return notes

    def _batch(self, subject_id: str) -> tuple[list[str], list[str], list[str]]:
        facts = self.store.rows(
            "SELECT id,fact_key,value,valid_to FROM memory_facts WHERE subject_id=? AND synced_at IS NULL",
            (subject_id,))
        events = self.store.rows(
            "SELECT e.id,e.event_type,e.weight,p.title,p.brand,c.name AS category "
            "FROM behavior_events e JOIN catalog_variants v ON v.id=e.variant_id "
            "JOIN catalog_products p ON p.id=v.product_id "
            "LEFT JOIN catalog_categories c ON c.id=p.category_id "
            "WHERE e.subject_id=? AND e.synced_at IS NULL AND e.weight<>0 ORDER BY e.created_at",
            (subject_id,))
        feedback = self.store.rows(
            "SELECT id,rating,reason FROM memory_feedback WHERE subject_id=? AND synced_at IS NULL",
            (subject_id,))
        texts = []
        for row in feedback:
            reason = f" ({row['reason'].replace('_', ' ')})" if row["reason"] else ""
            texts.append(f"The shopper rated an assistant answer {'helpful' if row['rating'] == 'up' else 'unhelpful'}{reason}.")
        who = "The store's record" if subject_id == MERCHANT_SUBJECT else "The shopper"
        for fact in facts:
            if subject_id == MERCHANT_SUBJECT:
                prefix = "(superseded) " if fact["valid_to"] else ""
                texts.append(f"{who} — {prefix}{fact['fact_key'].replace('_', ' ')}: {fact['value']}")
                continue
            verb = "no longer holds" if fact["valid_to"] else "states"
            texts.append(f"{who} {verb} {fact['fact_key'].replace('_', ' ')}: {fact['value']}.")
        grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
        for event in events:
            grouped[(event["event_type"], event["category"] or "products")].append(
                f"{event['title']} by {event['brand']}")
        for (event_type, category), titles in grouped.items():
            phrase = {
                "purchase": "bought", "add_to_cart": "added to cart",
                "suggestion_click": "opened a suggestion for",
                "product_view": "looked closely at",
                "remove_from_cart": "removed from cart",
                "rejected_recommendation": "turned down",
            }[event_type]
            texts.append(f"In {category}, the shopper {phrase}: {', '.join(sorted(set(titles)))}.")
        self._pending_feedback = [row["id"] for row in feedback]
        return texts, [f["id"] for f in facts], [e["id"] for e in events]

    def _mark_synced(self, fact_ids: list[str], event_ids: list[str]) -> None:
        stamp = now()
        feedback_ids, self._pending_feedback = getattr(self, "_pending_feedback", []), []
        for table, ids in (("memory_facts", fact_ids), ("behavior_events", event_ids),
                           ("memory_feedback", feedback_ids)):
            for row_id in ids:
                self.store.execute(f"UPDATE {table} SET synced_at=? WHERE id=?", (stamp, row_id))

    def _within_caps(self, subject_id: str) -> bool:
        day = datetime.now(UTC).date().isoformat()
        rows = {row["subject_id"]: int(row["calls"]) for row in self.store.rows(
            "SELECT subject_id,calls FROM memory_usage WHERE day=? AND subject_id IN (?,?)",
            (day, subject_id, "*"))}
        return (rows.get("*", 0) < self.caps.daily_total
                and rows.get(subject_id, 0) < self.caps.daily_per_subject)

    def _count(self, subject_id: str, calls: int) -> None:
        day = datetime.now(UTC).date().isoformat()
        for who in (subject_id, "*"):
            self.store.execute(
                "INSERT INTO memory_usage (day,subject_id,calls) VALUES (?,?,?) "
                "ON CONFLICT(day,subject_id) DO UPDATE SET calls=memory_usage.calls+excluded.calls",
                (day, who, calls))

    def _record(self, subject_id: str, action: str, reason: str, state_ref: Any = None) -> None:
        self.ledger.record(
            actor=Actor(type="system", id="memory-sync", surface="memory"), action=action,
            reason=reason, outcome="applied", target_type="memory_subject",
            target_id=self.subjects.dataset_name(subject_id, MemorySubjects.kind_of(subject_id)),
            state_ref=state_ref, correlation=Correlation())


def _bullets(answers: list[str]) -> list[str]:
    """Cognee's answer as short notes: one per bullet or line, markdown stripped."""
    notes = []
    for answer in answers:
        for line in answer.splitlines():
            line = line.strip().lstrip("-*•0123456789. ").strip()
            if line and not line.startswith("#"):
                notes.append(line[:200])
    return notes[:5]


def expire_guests(store: Store, outbox: Outbox, idle_days: int = GUEST_IDLE_DAYS) -> int:
    """Delete guest memory idle for longer than the window, here and in Cognee."""
    cutoff = (datetime.now(UTC) - timedelta(days=idle_days)).isoformat()
    stale = store.rows(
        "SELECT id FROM memory_subjects WHERE kind='guest' AND merged_into IS NULL "
        "AND last_active_at<?", (cutoff,))
    for row in stale:
        forget_locally(store, row["id"])
        store.execute("UPDATE memory_subjects SET merged_into='expired' WHERE id=?", (row["id"],))
        outbox.enqueue(topic=SYNC_TOPIC, payload={"subject_id": row["id"], "forget": True})
    return len(stale)
