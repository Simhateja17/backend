"""Merchant memory: lessons from the operator's own decisions, and outcomes (Q11).

A lesson is arithmetic over the approval record, not a model's reading of it:
"rejected 4 of 4 promotion proposals — margin too low; approved up to 12%, rejected
from 15%". The merchant agent reads lessons as saved memory and uses them to shape
what it proposes; nothing here approves, rejects or changes anything. Lessons and
outcomes are store-wide, so every operator's memory reads the one `merchant_ops`
subject.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime, timedelta

from commerce_common.types import MemoryFact

from .customer_memory import MERCHANT_FACT_TYPES, MERCHANT_SUBJECT, MemorySubjects, TypedFactStore
from .evidence import Outbox
from .store import Store

REASON_CODES = ("margin_too_low", "bad_timing", "brand_policy", "stock_risk", "other")
LESSON_WINDOW_DAYS = 90


class MerchantLessonStore(TypedFactStore):
    """The `MemoryStore` the merchant runtime reads. Whoever the operator is, the
    subject is the store's shared `merchant_ops` memory; only lessons and outcomes are
    kept, and only the host writes them."""

    def __init__(self, store: Store, subjects: MemorySubjects, outbox: Outbox) -> None:
        super().__init__(store, subjects, outbox, allowed=MERCHANT_FACT_TYPES)

    async def get_facts(self, subject_id: str) -> list[MemoryFact]:
        return await super().get_facts(MERCHANT_SUBJECT)

    async def upsert_facts(self, subject_id: str, facts: list[MemoryFact]) -> None:
        # The extraction pass has no business writing lessons; they are computed.
        return None

    async def search_facts(self, subject_id: str, query: str) -> list[MemoryFact]:
        return await super().search_facts(MERCHANT_SUBJECT, query)

    async def delete_fact(self, subject_id: str, key: str) -> bool:
        return await super().delete_fact(MERCHANT_SUBJECT, key)

    async def clear(self, subject_id: str) -> None:
        await super().clear(MERCHANT_SUBJECT)

    async def purge_generation(self, subject_id: str) -> int:
        return await super().purge_generation(MERCHANT_SUBJECT)


def record_reason(store: Store, change_id: str, reason_code: str | None) -> None:
    """Attach the operator's reason code to their latest decision on a change."""
    if reason_code is None:
        return
    if reason_code not in REASON_CODES:
        raise ValueError(f"reason_code must be one of {', '.join(REASON_CODES)}")
    rows = store.rows("SELECT id FROM merchant_approvals WHERE change_id=? ORDER BY decided_at DESC LIMIT 1",
                      (change_id,))
    if rows:
        store.execute("UPDATE merchant_approvals SET reason_code=? WHERE id=?", (reason_code, rows[0]["id"]))


def _magnitude(kind: str, before: dict, after: dict) -> float | None:
    """The one number an operator decides a change on: the discount, or the move."""
    if kind in {"promotion"} and after.get("discount_kind") == "percentage":
        return float(after.get("discount_value") or 0)
    if kind == "recovery_policy":
        return float(after.get("discount_percentage") or 0)
    if kind == "price_update" and before.get("amount_minor"):
        return round(100 * abs(after["amount_minor"] - before["amount_minor"]) / before["amount_minor"], 1)
    return None


def refresh_lessons(store: Store, lessons: TypedFactStore, outcomes: dict | None = None) -> list[str]:
    """Recompute lessons from the last 90 days of decisions, and outcome facts from
    the numbers passed in. Returns the keys that changed."""
    since = (datetime.now(UTC) - timedelta(days=LESSON_WINDOW_DAYS)).isoformat()
    rows = store.rows(
        "SELECT c.kind, c.before_doc, c.after_doc, a.decision, a.reason_code FROM merchant_changes c "
        "JOIN merchant_approvals a ON a.change_id=c.id WHERE a.decided_at>=?", (since,))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["kind"]].append(row)

    facts: list[tuple[str, str, str, str | None]] = []
    for kind, decisions in grouped.items():
        label = kind.replace("_", " ")
        rejected = [d for d in decisions if d["decision"] == "rejected"]
        approved = [d for d in decisions if d["decision"] == "approved"]
        reasons: dict[str, int] = defaultdict(int)
        for d in rejected:
            reasons[d["reason_code"] or "unstated"] += 1
        top_reason = max(reasons.items(), key=lambda kv: kv[1])[0] if reasons else None
        parts = [f"Operator approved {len(approved)} and rejected {len(rejected)} {label} proposals "
                 f"in the last {LESSON_WINDOW_DAYS} days"]
        if top_reason and top_reason != "unstated":
            parts.append(f"most common rejection reason: {top_reason.replace('_', ' ')}")
        approved_sizes = [m for d in approved if (m := _magnitude(kind, json.loads(d["before_doc"]), json.loads(d["after_doc"]))) is not None]
        rejected_sizes = [m for d in rejected if (m := _magnitude(kind, json.loads(d["before_doc"]), json.loads(d["after_doc"]))) is not None]
        if approved_sizes:
            parts.append(f"approved up to {max(approved_sizes):g}%")
        if rejected_sizes:
            parts.append(f"rejected from {min(rejected_sizes):g}%")
        facts.append((f"lesson:{kind}", "; ".join(parts) + ".", "constraint", None))

    if outcomes and outcomes.get("offers_issued"):
        facts.append((
            "outcome:cart_recovery",
            (f"Cart recovery, last {outcomes['window_days']} days: {outcomes['offers_issued']} offers "
             f"({outcomes['coupons']} coupons, {outcomes['reminders']} reminders), "
             f"{outcomes['offers_redeemed']} redeemed ({outcomes['redemption_rate']:.0%}), "
             f"₹{outcomes['recovered_revenue_minor'] // 100:,} recovered for "
             f"₹{outcomes['discount_given_minor'] // 100:,} of discount."),
            "context", None))
    return lessons.write(MERCHANT_SUBJECT, facts)
