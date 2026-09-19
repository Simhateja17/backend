"""The append-only evidence ledger, the transactional outbox, and the provider inbox.

These three together are what makes an external effect safe to attempt: the
internal write and the *intent* to act externally commit atomically (outbox), the
provider's answer is recorded once and only once (inbox), and every step leaves a
row that explains who did what, why, and with what outcome (evidence).

Nothing here is authoritative commerce state. The ledger explains operations; it
never replaces the transactional tables (ADR 0009).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from .state_machines import INBOX, OUTBOX, TransitionError
from .store import Store
from .timeutil import now as _now

ORIGINS = ("seeded", "live_app", "razorpay_test")
OUTCOMES = ("applied", "blocked", "unavailable", "failed", "conflict")


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:16]}"


@dataclass(frozen=True)
class Actor:
    """Who is acting. Always derived from verified authentication, never claimed."""

    type: str  # customer | merchant_operator | agent | system | provider
    id: str | None = None
    surface: str | None = None


@dataclass
class Correlation:
    """One lineage across browser action, turn, tool call, database work and provider call."""

    correlation_id: str = field(default_factory=lambda: _id("corr"))
    turn_id: str | None = None
    tool_execution_id: str | None = None
    demo_run_id: str | None = None
    prompt_version: str | None = None
    skill_versions: list[str] | None = None


class EvidenceLedger:
    """Append-only. Records successes, refusals, and failures alike (ADR 0023)."""

    def __init__(self, store: Store) -> None:
        self.store = store

    def record(self, *, actor: Actor, action: str, reason: str, outcome: str,
               target_type: str | None = None, target_id: str | None = None,
               policy_checks: Any = None, state_ref: Any = None,
               data_origin: str = "live_app", correlation: Correlation | None = None,
               tx: Any = None) -> dict:
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown evidence outcome {outcome!r}; expected one of {OUTCOMES}")
        if data_origin not in ORIGINS:
            raise ValueError(f"unknown data origin {data_origin!r}; expected one of {ORIGINS}")
        correlation = correlation or Correlation()
        row = {
            "id": _id("ev"), "recorded_at": _now(), "actor_type": actor.type, "actor_id": actor.id,
            "surface": actor.surface, "action": action, "target_type": target_type,
            "target_id": target_id, "reason": reason, "outcome": outcome,
            "policy_checks": json.dumps(policy_checks) if policy_checks is not None else None,
            "state_ref": json.dumps(state_ref) if state_ref is not None else None,
            "prompt_version": correlation.prompt_version,
            "skill_versions": json.dumps(correlation.skill_versions) if correlation.skill_versions else None,
            "data_origin": data_origin, "demo_run_id": correlation.demo_run_id,
            "correlation_id": correlation.correlation_id, "turn_id": correlation.turn_id,
            "tool_execution_id": correlation.tool_execution_id,
        }
        # `tx` lets a caller commit the evidence in the same transaction as the
        # state change it explains, so a successful mutation can never be silent.
        target = tx or self.store
        target.execute(
            "INSERT INTO evidence_records (id,recorded_at,actor_type,actor_id,surface,action,"
            "target_type,target_id,reason,outcome,policy_checks,state_ref,prompt_version,"
            "skill_versions,data_origin,demo_run_id,correlation_id,turn_id,tool_execution_id) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            tuple(row[key] for key in (
                "id", "recorded_at", "actor_type", "actor_id", "surface", "action",
                "target_type", "target_id", "reason", "outcome", "policy_checks", "state_ref",
                "prompt_version", "skill_versions", "data_origin", "demo_run_id",
                "correlation_id", "turn_id", "tool_execution_id")))
        return row

    def for_correlation(self, correlation_id: str) -> list[dict]:
        """One lineage, oldest first — the view a judge follows end to end."""
        return self.store.rows(
            "SELECT * FROM evidence_records WHERE correlation_id=? ORDER BY recorded_at", (correlation_id,))

    def for_actor(self, actor_id: str, limit: int = 200) -> list[dict]:
        return self.store.rows(
            "SELECT * FROM evidence_records WHERE actor_id=? ORDER BY recorded_at DESC LIMIT ?",
            (actor_id, min(limit, 500)))


class Outbox:
    """External effects leave through here (ADR 0024).

    `enqueue` is called inside the transaction that makes the internal change, so
    the effect is scheduled if and only if that change committed. Delivery happens
    afterwards and may be retried: the message, not the request, is the unit of work.
    """

    def __init__(self, store: Store, max_attempts: int = 5) -> None:
        self.store, self.max_attempts = store, max_attempts

    def enqueue(self, *, topic: str, payload: dict, correlation_id: str | None = None,
                tx: Any = None) -> str:
        message_id = _id("out")
        (tx or self.store).execute(
            "INSERT INTO outbox_messages (id,topic,payload,status,attempts,correlation_id,available_at,created_at) "
            "VALUES (?,?,?,'pending',0,?,?,?)",
            (message_id, topic, json.dumps(payload), correlation_id, _now(), _now()))
        return message_id

    def claim(self, limit: int = 10, topic: str | None = None) -> list[dict]:
        """Take due pending messages and mark them in flight, so two workers
        cannot deliver the same effect twice. A worker that handles one topic
        passes it, so it never claims (and burns an attempt on) another's message."""
        claimed = []
        topic_clause, topic_params = ("AND topic=? ", (topic,)) if topic else ("", ())
        with self.store.transaction() as tx:
            due = tx.rows(
                "SELECT * FROM outbox_messages WHERE status='pending' AND available_at<=? "
                f"{topic_clause}ORDER BY available_at LIMIT ?", (_now(), *topic_params, limit))
            for message in due:
                OUTBOX.check(message["status"], "in_flight")
                tx.execute(
                    "UPDATE outbox_messages SET status='in_flight', attempts=attempts+1 WHERE id=? AND status='pending'",
                    (message["id"],))
                message["payload"] = json.loads(message["payload"])
                claimed.append(message)
        return claimed

    def delivered(self, message_id: str) -> None:
        self._transition(message_id, "delivered", extra="delivered_at=?", params=(_now(),))

    def failed(self, message_id: str, error: str, retry_in_seconds: int = 30) -> str:
        """Record the failure, then either schedule a retry or park the message."""
        rows = self.store.rows("SELECT status,attempts FROM outbox_messages WHERE id=?", (message_id,))
        if not rows:
            raise ValueError(f"unknown outbox message {message_id!r}")
        OUTBOX.check(rows[0]["status"], "failed")
        self.store.execute(
            "UPDATE outbox_messages SET status='failed', last_error=? WHERE id=?", (error, message_id))
        if rows[0]["attempts"] >= self.max_attempts:
            OUTBOX.check("failed", "dead_letter")
            self.store.execute("UPDATE outbox_messages SET status='dead_letter' WHERE id=?", (message_id,))
            return "dead_letter"
        OUTBOX.check("failed", "pending")
        available = (datetime.now(UTC) + timedelta(seconds=retry_in_seconds)).isoformat()
        self.store.execute(
            "UPDATE outbox_messages SET status='pending', available_at=? WHERE id=?", (available, message_id))
        return "pending"

    def _transition(self, message_id: str, target: str, extra: str = "", params: tuple = ()) -> None:
        rows = self.store.rows("SELECT status FROM outbox_messages WHERE id=?", (message_id,))
        if not rows:
            raise ValueError(f"unknown outbox message {message_id!r}")
        OUTBOX.check(rows[0]["status"], target)
        clause = f", {extra}" if extra else ""
        self.store.execute(
            f"UPDATE outbox_messages SET status=?{clause} WHERE id=?", (target, *params, message_id))


class Inbox:
    """Durable, idempotent intake for provider callbacks.

    A redelivered webhook is stored once (unique on provider + event id) and
    processed once. A payload that does not match the order it claims is
    quarantined rather than applied.
    """

    def __init__(self, store: Store) -> None:
        self.store = store

    def receive(self, *, provider: str, provider_event_id: str, event_type: str,
                payload: dict, correlation: Correlation | None = None) -> tuple[dict, bool]:
        """Return (row, is_new). A duplicate returns the original row untouched.

        `correlation` is optional because a live webhook does not know its journey
        until the payload has been matched to an attempt; the processor sets it then.
        A caller that already knows — a scenario pack building a story on purpose —
        passes it here so the stored event joins that story from the start.
        """
        existing = self.store.rows(
            "SELECT * FROM inbox_events WHERE provider=? AND provider_event_id=?",
            (provider, provider_event_id))
        if existing:
            return existing[0], False
        row_id = _id("in")
        try:
            self.store.execute(
                "INSERT INTO inbox_events (id,provider,provider_event_id,event_type,payload,"
                "status,correlation_id,received_at) VALUES (?,?,?,?,?,'received',?,?)",
                (row_id, provider, provider_event_id, event_type, json.dumps(payload),
                 correlation.correlation_id if correlation else None, _now()))
        except Exception:
            # Lost a race with a concurrent delivery of the same event; the winner's
            # row is the one row that exists, which is exactly the intended outcome.
            existing = self.store.rows(
                "SELECT * FROM inbox_events WHERE provider=? AND provider_event_id=?",
                (provider, provider_event_id))
            if not existing:
                raise
            return existing[0], False
        return self.store.rows("SELECT * FROM inbox_events WHERE id=?", (row_id,))[0], True

    def mark(self, event_id: str, status: str, quarantine_reason: str | None = None) -> None:
        rows = self.store.rows("SELECT status FROM inbox_events WHERE id=?", (event_id,))
        if not rows:
            raise ValueError(f"unknown inbox event {event_id!r}")
        INBOX.check(rows[0]["status"], status)
        if status == "quarantined" and not quarantine_reason:
            raise ValueError("quarantining an event requires a reason")
        self.store.execute(
            "UPDATE inbox_events SET status=?, quarantine_reason=?, processed_at=? WHERE id=?",
            (status, quarantine_reason, _now(), event_id))

    def pending(self, limit: int = 50) -> list[dict]:
        return self.store.rows(
            "SELECT * FROM inbox_events WHERE status='received' ORDER BY received_at LIMIT ?", (limit,))


class CommerceEventLog:
    """Append-only business facts. Derived metrics read these, never running totals."""

    def __init__(self, store: Store) -> None:
        self.store = store

    def append(self, *, event_type: str, subject_type: str, subject_id: str,
               customer_id: str | None = None, amount_minor: int | None = None,
               quantity: int | None = None, origin: str = "live_app",
               correlation: Correlation | None = None, detail: Any = None,
               tx: Any = None) -> dict:
        if origin not in ORIGINS:
            raise ValueError(f"unknown origin {origin!r}; expected one of {ORIGINS}")
        correlation = correlation or Correlation()
        row_id = _id("evt")
        (tx or self.store).execute(
            "INSERT INTO commerce_events (id,occurred_at,event_type,subject_type,subject_id,"
            "customer_id,amount_minor,quantity,origin,demo_run_id,correlation_id,detail) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (row_id, _now(), event_type, subject_type, subject_id, customer_id, amount_minor,
             quantity, origin, correlation.demo_run_id, correlation.correlation_id,
             json.dumps(detail) if detail is not None else None))
        return {"id": row_id, "event_type": event_type, "subject_id": subject_id}


__all__ = [
    "Actor", "CommerceEventLog", "Correlation", "EvidenceLedger", "Inbox", "Outbox",
    "TransitionError",
]
