"""The durable turn-and-tool state machine (ADR 0029).

A turn is a row in `turns` that moves `received -> running -> completed | failed |
abandoned`, and every tool call inside it is a row in `tool_executions` carrying its
arguments, its typed outcome, and its latency. Two things follow from that, and both
are the point of persisting turns at all:

* **Reconnect.** A disconnected client rejoins the turn already running rather than
  starting a second one, and after a process failure `recover_stale` closes what can no
  longer be resumed instead of leaving it running forever.
* **Evidence.** Each tool execution writes an `evidence_records` row carrying the turn
  and execution ids, the prompt and skill versions, and the outcome — blocked,
  unavailable and failed included (ADR 0023). One lineage, `correlation_id`, ties the
  browser's request to the model call, the tool, the database work, and later the
  Paytm attempt.

Turns are serialized per conversation and concurrent across principals: the lock is
keyed by conversation, and `begin` refuses a second turn while one is live.
"""

from __future__ import annotations

import asyncio
import weakref
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from marketplace_backend.evidence import Actor, Correlation, EvidenceLedger
from marketplace_backend.store import Store
from marketplace_backend.timeutil import as_datetime, now as iso_now

from .outcomes import Outcome
from .types import SessionContext

LIVE_STATES = ("received", "running", "awaiting_tool")


class TurnInProgress(Exception):
    """A turn is already live for this conversation. The host reconnects to it rather
    than starting another; `turn_id` says which."""

    def __init__(self, turn_id: str) -> None:
        super().__init__(f"conversation already has a live turn: {turn_id}")
        self.turn_id = turn_id


@dataclass
class TurnRecord:
    turn_id: str
    conversation_id: str
    sequence: int
    correlation_id: str
    prompt_version: str
    tool_contract_version: str
    # The stamp stored on the turn (`name@digest` per skill, JSON) and the plain names
    # the evidence ledger carries beside each record.
    skill_versions: str
    skill_names: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


_turn_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()


def conversation_lock(conversation_id: str) -> asyncio.Lock:
    lock = _turn_locks.get(conversation_id)
    if lock is None:
        lock = _turn_locks[conversation_id] = asyncio.Lock()
    return lock


class TurnStore:
    def __init__(self, store: Store, ledger: EvidenceLedger | None = None) -> None:
        self.store = store
        self.ledger = ledger or EvidenceLedger(store)

    # -- conversations --------------------------------------------------------

    def ensure_conversation(self, session: SessionContext) -> str:
        rows = self.store.rows(
            "SELECT id FROM conversations WHERE id = ?", (session.conversation_id,)
        )
        if rows:
            return session.conversation_id
        self.store.execute(
            "INSERT INTO conversations (id,principal_id,surface,created_at) VALUES (?,?,?,?)",
            (session.conversation_id, session.customer_id, session.surface, iso_now()),
        )
        return session.conversation_id

    def live_turn(self, conversation_id: str) -> dict | None:
        rows = self.store.rows(
            "SELECT * FROM turns WHERE conversation_id = ? AND state IN "
            f"({','.join('?' * len(LIVE_STATES))}) ORDER BY sequence DESC LIMIT 1",
            (conversation_id, *LIVE_STATES),
        )
        return rows[0] if rows else None

    def read_turn(self, turn_id: str) -> dict | None:
        rows = self.store.rows("SELECT * FROM turns WHERE id = ?", (turn_id,))
        return rows[0] if rows else None

    # -- the state machine ----------------------------------------------------

    def begin(
        self,
        session: SessionContext,
        *,
        user_message: str,
        prompt_version: str,
        tool_contract_version: str,
        skill_versions: str,
        skill_names: list[str] | None = None,
        correlation_id: str | None = None,
    ) -> TurnRecord:
        """Open a turn, or raise `TurnInProgress` when one is already live."""
        self.ensure_conversation(session)
        live = self.live_turn(session.conversation_id)
        if live is not None:
            raise TurnInProgress(live["id"])
        sequence = self.store.rows(
            "SELECT COALESCE(MAX(sequence), -1) AS last FROM turns WHERE conversation_id = ?",
            (session.conversation_id,),
        )[0]["last"]
        turn_id = f"turn_{uuid4().hex[:12]}"
        # The turn continues the lineage the request arrived on rather than starting
        # its own: a browser action and the turn it causes are one journey, not two.
        lineage = correlation_id or session.correlation_id or f"corr_{uuid4().hex[:12]}"
        self.store.execute(
            "INSERT INTO turns (id,conversation_id,sequence,state,user_message,prompt_version,"
            "tool_contract_version,skill_versions,correlation_id,demo_run_id,started_at) "
            "VALUES (?,?,?,'running',?,?,?,?,?,?,?)",
            (
                turn_id,
                session.conversation_id,
                int(sequence) + 1,
                user_message,
                prompt_version,
                tool_contract_version,
                skill_versions,
                lineage,
                session.demo_run_id,
                iso_now(),
            ),
        )
        return TurnRecord(
            turn_id=turn_id,
            conversation_id=session.conversation_id,
            sequence=int(sequence) + 1,
            correlation_id=lineage,
            prompt_version=prompt_version,
            tool_contract_version=tool_contract_version,
            skill_versions=skill_versions,
            skill_names=list(skill_names or []),
        )

    def record_tool(
        self,
        turn: TurnRecord,
        session: SessionContext,
        *,
        name: str,
        arguments: dict[str, Any],
        outcome: Outcome,
        result: str,
        latency_ms: int,
        reason: str,
        target_type: str | None = None,
        target_id: str | None = None,
    ) -> str:
        """One tool execution and its evidence record, written together. Every call is
        recorded, whatever its outcome; routine UI polling never reaches here."""
        execution_id = f"tex_{uuid4().hex[:12]}"
        with self.store.transaction() as tx:
            tx.execute(
                "INSERT INTO tool_executions (id,turn_id,tool_name,arguments,outcome,result,"
                "latency_ms,created_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    execution_id,
                    turn.turn_id,
                    name,
                    self.store.dump(arguments),
                    str(outcome),
                    result[:4000],
                    latency_ms,
                    iso_now(),
                ),
            )
            self.ledger.record(
                actor=Actor("agent", session.customer_id, session.surface),
                action=name,
                reason=reason,
                outcome=str(outcome),
                target_type=target_type,
                target_id=target_id,
                state_ref={"turn_id": turn.turn_id, "arguments": arguments},
                correlation=Correlation(
                    correlation_id=turn.correlation_id,
                    turn_id=turn.turn_id,
                    tool_execution_id=execution_id,
                    demo_run_id=session.demo_run_id,
                    prompt_version=turn.prompt_version,
                    skill_versions=turn.skill_names,
                ),
                tx=tx,
            )
        return execution_id

    def complete(self, turn: TurnRecord, *, agent_message: str, usage: dict[str, int]) -> None:
        self.store.execute(
            "UPDATE turns SET state='completed', agent_message=?, input_tokens=?, "
            "output_tokens=?, cache_read_tokens=?, completed_at=? WHERE id=?",
            (
                agent_message,
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                usage.get("cache_read_input_tokens", 0),
                iso_now(),
                turn.turn_id,
            ),
        )

    def fail(self, turn: TurnRecord, reason: str) -> None:
        """A turn that raised. It is closed rather than left running, so the next
        request opens a new one instead of reconnecting to a dead one."""
        self.store.execute(
            "UPDATE turns SET state='failed', agent_message=?, completed_at=? WHERE id=?",
            (reason[:2000], iso_now(), turn.turn_id),
        )

    def abandon(self, turn_id: str) -> None:
        self.store.execute(
            "UPDATE turns SET state='abandoned', completed_at=? WHERE id=? AND state IN "
            f"({','.join('?' * len(LIVE_STATES))})",
            (iso_now(), turn_id, *LIVE_STATES),
        )

    def recover_stale(self, *, older_than_seconds: int = 300) -> list[str]:
        """Turns left live by a process that died. Called on startup and by the host's
        sweeper; a turn younger than the cutoff is assumed still streaming somewhere."""
        from datetime import UTC, datetime, timedelta

        cutoff = datetime.now(UTC) - timedelta(seconds=older_than_seconds)
        stale = [
            row["id"]
            for row in self.store.rows(
                "SELECT id, started_at FROM turns WHERE state IN "
                f"({','.join('?' * len(LIVE_STATES))})",
                LIVE_STATES,
            )
            if (started := as_datetime(row["started_at"])) is not None and started <= cutoff
        ]
        for turn_id in stale:
            self.abandon(turn_id)
        return stale

    # -- reconnect ------------------------------------------------------------

    def resume(self, conversation_id: str) -> dict | None:
        """What a reconnecting client should be shown: the live turn to wait on, or the
        last completed turn's reply when the turn finished while it was away."""
        live = self.live_turn(conversation_id)
        if live is not None:
            return {"state": live["state"], "turn_id": live["id"], "agent_message": None}
        rows = self.store.rows(
            "SELECT id,state,agent_message FROM turns WHERE conversation_id = ? "
            "ORDER BY sequence DESC LIMIT 1",
            (conversation_id,),
        )
        if not rows:
            return None
        return {
            "state": rows[0]["state"],
            "turn_id": rows[0]["id"],
            "agent_message": rows[0]["agent_message"],
        }

    def history(self, conversation_id: str, limit: int = 40) -> list[dict]:
        """The conversation as the `turns` table remembers it, oldest first.

        This is not the model's message array — the tool_use blocks that array needs
        stay in the process that is running the turn. It is what a person should see
        again after a reload or a restart: what they asked, and what they were told.
        """
        rows = self.store.rows(
            "SELECT id,sequence,state,user_message,agent_message,correlation_id,started_at "
            "FROM turns WHERE conversation_id = ? ORDER BY sequence DESC LIMIT ?",
            (conversation_id, max(1, min(limit, 200))),
        )
        return list(reversed(rows))

    def tool_executions(self, turn_id: str) -> list[dict]:
        return self.store.rows(
            "SELECT tool_name,arguments,outcome,result,latency_ms FROM tool_executions "
            "WHERE turn_id = ? ORDER BY created_at, id",
            (turn_id,),
        )
