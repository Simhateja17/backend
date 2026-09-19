"""Staged merchant changes and host-controlled approval.

The split that matters: `stage` is the only verb the merchant agent can reach,
and it can only ever produce a `pending` row. `approve`/`reject` require an
authenticated operator, and `apply` re-validates policy against *current* state
before touching anything (ADR 0016). There is no method here that stages and
applies in one step, so no model-accessible path can apply a change.

Revalidation is deliberately against the world as it is at application time, not
against the documents the proposal stored. Two different things can go wrong
between staging and approval, and they are reported separately because they need
different answers:

* **Drift.** The `before` document no longer describes the record. The proposal
  was written about a price or a stock level that has since moved, so approving
  it would apply a diff to a different starting point. That is `StaleProposal`.
* **Bounds.** The proposal still describes the record, but measured against the
  record's *current* values the change is now outside policy — a 10% cut becomes
  a 40% cut when the list price drops underneath it. That is `PolicyViolation`.

Both leave the change `failed` with the reason in the evidence ledger, and
neither applies anything.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from .evidence import Actor, Correlation, EvidenceLedger
from .state_machines import MERCHANT_CHANGE
from .store import Store
from .timeutil import now as _now

KINDS = ("inventory_action", "price_update", "promotion", "campaign", "listing_update",
         "recovery_policy")

# Bounds re-checked at application time, not just at staging time, because the
# world may have moved between the proposal and the operator's decision.
POLICY_BOUNDS = {
    "price_update": {"max_change_ratio": 0.25},
    "inventory_action": {"max_units": 500},
    "promotion": {"max_percentage": 30, "max_fixed_minor": 500_000},
    "campaign": {"max_budget_minor": 5_000_000},
    "listing_update": {"max_title_chars": 140, "max_description_chars": 1200},
    # Cart recovery: the ceiling on what the deterministic worker may ever give away
    # without a further approval (Q3).
    "recovery_policy": {"max_percentage": 20, "max_discount_minor": 200_000,
                        "max_monthly_budget_minor": 5_000_000, "min_abandon_minutes": 30,
                        "min_cooldown_days": 7, "max_offer_valid_hours": 168},
}

LISTING_STATUSES = ("draft", "active", "discontinued")


class PolicyViolation(Exception):
    """The change is outside what policy allows, at staging or at application."""


class StaleProposal(PolicyViolation):
    """The record moved between staging and approval, so the `before` document the
    operator approved is no longer what the change would be applied to."""


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:16]}"


class MerchantChangeRepository:
    def __init__(self, store: Store, ledger: EvidenceLedger) -> None:
        self.store, self.ledger = store, ledger

    # -------------------------------------------------- staging (agent-reachable)

    def stage(self, *, operator_id: str, kind: str, target_type: str, target_id: str | None,
              before: dict, after: dict, rationale: str,
              conversation_id: str | None = None,
              correlation: Correlation | None = None) -> dict:
        """Record a proposal. Always `pending`; never applies anything."""
        correlation = correlation or Correlation()
        actor = Actor("agent", operator_id, "merchant")
        if kind not in KINDS:
            self.ledger.record(
                actor=actor, action="stage_merchant_change", reason=f"Unknown change kind {kind!r}",
                outcome="blocked", correlation=correlation)
            raise ValueError(f"unknown change kind {kind!r}; expected one of {KINDS}")
        if not rationale.strip():
            raise ValueError("a staged change requires a rationale")
        try:
            self.check_policy(kind, before, after)
        except PolicyViolation as exc:
            self.ledger.record(
                actor=actor, action="stage_merchant_change", reason=str(exc), outcome="blocked",
                target_type=target_type, target_id=target_id,
                policy_checks={"kind": kind, "before": before, "after": after},
                correlation=correlation)
            raise

        change_id = _id("chg")
        with self.store.transaction() as tx:
            tx.execute(
                "INSERT INTO merchant_changes (id,operator_id,conversation_id,kind,target_type,"
                "target_id,before_doc,after_doc,rationale,status,created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,'pending',?)",
                (change_id, operator_id, conversation_id, kind, target_type, target_id,
                 json.dumps(before), json.dumps(after), rationale, _now()))
            self.ledger.record(
                actor=actor, action="stage_merchant_change", reason=rationale, outcome="applied",
                target_type=target_type, target_id=target_id,
                policy_checks={"kind": kind, "bounds": POLICY_BOUNDS.get(kind, {}),
                               "checked_at": "staging"},
                state_ref={"change_id": change_id, "before": before, "after": after},
                correlation=correlation, tx=tx)
        return self.read(change_id)

    def read(self, change_id: str) -> dict:
        rows = self.store.rows("SELECT * FROM merchant_changes WHERE id=?", (change_id,))
        if not rows:
            raise ValueError(f"unknown merchant change {change_id!r}")
        change = rows[0]
        change["before"] = json.loads(change.pop("before_doc"))
        change["after"] = json.loads(change.pop("after_doc"))
        return change

    def pending(self, limit: int = 50) -> list[dict]:
        return [self.read(row["id"]) for row in self.store.rows(
            "SELECT id FROM merchant_changes WHERE status='pending' ORDER BY created_at LIMIT ?", (limit,))]

    def recent(self, limit: int = 50, conversation_id: str | None = None) -> list[dict]:
        """Everything the approval surface shows: pending first, then what was decided.

        The two halves are scoped differently on purpose. A pending change is work the
        operator still owes an answer on, so it follows them into any conversation. A
        decided one is a record of what was already answered, and it belongs to the
        conversation it was staged in — carrying it into a fresh chat would show an
        operator a rejection they made somewhere else as if it had just happened.
        Passing no `conversation_id` keeps the unscoped view, which is what the
        evidence surface and the tests read.
        """
        if conversation_id is None:
            return [self.read(row["id"]) for row in self.store.rows(
                "SELECT id FROM merchant_changes ORDER BY "
                "CASE status WHEN 'pending' THEN 0 ELSE 1 END, created_at DESC LIMIT ?", (limit,))]
        return [self.read(row["id"]) for row in self.store.rows(
            "SELECT id FROM merchant_changes WHERE status='pending' OR conversation_id=? "
            "ORDER BY CASE status WHEN 'pending' THEN 0 ELSE 1 END, created_at DESC LIMIT ?",
            (conversation_id, limit))]

    def approvals(self, change_id: str) -> list[dict]:
        return self.store.rows(
            "SELECT * FROM merchant_approvals WHERE change_id=? ORDER BY decided_at", (change_id,))

    # ------------------------------------------- decision (operator-only)

    def decide(self, *, change_id: str, operator_id: str, decision: str, note: str | None = None,
               correlation: Correlation | None = None) -> dict:
        if decision not in {"approved", "rejected"}:
            raise ValueError("a decision is either 'approved' or 'rejected'")
        change = self.read(change_id)
        MERCHANT_CHANGE.check(change["status"], decision)
        correlation = correlation or Correlation()
        with self.store.transaction() as tx:
            tx.execute(
                "UPDATE merchant_changes SET status=?, decided_at=? WHERE id=? AND status='pending'",
                (decision, _now(), change_id))
            tx.execute(
                "INSERT INTO merchant_approvals (id,change_id,operator_id,decision,note,policy_checks,decided_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (_id("apr"), change_id, operator_id, decision, note,
                 json.dumps({"bounds": POLICY_BOUNDS.get(change["kind"], {})}), _now()))
            # A rejection is evidence too, and it carries the same lineage a staging
            # and an application do: the exact documents, and why they were refused
            # (ADR 0023). "The operator said no" is not a reason a judge can check.
            self.ledger.record(
                actor=Actor("merchant_operator", operator_id, "merchant"),
                action=f"{decision}_merchant_change",
                reason=note or f"Operator {decision} the staged change", outcome="applied",
                target_type="merchant_change", target_id=change_id,
                policy_checks={"kind": change["kind"], "bounds": POLICY_BOUNDS.get(change["kind"], {}),
                               "checked_at": "decision"},
                state_ref={"kind": change["kind"], "target_type": change["target_type"],
                           "target_id": change["target_id"], "before": change["before"],
                           "after": change["after"], "rationale": change["rationale"]},
                correlation=correlation, tx=tx)
        return self.read(change_id)

    def apply(self, *, change_id: str, operator_id: str, applier: Callable[[dict, Any], None],
              current: Callable[[dict], dict] | None = None,
              correlation: Correlation | None = None) -> dict:
        """Apply an approved change, re-validating against current state first.

        `current` reads the live `before` document for the change's target. Without
        it the stored document stands in, which can only re-check bounds — so a host
        that owns real state passes it, and the drift check becomes meaningful.
        """
        change = self.read(change_id)
        MERCHANT_CHANGE.check(change["status"], "applied")
        correlation = correlation or Correlation()
        actor = Actor("merchant_operator", operator_id, "merchant")
        live = dict(change["before"])
        try:
            if current is not None:
                live = current(change)
                self.check_drift(change["before"], live)
            self.check_policy(change["kind"], live, change["after"])
        except PolicyViolation as exc:
            with self.store.transaction() as tx:
                tx.execute(
                    "UPDATE merchant_changes SET status='failed' WHERE id=? AND status='approved'",
                    (change_id,))
                self.ledger.record(
                    actor=actor, action="apply_merchant_change",
                    reason=f"Revalidation failed at application time: {exc}", outcome="blocked",
                    target_type="merchant_change", target_id=change_id,
                    policy_checks={"kind": change["kind"], "bounds": POLICY_BOUNDS.get(change["kind"], {}),
                                   "checked_at": "application", "staged_before": change["before"],
                                   "current_before": live, "after": change["after"],
                                   "violation": type(exc).__name__},
                    correlation=correlation, tx=tx)
            raise

        with self.store.transaction() as tx:
            applier(change, tx)
            tx.execute(
                "UPDATE merchant_changes SET status='applied', applied_at=? WHERE id=? AND status='approved'",
                (_now(), change_id))
            self.ledger.record(
                actor=actor, action="apply_merchant_change", reason=change["rationale"],
                outcome="applied", target_type=change["target_type"], target_id=change["target_id"],
                policy_checks={"kind": change["kind"], "bounds": POLICY_BOUNDS.get(change["kind"], {}),
                               "checked_at": "application", "current_before": live},
                state_ref={"change_id": change_id, "before": live, "after": change["after"]},
                correlation=correlation, tx=tx)
        return self.read(change_id)

    # ------------------------------------------------------------- policy

    @staticmethod
    def check_drift(staged_before: dict, live_before: dict) -> None:
        """Refuse when the record no longer matches the proposal's starting point.

        Only the keys the proposal actually stated are compared: a `before` naming a
        price says nothing about the title, and a change that moved elsewhere in the
        record is not this proposal's business.
        """
        moved = {
            key: {"staged": value, "current": live_before.get(key)}
            for key, value in staged_before.items()
            if live_before.get(key) != value
        }
        if moved:
            detail = "; ".join(
                f"{key} was {pair['staged']!r} when this was staged and is {pair['current']!r} now"
                for key, pair in sorted(moved.items()))
            raise StaleProposal(
                f"the record moved after this change was staged ({detail}), so the approved "
                "before-and-after no longer describes it; stage the change again against "
                "current figures")

    @staticmethod
    def check_policy(kind: str, before: dict, after: dict) -> None:
        """Deterministic bounds. The same function runs at staging and at application,
        and at application `before` is the record as it stands now."""
        if kind == "price_update":
            old, new = before.get("amount_minor"), after.get("amount_minor")
            if not isinstance(old, int) or not isinstance(new, int) or old <= 0 or new <= 0:
                raise PolicyViolation("a price update needs positive integer amount_minor in both documents")
            ratio = abs(new - old) / old
            limit = POLICY_BOUNDS["price_update"]["max_change_ratio"]
            if ratio > limit:
                raise PolicyViolation(
                    f"price change of {ratio:.0%} exceeds the {limit:.0%} bound")
        elif kind == "inventory_action":
            units = after.get("units")
            if not isinstance(units, int) or units == 0:
                raise PolicyViolation("an inventory action needs a non-zero integer unit count")
            if abs(units) > POLICY_BOUNDS["inventory_action"]["max_units"]:
                raise PolicyViolation(
                    f"{units} units exceeds the {POLICY_BOUNDS['inventory_action']['max_units']} unit bound")
            # Stock cannot be taken away that is not there. `on_hand` is the record's
            # current figure at application time, which is what makes this a real check
            # rather than a restatement of what staging already saw.
            on_hand = before.get("on_hand")
            if units < 0 and isinstance(on_hand, int) and on_hand + units < 0:
                raise PolicyViolation(
                    f"removing {abs(units)} units would take on-hand stock below zero "
                    f"(currently {on_hand})")
        elif kind == "promotion":
            discount_kind, value = after.get("discount_kind"), after.get("discount_value")
            if discount_kind not in {"percentage", "fixed_minor"}:
                raise PolicyViolation("a promotion discounts by 'percentage' or 'fixed_minor'")
            if not isinstance(value, int) or value <= 0:
                raise PolicyViolation("a promotion needs a positive integer discount_value")
            if discount_kind == "percentage":
                limit = POLICY_BOUNDS["promotion"]["max_percentage"]
                if value > limit:
                    raise PolicyViolation(f"a {value}% discount exceeds the {limit}% bound")
            else:
                limit = POLICY_BOUNDS["promotion"]["max_fixed_minor"]
                if value > limit:
                    raise PolicyViolation(
                        f"a fixed discount of {value} paise exceeds the {limit} paise bound")
            minimum = after.get("min_subtotal_minor", 0)
            if not isinstance(minimum, int) or minimum < 0:
                raise PolicyViolation("min_subtotal_minor must be a non-negative integer")
        elif kind == "campaign":
            budget = after.get("budget_minor")
            if not isinstance(budget, int) or budget < 0:
                raise PolicyViolation("a campaign needs a non-negative integer budget_minor")
            limit = POLICY_BOUNDS["campaign"]["max_budget_minor"]
            if budget > limit:
                raise PolicyViolation(
                    f"a budget of {budget} paise exceeds the {limit} paise bound")
        elif kind == "recovery_policy":
            bounds = POLICY_BOUNDS["recovery_policy"]
            fields = ("abandon_after_minutes", "min_cart_minor", "discount_percentage",
                      "max_discount_minor", "cooldown_days", "monthly_budget_minor",
                      "offer_valid_hours")
            for field in fields:
                if not isinstance(after.get(field), int) or after[field] < 0:
                    raise PolicyViolation(f"a recovery policy needs a non-negative integer {field}")
            if not 1 <= after["discount_percentage"] <= bounds["max_percentage"]:
                raise PolicyViolation(
                    f"a recovery discount must be 1-{bounds['max_percentage']}%")
            if not 0 < after["max_discount_minor"] <= bounds["max_discount_minor"]:
                raise PolicyViolation(
                    f"a recovery discount cap must be at most {bounds['max_discount_minor']} paise")
            if after["monthly_budget_minor"] > bounds["max_monthly_budget_minor"]:
                raise PolicyViolation(
                    f"a monthly recovery budget of {after['monthly_budget_minor']} paise exceeds "
                    f"the {bounds['max_monthly_budget_minor']} paise bound")
            if after["abandon_after_minutes"] < bounds["min_abandon_minutes"]:
                raise PolicyViolation(
                    f"a cart counts as abandoned after at least {bounds['min_abandon_minutes']} minutes")
            if after["cooldown_days"] < bounds["min_cooldown_days"]:
                raise PolicyViolation(
                    f"offers to one customer must be at least {bounds['min_cooldown_days']} days apart")
            if not 1 <= after["offer_valid_hours"] <= bounds["max_offer_valid_hours"]:
                raise PolicyViolation(
                    f"an offer is valid for 1-{bounds['max_offer_valid_hours']} hours")
        elif kind == "listing_update":
            if not any(key in after for key in ("title", "description", "status")):
                raise PolicyViolation(
                    "a listing update must change the title, the description, or the status")
            bounds = POLICY_BOUNDS["listing_update"]
            title, description = after.get("title"), after.get("description")
            if title is not None and not 3 <= len(str(title)) <= bounds["max_title_chars"]:
                raise PolicyViolation(
                    f"a listing title must be 3-{bounds['max_title_chars']} characters")
            if description is not None and not 20 <= len(str(description)) <= bounds["max_description_chars"]:
                raise PolicyViolation(
                    f"a listing description must be 20-{bounds['max_description_chars']} characters")
            status = after.get("status")
            if status is not None and status not in LISTING_STATUSES:
                raise PolicyViolation(
                    f"a listing status is one of {', '.join(LISTING_STATUSES)}")
