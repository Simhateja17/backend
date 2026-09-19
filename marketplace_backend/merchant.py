"""The host's merchant surface.

This is the other half of ADR 0016. The agent's reach ends at a `pending` row in
`merchant_changes`; everything past that point lives here, behind an
authenticated `merchant_operator` principal, and none of it is callable by a
model. `MerchantService` has no tool contract, appears in no tool list, and is
reached only from the REST endpoints the portal's approval queue calls.

Approval and application are two steps for the same reason staging and
confirming are on the shopping side: they have different authorities and
different failure modes. The operator's decision is recorded first and stands on
its own — a rejection is a complete, evidenced outcome. Application then
re-reads the record, refuses a proposal whose target has moved
(`StaleProposal`) or whose bounds no longer hold (`PolicyViolation`), and only
then writes.

The writes go to the normalized commerce core — `variant_prices`,
`inventory_levels` and `inventory_movements`, `catalog_products`, `promotions`,
`campaigns` — which is what retires the flat `products` table the merchant
surfaces used to read. One catalogue: an approved price change moves the price a
shopper is charged and the price the agent quotes, because they are the same row.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from cartisan_agent.merchant_core_port import CoreMerchantPort
from cartisan_agent.merchant_types import MerchantSessionContext

from .evidence import CommerceEventLog, Correlation, EvidenceLedger
from .merchant_changes import (
    MerchantChangeRepository,
    PolicyViolation,
    StaleProposal,
)
from .state_machines import TransitionError
from .store import Store
from .timeutil import now as iso_now


class DecisionRefused(Exception):
    """A decision or application the rules do not allow. Carries the sentence the
    operator sees, and nothing was changed."""


class MerchantService:
    def __init__(
        self,
        store: Store,
        port: CoreMerchantPort,
        changes: MerchantChangeRepository,
        ledger: EvidenceLedger | None = None,
        events: CommerceEventLog | None = None,
    ) -> None:
        self.store, self.port, self.changes = store, port, changes
        self.ledger = ledger or EvidenceLedger(store)
        self.events = events or CommerceEventLog(store)

    @staticmethod
    def session(operator_id: str) -> MerchantSessionContext:
        return MerchantSessionContext(
            conversation_id=f"rest:{operator_id}", customer_id=operator_id
        )

    # ------------------------------------------------------------------ reads

    async def snapshot(self, operator_id: str, window_days: int = 7) -> dict:
        snapshot = await self.port.get_business_snapshot(self.session(operator_id), window_days)
        return {
            "window_days": snapshot.window_days,
            "currency": snapshot.currency,
            "origins": snapshot.origins,
            "claims": [claim.payload() for claim in snapshot.claims],
            "movements": [claim.payload() for claim in snapshot.movements],
            "limitations": snapshot.limitations,
        }

    async def metrics(self, operator_id: str, metric: str, window_days: int = 30,
                      group_by: str | None = None) -> dict:
        series = await self.port.query_metrics(
            self.session(operator_id), metric, window_days, group_by)
        return series.model_dump()

    def changes_list(self, limit: int = 50, conversation_id: str | None = None) -> list[dict]:
        """What the approval queue shows: pending first, then what was decided, each
        carrying the approvals recorded against it.

        `conversation_id` scopes the decided half to one conversation; pending changes
        are never scoped, because an unanswered proposal is outstanding wherever the
        operator is standing."""
        return [self._decorate(change)
                for change in self.changes.recent(limit=limit, conversation_id=conversation_id)]

    def change(self, change_id: str) -> dict:
        try:
            return self._decorate(self.changes.read(change_id))
        except ValueError as exc:
            raise LookupError(str(exc)) from exc

    def _decorate(self, change: dict) -> dict:
        return {**change, "approvals": self.changes.approvals(change["id"])}

    # -------------------------------------------------------------- decisions

    def decide(self, *, operator_id: str, change_id: str, decision: str,
               note: str | None = None) -> dict:
        """Record the operator's decision, and apply it when it is an approval.

        Application is not a separate act the operator has to remember: approving *is*
        the instruction to apply. What stays separate is the record — the decision is
        committed before anything is written, so a refusal at application time leaves an
        approval that happened and a change that failed, which is the truth of what
        occurred rather than a decision quietly erased.
        """
        correlation = Correlation(correlation_id=f"corr_{uuid4().hex[:12]}")
        try:
            decided = self.changes.decide(
                change_id=change_id, operator_id=operator_id, decision=decision, note=note,
                correlation=correlation)
        except TransitionError as exc:
            raise DecisionRefused(
                f"That change cannot be {decision}: {exc}. Nothing was changed.") from exc
        except ValueError as exc:
            raise DecisionRefused(str(exc)) from exc
        if decision != "approved":
            return self._decorate(decided)
        return self.apply(operator_id=operator_id, change_id=change_id, correlation=correlation)

    def apply(self, *, operator_id: str, change_id: str,
              correlation: Correlation | None = None) -> dict:
        """Write an approved change, after the repository has revalidated it against
        current state. A refusal here is a refusal to write, and it says which of the
        two checks refused it."""
        try:
            applied = self.changes.apply(
                change_id=change_id, operator_id=operator_id, applier=self._apply,
                current=self.port.current_before,
                correlation=correlation or Correlation(correlation_id=f"corr_{uuid4().hex[:12]}"))
        except StaleProposal as exc:
            raise DecisionRefused(
                f"This change was not applied because {exc}. The approval is recorded and "
                "the change is marked failed; ask the agent to stage it again against "
                "current figures.") from exc
        except PolicyViolation as exc:
            raise DecisionRefused(
                f"This change was not applied: {exc}. The approval is recorded and the "
                "change is marked failed; nothing was written.") from exc
        except TransitionError as exc:
            raise DecisionRefused(f"That change cannot be applied: {exc}.") from exc
        return self._decorate(applied)

    # ------------------------------------------------------------- the writes

    def _apply(self, change: dict, tx: Any) -> None:
        """The one place a merchant change becomes a fact.

        Runs inside the repository's transaction, beside the status update and the
        evidence record, so an applied change and the row that says it was applied
        commit together or not at all (ADR 0009).
        """
        kind, target = change["kind"], change["target_id"]
        after = change["after"]
        if kind == "price_update":
            self._apply_price(tx, target, after)
        elif kind == "inventory_action":
            self._apply_inventory(tx, target, after, change)
        elif kind == "listing_update":
            self._apply_listing(tx, target, after)
        elif kind == "promotion":
            self._apply_promotion(tx, after)
        elif kind == "campaign":
            self._apply_campaign(tx, after)
        elif kind == "recovery_policy":
            self._apply_recovery_policy(tx, after, change["id"])
        elif kind == "loan_request":
            self._apply_loan_request(tx, change)
        else:  # unreachable: `stage` refuses an unknown kind
            raise DecisionRefused(f"no applier for change kind {kind!r}")

    @staticmethod
    def _apply_price(tx: Any, variant_id: str, after: dict) -> None:
        """Close the price in force and open a new one. Prices are a history, not a
        column: an order placed yesterday must still be explicable at yesterday's
        price, so nothing is overwritten."""
        now = iso_now()
        tx.execute(
            "UPDATE variant_prices SET valid_to=? WHERE variant_id=? AND valid_to IS NULL",
            (now, variant_id))
        tx.execute(
            "INSERT INTO variant_prices (id,variant_id,currency,amount_minor,price_kind,"
            "valid_from) VALUES (?,?,'INR',?,'list',?)",
            (f"price_{uuid4().hex[:12]}", variant_id, int(after["amount_minor"]), now))

    @staticmethod
    def _apply_inventory(tx: Any, variant_id: str, after: dict, change: dict) -> None:
        """Move stock and explain the movement. Every change to `on_hand` is matched by
        one `inventory_movements` row, which is what makes `reconcile` meaningful."""
        units = int(after["units"])
        rows = tx.rows(
            "SELECT location_id, on_hand, reserved FROM inventory_levels WHERE variant_id=? "
            "ORDER BY location_id", (variant_id,))
        if not rows:
            raise DecisionRefused(
                f"variant {variant_id} has no inventory level to move; nothing was written.")
        location = after.get("location_id") or rows[0]["location_id"]
        level = next((row for row in rows if row["location_id"] == location), rows[0])
        location = level["location_id"]
        target = int(level["on_hand"]) + units
        if target < int(level["reserved"]):
            # Reserved stock belongs to confirmed orders; a write-off cannot take it.
            raise DecisionRefused(
                f"removing {abs(units)} units would leave {target} on hand against "
                f"{level['reserved']} already reserved for confirmed orders; nothing was written.")
        tx.execute(
            "UPDATE inventory_levels SET on_hand=?, updated_at=? WHERE variant_id=? AND location_id=?",
            (target, iso_now(), variant_id, location))
        reason = {"restock": "receipt", "write_off": "damage"}.get(
            str(after.get("action")), "adjustment")
        tx.execute(
            "INSERT INTO inventory_movements (id,variant_id,location_id,delta,reason,"
            "reference_type,reference_id,created_at) VALUES (?,?,?,?,?,'merchant_change',?,?)",
            (f"mv_{uuid4().hex[:12]}", variant_id, location, units, reason, change["id"],
             iso_now()))

    @staticmethod
    def _apply_listing(tx: Any, product_id: str, after: dict) -> None:
        for field in ("title", "description", "status"):
            if field in after:
                tx.execute(
                    f"UPDATE catalog_products SET {field}=? WHERE id=?",
                    (after[field], product_id))

    @staticmethod
    def _apply_promotion(tx: Any, after: dict) -> None:
        tx.execute(
            "INSERT INTO promotions (id,code,description,discount_kind,discount_value,"
            "min_subtotal_minor,status,starts_at) VALUES (?,?,?,?,?,?,'draft',?)",
            (f"promo_{uuid4().hex[:12]}", after["code"], after["description"],
             after["discount_kind"], int(after["discount_value"]),
             int(after.get("min_subtotal_minor") or 0), iso_now()))

    @staticmethod
    def _apply_loan_request(tx: Any, change: dict) -> None:
        """Record the request as submitted to Paytm's lending partner. Simulated: no
        lender is called and no money moves; the row is the whole effect."""
        after = change["after"]
        tx.execute(
            "INSERT INTO loan_requests (id,change_id,amount_minor,tenure_months,purpose,"
            "eligible_limit_minor,status,created_at) VALUES (?,?,?,?,?,?,'submitted',?)",
            (f"loan_{uuid4().hex[:12]}", change["id"], int(after["amount_minor"]),
             int(after["tenure_months"]), after["purpose"],
             int(change["before"]["eligible_limit_minor"]), iso_now()))

    @staticmethod
    def _apply_recovery_policy(tx: Any, after: dict, change_id: str) -> None:
        """One active policy: the approved one replaces whatever was in force. Offers
        already issued keep the terms they were issued under."""
        now = iso_now()
        tx.execute("UPDATE recovery_policies SET status='retired', retired_at=? WHERE status='active'",
                   (now,))
        tx.execute(
            "INSERT INTO recovery_policies (id,status,abandon_after_minutes,min_cart_minor,"
            "discount_percentage,max_discount_minor,cooldown_days,monthly_budget_minor,"
            "offer_valid_hours,change_id,created_at) VALUES (?,'active',?,?,?,?,?,?,?,?,?)",
            (f"rpol_{uuid4().hex[:12]}", after["abandon_after_minutes"], after["min_cart_minor"],
             after["discount_percentage"], after["max_discount_minor"], after["cooldown_days"],
             after["monthly_budget_minor"], after["offer_valid_hours"], change_id, now))

    @staticmethod
    def _apply_campaign(tx: Any, after: dict) -> None:
        promotion_id = None
        if after.get("promotion_code"):
            rows = tx.rows("SELECT id FROM promotions WHERE code=?", (after["promotion_code"],))
            promotion_id = rows[0]["id"] if rows else None
        # A campaign lands as a draft with no spend. Starting one and spending money on
        # it are outside Cartisan, and a campaign that began itself on approval would be
        # an external effect nobody asked for (ADR 0024).
        tx.execute(
            "INSERT INTO campaigns (id,name,channel,promotion_id,status,budget_minor,"
            "spend_minor,starts_at) VALUES (?,?,?,?,'draft',?,0,?)",
            (f"camp_{uuid4().hex[:12]}", after["name"], after["channel"], promotion_id,
             int(after["budget_minor"]), iso_now()))
