"""Payment health and restock financing for the merchant agent.

Two reads over records the store already keeps:

  `payment_health`     what Paytm has verified (paid orders), what is still waiting
                       for verification, and which payment attempts failed. Only a
                       verified payment counts as collected money (ADR 0013).
  `restock_financing`  what topping up fast-moving stock would cost against what
                       Paytm collected, and, when there is a shortfall, the size of a
                       Paytm merchant loan that would cover it.

Nothing here applies for a loan. A loan offer is staged like any merchant change
(`loan_request`) and only an operator's approval sends it, and that submission is
simulated: no lender is called and no money moves.

The cost of stock and the loan limit are estimates, and say so: the store records
selling prices, not purchase costs, so cost is a stated fraction of the selling
price; and the limit is an illustrative multiple of verified collections, not
Paytm's underwriting.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

from .store import Store

# Purchase cost as a share of selling price, for want of recorded costs.
COST_RATIO = 0.6
# Illustrative loan limit: this multiple of the last 30 days' verified collections.
LIMIT_MULTIPLE = 1.5
MAX_LOAN_MINOR = 50_00_000_00  # ₹50 lakh
STUCK_AFTER_MINUTES = 15
TENURE_MONTHS = (3, 6, 9, 12)
# Illustrative flat monthly rate, for the repayment estimate only.
MONTHLY_RATE = 0.015


def _cutoff(days: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days)).isoformat()


def _inr(minor: int) -> str:
    rupees = minor // 100
    s = str(rupees)
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        s = ",".join(groups) + "," + tail
    return f"₹{s}"


def verified_collections(store: Store, days: int) -> int:
    rows = store.rows(
        "SELECT COALESCE(SUM(amount_paid_minor),0) AS total FROM commerce_orders "
        "WHERE status='paid' AND created_at >= ?", (_cutoff(days),))
    return int(rows[0]["total"]) if rows else 0


def payment_health(store: Store, window_days: int = 7) -> dict[str, Any]:
    since = _cutoff(window_days)
    by_status = {
        row["status"]: {"orders": int(row["n"]), "amount_minor": int(row["total"])}
        for row in store.rows(
            "SELECT status, COUNT(*) AS n, COALESCE(SUM(total_minor),0) AS total "
            "FROM commerce_orders WHERE created_at >= ? GROUP BY status", (since,))
    }
    stuck_before = (datetime.now(UTC) - timedelta(minutes=STUCK_AFTER_MINUTES)).isoformat()
    stuck = [
        {"order_id": row["id"], "status": row["status"], "amount": _inr(int(row["total_minor"])),
         "since": str(row["created_at"])}
        for row in store.rows(
            "SELECT id,status,total_minor,created_at FROM commerce_orders "
            "WHERE status IN ('pending_payment','payment_verification_pending') "
            "AND created_at >= ? AND created_at < ? ORDER BY created_at LIMIT 5",
            (since, stuck_before))
    ]
    failures = [
        {"reason": row["reason"] or "unspecified", "attempts": int(row["n"])}
        for row in store.rows(
            "SELECT failure_reason AS reason, COUNT(*) AS n FROM payment_attempts "
            "WHERE status='failed' AND created_at >= ? GROUP BY failure_reason ORDER BY n DESC",
            (since,))
    ]
    paid = by_status.get("paid", {"orders": 0, "amount_minor": 0})
    waiting = sum(by_status.get(s, {}).get("amount_minor", 0)
                  for s in ("pending_payment", "payment_verification_pending"))
    return {
        "window_days": window_days,
        "verified_collections": _inr(paid["amount_minor"]),
        "verified_collections_minor": paid["amount_minor"],
        "paid_orders": paid["orders"],
        "awaiting_verification": _inr(waiting),
        "awaiting_verification_minor": waiting,
        "orders_by_status": by_status,
        "stuck_orders": stuck,
        "failed_attempts": failures,
        "basis": (
            "Collected money is the sum of amount_paid on orders Paytm has verified as paid. "
            "Orders still pending or awaiting verification are not revenue yet; stuck orders "
            f"have waited more than {STUCK_AFTER_MINUTES} minutes."
        ),
    }


def _restock_lines(store: Store, horizon_days: int, demand: float = 1.0, limit: int = 10) -> list[dict[str, Any]]:
    """Variants that sell, with too little stock to last the horizon plus a 20% buffer."""
    rows = store.rows(
        "SELECT v.id AS variant_id, v.title AS variant_title, p.title AS product_title, "
        "COALESCE((SELECT SUM(l.on_hand - l.reserved) FROM inventory_levels l "
        "          WHERE l.variant_id = v.id),0) AS sellable, "
        "COALESCE((SELECT SUM(ol.quantity) FROM commerce_order_lines ol "
        "          JOIN commerce_orders o ON o.id = ol.order_id "
        "          WHERE ol.variant_id = v.id AND o.status='paid' AND o.created_at >= ?),0) AS sold "
        "FROM catalog_variants v JOIN catalog_products p ON p.id = v.product_id",
        (_cutoff(30),))
    lines = []
    for row in rows:
        rate = int(row["sold"]) / 30
        if rate <= 0:
            continue
        need = math.ceil(rate * demand * horizon_days * 1.2) - int(row["sellable"])
        if need <= 0:
            continue
        price = store.rows(
            "SELECT amount_minor FROM variant_prices WHERE variant_id = ? "
            "AND (valid_to IS NULL OR valid_to > ?) ORDER BY valid_from DESC LIMIT 1",
            (row["variant_id"], datetime.now(UTC).isoformat()))
        unit_price = int(price[0]["amount_minor"]) if price else 0
        unit_cost = int(unit_price * COST_RATIO)
        lines.append({
            "variant_id": row["variant_id"],
            "title": f"{row['product_title']} — {row['variant_title']}",
            "sellable": int(row["sellable"]), "daily_sales": round(rate, 2),
            "units_to_order": need, "unit_cost_estimate": _inr(unit_cost),
            "cost_minor": need * unit_cost,
        })
    lines.sort(key=lambda line: line["cost_minor"], reverse=True)
    return lines[:limit]


def loan_limit(store: Store) -> int:
    return min(int(verified_collections(store, 30) * LIMIT_MULTIPLE), MAX_LOAN_MINOR)


def monthly_repayment(amount_minor: int, tenure_months: int) -> int:
    # Integer arithmetic in tenths of a percent, so paise never pick up float error.
    total = amount_minor * (1000 + int(MONTHLY_RATE * 1000) * tenure_months)
    return -(-total // (1000 * tenure_months))


def restock_financing(store: Store, horizon_days: int = 21,
                      demand_multiplier: float = 1.0) -> dict[str, Any]:
    """`demand_multiplier` is the operator's own expectation (2.0 for a festival
    week that sells double), applied to the observed daily sales rate."""
    demand = max(1.0, min(float(demand_multiplier or 1.0), 3.0))
    lines = _restock_lines(store, horizon_days, demand)
    cost = sum(line["cost_minor"] for line in lines)
    cash = verified_collections(store, 7)
    shortfall = max(0, cost - cash)
    limit = loan_limit(store)
    # Round the suggestion up to the next ₹10,000, and never past the limit.
    step = 10_000_00
    suggested = min(math.ceil(shortfall / step) * step, limit) if shortfall else 0
    return {
        "horizon_days": horizon_days,
        "demand_multiplier": demand,
        "restock_lines": lines,
        "restock_cost_estimate": _inr(cost),
        "restock_cost_minor": cost,
        "cash_last_7_days": _inr(cash),
        "cash_last_7_days_minor": cash,
        "shortfall": _inr(shortfall),
        "shortfall_minor": shortfall,
        "loan": ({
            "suggested_amount": _inr(suggested),
            "suggested_amount_minor": suggested,
            "eligible_limit": _inr(limit),
            "eligible_limit_minor": limit,
            "tenure_months_options": list(TENURE_MONTHS),
            "monthly_repayment_at_6_months": _inr(monthly_repayment(suggested, 6)),
        } if suggested else None),
        "claim_kinds": {
            "cash_last_7_days": "observed: verified Paytm collections",
            "restock_cost_estimate": (
                f"estimated: units needed for {horizon_days} days of sales at {demand}x the "
                "observed rate (the operator's expectation) plus 20%, at "
                f"{int(COST_RATIO * 100)}% of selling price (purchase costs are not recorded)"),
            "eligible_limit": (
                f"estimated: {LIMIT_MULTIPLE}x the last 30 days of verified collections; "
                "illustrative, not Paytm's underwriting"),
        },
    }
