"""The `MerchantPort` adapter over the normalized commerce core.

This is what retires the Phase 5 carry-over: the merchant surfaces read
`catalog_products`, `catalog_variants`, `variant_prices`, `inventory_levels` and
`commerce_events` — the same tables shopping reads — instead of the flat
`products` table shopping stopped touching. One catalogue, so a price the
operator sees is the price a shopper is charged, and a stock figure on the
alerts card is the figure that refuses a checkout.

Every read returns figures with their formula and operands attached, because the
acceptance criterion for this phase is lineage, not plausibility. Metrics come
from `MetricsRepository`, which recomputes from the event log rather than
carrying a running total (ADR 0018); estimates come from
`merchant_estimates`, which does arithmetic over those observed inputs and
nothing else (ADR 0017).

Reads are scoped to `seeded` origin by default. The store's trading history is
the seeded ninety days; a live demo purchase is `live_app` and is deliberately
not folded into it, and every payload names the origins it covered so the
distinction is visible rather than assumed (ADR 0032).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from marketplace_backend.evidence import Correlation, EvidenceLedger
from marketplace_backend.merchant_changes import (
    MerchantChangeRepository,
    PolicyViolation,
)
from marketplace_backend import merchant_finance
from marketplace_backend.cart_recovery import active_policy
from marketplace_backend.merchant_changes import POLICY_BOUNDS
from marketplace_backend.metrics import MetricsRepository
from marketplace_backend.store import Store
from marketplace_backend.timeutil import now as iso_now

from .config import MerchantAgentConfig
from .merchant_estimates import (
    TARGET_COVER_DAYS,
    days_of_cover,
    price_change_ratio,
    restock_quantity,
    stockout_exposure,
)
from .merchant_ports import MerchantPort
from .merchant_types import (
    OBSERVED,
    BusinessSnapshot,
    CampaignPerformance,
    Claim,
    InventoryAlert,
    Listing,
    ListingDetails,
    ListingVariant,
    MerchantSessionContext,
    MetricPoint,
    MetricSeries,
    PriceHistoryEntry,
    PricingContext,
    StagedChange,
    UnmetDemandSignal,
)
from .outcomes import BusinessRefusal, Unavailable

# The metrics the contract's `query_metrics` enum offers, and how each is derived.
# A metric with no derivation is absent from this table and answers `unavailable`
# rather than zero: a figure the event log cannot support is not a figure.
SERIES_METRICS = ("revenue", "orders", "units", "conversion", "refund_rate", "cart_abandonment")

DEFAULT_ORIGINS = ("seeded",)

# Alerts fire on cover, not on a stored reorder point: the commerce core has no
# reorder-point column, and inventing one would be a number with no source.
ALERT_COVER_DAYS = 10

_LISTING_SELECT = """
SELECT p.id AS product_id, p.title, p.brand, p.status, p.origin, p.description,
       c.name AS category
FROM catalog_products p
LEFT JOIN catalog_categories c ON c.id = p.category_id
"""


class CoreMerchantPort(MerchantPort):
    def __init__(
        self,
        store: Store,
        *,
        changes: MerchantChangeRepository | None = None,
        metrics: MetricsRepository | None = None,
        config: MerchantAgentConfig | None = None,
        origins: tuple[str, ...] = DEFAULT_ORIGINS,
    ) -> None:
        self.store = store
        self.config = config or MerchantAgentConfig()
        self.metrics = metrics or MetricsRepository(store)
        self.changes = changes or MerchantChangeRepository(store, EvidenceLedger(store))
        self.origins = origins

    # -- snapshot --------------------------------------------------------------

    async def get_business_snapshot(
        self, session: MerchantSessionContext, window_days: int = 7
    ) -> BusinessSnapshot:
        window = _window(window_days)
        claims = [
            _claim(self.metrics.revenue(origin=self.origins, days=window)),
            _claim(self.metrics.orders(origin=self.origins, days=window)),
            _claim(self.metrics.average_order_value(origin=self.origins, days=window)),
            _claim(self.metrics.conversion(origin=self.origins, days=window)),
            # Attribution is computed over all recorded history, not over the window:
            # it joins order lines to the recommendations behind them, and a
            # recommendation has no window of its own. Left unsaid it reads as part of
            # the window beside it, which on a short window can make it look larger
            # than total revenue. So it says what it covers (ADR 0019).
            _all_time(_claim(self.metrics.agent_assisted_revenue(origin=self.origins))),
        ]
        # Movement is this window against the one before it, computed the same way
        # both times. It is a difference between two observed figures, so it is
        # observed too — and it is a difference, never a cause.
        previous = self._previous_window(window)
        movements = [
            _movement(claim, previous.get(claim.key), window)
            for claim in claims
            if claim.key in previous
        ]
        return BusinessSnapshot(
            window_days=window,
            origins=list(self.origins),
            claims=claims,
            comparison_window_days=window,
            movements=[m for m in movements if m is not None],
            limitations=[
                f"Covers {', '.join(self.origins)} activity only; other origins are excluded "
                "from every figure above.",
                "Traffic and visit-to-purchase conversion are not connected, so the "
                "conversion figure is checkout conversion: orders paid over orders created.",
            ],
        )

    def _previous_window(self, window_days: int) -> dict[str, float | int | None]:
        """The same figures over the window before this one, for movement only."""
        double = self.metrics.revenue(origin=self.origins, days=window_days * 2)
        double_orders = self.metrics.orders(origin=self.origins, days=window_days * 2)
        recent_revenue = self.metrics.revenue(origin=self.origins, days=window_days)
        recent_orders = self.metrics.orders(origin=self.origins, days=window_days)
        return {
            "net_revenue_minor": (double.value or 0) - (recent_revenue.value or 0),
            "paid_orders": (double_orders.value or 0) - (recent_orders.value or 0),
        }

    # -- metrics ---------------------------------------------------------------

    async def query_metrics(
        self,
        session: MerchantSessionContext,
        metric: str,
        window_days: int = 30,
        group_by: str | None = None,
    ) -> MetricSeries:
        window = _window(window_days)
        if metric not in SERIES_METRICS:
            raise Unavailable(
                f"{metric!r} is not a metric Cartisan derives. Available: "
                f"{', '.join(SERIES_METRICS)}."
            )
        if metric in {"revenue", "orders", "units"}:
            return self._quantity_series(metric, window, group_by)
        if metric == "conversion":
            observed = self.metrics.conversion(origin=self.origins, days=window)
            return self._scalar_series(metric, window, observed, "ratio")
        if metric == "refund_rate":
            return self._refund_rate(window)
        return self._cart_abandonment(window)

    def _quantity_series(self, metric: str, window: int, group_by: str | None) -> MetricSeries:
        if group_by in {None, "day"}:
            return self._daily_series(metric, window)
        return self._grouped_series(metric, window, group_by)

    def _daily_series(self, metric: str, window: int) -> MetricSeries:
        placeholders = ",".join("?" for _ in self.origins)
        day = self.metrics._day_expr("occurred_at")
        cutoff = _cutoff(window)
        if metric == "units":
            rows = self.store.rows(
                f"SELECT {self.metrics._day_expr('e.occurred_at')} AS day, "
                "COALESCE(SUM(l.quantity),0) AS value, COUNT(DISTINCT e.subject_id) AS orders "
                "FROM commerce_events e JOIN commerce_order_lines l ON l.order_id = e.subject_id "
                f"WHERE e.event_type='order_paid' AND e.origin IN ({placeholders}) "
                "AND e.occurred_at >= ? "
                f"GROUP BY {self.metrics._day_expr('e.occurred_at')} ORDER BY day",
                (*self.origins, cutoff))
            unit, basis = "units", "SUM(order line quantity) over paid orders, by day"
        else:
            value = "COALESCE(SUM(amount_minor),0)" if metric == "revenue" else "COUNT(*)"
            rows = self.store.rows(
                f"SELECT {day} AS day, {value} AS value, COUNT(*) AS orders "
                f"FROM commerce_events WHERE event_type='order_paid' "
                f"AND origin IN ({placeholders}) AND occurred_at >= ? "
                f"GROUP BY {day} ORDER BY day",
                (*self.origins, cutoff))
            unit = "INR paise" if metric == "revenue" else "orders"
            basis = f"{value} over commerce_events where event_type='order_paid', by day"
        points = [
            MetricPoint(date=str(row["day"]), value=_number(row["value"]),
                        orders=int(row["orders"]))
            for row in rows
        ]
        return MetricSeries(
            metric=metric, window_days=window, group_by="day", unit=unit,
            origins=list(self.origins), points=points,
            total=sum(point.value for point in points), basis=basis,
            limitations=[
                "Gross of refunds: a refund is recorded on the day it completed, not "
                "against the day of the sale.",
                f"Days with no paid order are absent rather than zero; the window is "
                f"{window} days and {len(points)} of them carry a point.",
            ])

    def _grouped_series(self, metric: str, window: int, group_by: str) -> MetricSeries:
        placeholders = ",".join("?" for _ in self.origins)
        columns = {
            "category": ("COALESCE(c.name,'uncategorised')", None),
            "brand": ("p.brand", None),
            "origin": ("o.origin", None),
            "product": ("p.title", "p.id"),
            "variant": ("p.title || ' — ' || v.title", "v.id"),
        }
        if group_by not in columns:
            raise Unavailable(
                f"{group_by!r} is not a breakdown Cartisan carries. Available: "
                "day, category, brand, origin, product, variant."
            )
        column, id_column = columns[group_by]
        select_id = f", {id_column} AS bucket_id" if id_column else ", NULL AS bucket_id"
        group_id = f", {id_column}" if id_column else ""
        value = {"revenue": "COALESCE(SUM(l.amount_minor),0)",
                 "units": "COALESCE(SUM(l.quantity),0)",
                 "orders": "COUNT(DISTINCT o.id)"}[metric]
        rows = self.store.rows(
            f"SELECT {column} AS bucket{select_id}, {value} AS value, "
            "COUNT(DISTINCT o.id) AS orders, COALESCE(SUM(l.quantity),0) AS units "
            "FROM commerce_orders o "
            "JOIN commerce_order_lines l ON l.order_id = o.id "
            "JOIN catalog_variants v ON v.id = l.variant_id "
            "JOIN catalog_products p ON p.id = v.product_id "
            "LEFT JOIN catalog_categories c ON c.id = p.category_id "
            f"WHERE o.status='paid' AND o.origin IN ({placeholders}) AND o.created_at >= ? "
            f"GROUP BY {column}{group_id} ORDER BY value DESC",
            (*self.origins, _cutoff(window)))
        unit = {"revenue": "INR paise", "units": "units", "orders": "orders"}[metric]
        points = [
            MetricPoint(date=str(row["bucket"]), value=_number(row["value"]),
                        orders=int(row["orders"]), units=int(row["units"]),
                        bucket_id=row.get("bucket_id"))
            for row in rows
        ]
        return MetricSeries(
            metric=metric, window_days=window, group_by=group_by, unit=unit,
            origins=list(self.origins), points=points,
            total=sum(point.value for point in points),
            basis=f"{value} over paid order lines joined to the catalogue, grouped by {group_by}",
            limitations=[
                "Grouped on the order line, so an order spanning two categories "
                "contributes to both and its order count is counted once in each.",
                "Shipping, tax, and order-level discounts sit on the order, not the "
                "line, so the grouped totals do not add up to order revenue.",
            ])

    def _scalar_series(self, metric: str, window: int, observed: Any, unit: str) -> MetricSeries:
        """A metric that is one ratio over the window rather than a series."""
        return MetricSeries(
            metric=metric, window_days=window, unit=unit, origins=list(self.origins),
            points=[], total=observed.value, basis=observed.basis,
            limitations=list(observed.limitations) + [
                "Reported for the window as a whole; a daily series would be too sparse "
                "to read at this order volume.",
            ])

    def _refund_rate(self, window: int) -> MetricSeries:
        placeholders = ",".join("?" for _ in self.origins)
        cutoff = _cutoff(window)
        paid = int(self.store.rows(
            f"SELECT COUNT(*) AS n FROM commerce_events WHERE event_type='order_paid' "
            f"AND origin IN ({placeholders}) AND occurred_at >= ?",
            (*self.origins, cutoff))[0]["n"])
        refunded = int(self.store.rows(
            f"SELECT COUNT(*) AS n FROM commerce_events WHERE event_type='order_refunded' "
            f"AND origin IN ({placeholders}) AND occurred_at >= ?",
            (*self.origins, cutoff))[0]["n"])
        return MetricSeries(
            metric="refund_rate", window_days=window, unit="ratio", origins=list(self.origins),
            total=round(refunded / paid, 4) if paid else None,
            basis="COUNT(order_refunded) / COUNT(order_paid) over commerce_events",
            limitations=[
                "Numerator and denominator are counted in the same window, so a refund of "
                "an order paid before the window still lands in it. This is a period rate, "
                "not a cohort rate.",
                "No paid orders in this window, so no rate exists." if not paid else
                f"Computed from {refunded} refunds against {paid} paid orders.",
            ])

    def _cart_abandonment(self, window: int) -> MetricSeries:
        placeholders = ",".join("?" for _ in self.origins)
        cutoff = _cutoff(window)
        created = int(self.store.rows(
            f"SELECT COUNT(*) AS n FROM commerce_events WHERE event_type='order_created' "
            f"AND origin IN ({placeholders}) AND occurred_at >= ?",
            (*self.origins, cutoff))[0]["n"])
        paid = int(self.store.rows(
            f"SELECT COUNT(*) AS n FROM commerce_events WHERE event_type='order_paid' "
            f"AND origin IN ({placeholders}) AND occurred_at >= ?",
            (*self.origins, cutoff))[0]["n"])
        return MetricSeries(
            metric="cart_abandonment", window_days=window, unit="ratio",
            origins=list(self.origins),
            total=round((created - paid) / created, 4) if created else None,
            basis="(COUNT(order_created) - COUNT(order_paid)) / COUNT(order_created)",
            limitations=[
                "This is checkout abandonment: an order was created and never paid. Carts "
                "abandoned before checkout never became orders and are not counted, so the "
                "true browse-to-buy drop-off is larger than this figure.",
                "An order created near the end of the window may still be paid after it.",
            ])

    # -- catalogue -------------------------------------------------------------

    async def search_listings(
        self, session: MerchantSessionContext, query: str, limit: int = 10
    ) -> list[Listing]:
        terms = [token for token in _tokens(query) if len(token) > 2][:6]
        clauses: list[str] = []
        params: list[Any] = []
        if terms:
            ors = " OR ".join(
                "(lower(p.title) LIKE ? OR lower(p.brand) LIKE ? OR lower(c.name) LIKE ? "
                "OR lower(p.description) LIKE ?)" for _ in terms)
            clauses.append(f"({ors})")
            for term in terms:
                params += [f"%{term}%"] * 4
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.store.rows(
            f"{_LISTING_SELECT}{where} ORDER BY p.id LIMIT ?", (*params, max(1, limit) * 3))
        listings = [self._listing(row) for row in rows]
        if terms:
            listings.sort(
                key=lambda listing: -sum(
                    term in f"{listing.title} {listing.brand} {listing.category or ''}".lower()
                    for term in terms))
        return listings[:limit]

    async def get_listing(
        self, session: MerchantSessionContext, product_id: str
    ) -> ListingDetails | None:
        rows = self.store.rows(f"{_LISTING_SELECT} WHERE p.id = ?", (product_id,))
        if not rows:
            return None
        row = rows[0]
        base = self._listing(row)
        variants = self._variants(product_id)
        window = 30
        sold = self._sold_by_product(product_id, window)
        return ListingDetails(
            **base.model_dump(),
            description=row.get("description") or "",
            variants=variants,
            units_sold=sold["units"],
            revenue_minor=sold["revenue_minor"],
            window_days=window,
        )

    async def get_inventory_alerts(
        self, session: MerchantSessionContext, limit: int = 10
    ) -> list[InventoryAlert]:
        window = 30
        placeholders = ",".join("?" for _ in self.origins)
        # Ranked by cover, not by raw stock. Ordering on `sellable` alone reads the
        # wrong quantity: it puts every dead item with two units on the shelf ahead
        # of a fast mover with sixty, and a fast mover past the cut-off is never even
        # read, so the alert can come back empty while a real stockout is days away.
        # Cover is the criterion the loop below tests, so it is the criterion the
        # candidates are ordered by. Variants with no sales in the window are excluded
        # here for the same reason the loop skips them: no rate, so no cover to be
        # below — and excluding them in SQL is what keeps them from crowding out the
        # variants that do have one.
        rows = self.store.rows(
            "SELECT l.variant_id AS variant_id, v.product_id AS product_id, v.title AS title, "
            "SUM(l.on_hand) AS on_hand, SUM(l.reserved) AS reserved, "
            "SUM(l.on_hand - l.reserved) AS sellable, MIN(l.location_id) AS location_id, "
            "s.units AS units "
            "FROM inventory_levels l JOIN catalog_variants v ON v.id = l.variant_id "
            "JOIN catalog_products p ON p.id = v.product_id "
            "JOIN (SELECT ol.variant_id AS variant_id, SUM(ol.quantity) AS units "
            "FROM commerce_order_lines ol JOIN commerce_orders o ON o.id = ol.order_id "
            f"WHERE o.status = 'paid' AND o.origin IN ({placeholders}) AND o.created_at >= ? "
            "GROUP BY ol.variant_id) s ON s.variant_id = l.variant_id "
            "WHERE v.status = 'active' AND p.status = 'active' AND s.units > 0 "
            "GROUP BY l.variant_id, v.product_id, v.title, s.units "
            "ORDER BY (SUM(l.on_hand - l.reserved) * 1.0 * ?) / s.units, l.variant_id LIMIT ?",
            (*self.origins, _cutoff(window), window, max(1, limit) * 2))
        alerts: list[InventoryAlert] = []
        for row in rows:
            sellable = int(row["sellable"])
            units = int(row["units"])
            cover = days_of_cover(
                variant_id=row["variant_id"], sellable=sellable, units_sold=units,
                window_days=window)
            # An item with no sales rate has no cover to be below; it is slow stock,
            # not a stockout risk, and saying otherwise would be a claim with no basis.
            # The rows arrive in ascending cover, so the first one over the threshold
            # means every one after it is too.
            if cover.value is None:
                continue
            if cover.value > ALERT_COVER_DAYS:
                break
            alerts.append(InventoryAlert(
                variant_id=row["variant_id"], product_id=row["product_id"], title=row["title"],
                sellable=sellable, on_hand=int(row["on_hand"]), reserved=int(row["reserved"]),
                location_id=row["location_id"], units_sold=units, window_days=window,
                days_of_cover=cover))
            if len(alerts) >= limit:
                break
        return alerts

    async def get_unmet_demand(
        self, session: MerchantSessionContext, window_days: int = 30, limit: int = 10
    ) -> list[UnmetDemandSignal]:
        window = _window(window_days)
        rows = self.store.rows(
            "SELECT subject_id AS query, COUNT(*) AS requests, "
            "COUNT(DISTINCT customer_id) AS unique_customers, "
            "MIN(occurred_at) AS first_seen, MAX(occurred_at) AS last_seen "
            "FROM commerce_events WHERE event_type='catalog_search_no_results' "
            "AND origin='live_app' AND occurred_at >= ? GROUP BY subject_id "
            "ORDER BY requests DESC, last_seen DESC LIMIT ?",
            (_cutoff(window), max(1, min(limit, 50))),
        )
        return [UnmetDemandSignal(
            query=str(row["query"]), requests=int(row["requests"]),
            unique_customers=int(row["unique_customers"]), first_seen=str(row["first_seen"]),
            last_seen=str(row["last_seen"]), window_days=window,
        ) for row in rows]

    async def get_pricing_context(
        self, session: MerchantSessionContext, variant_id: str
    ) -> PricingContext | None:
        rows = self.store.rows(
            "SELECT v.id, v.product_id, v.title FROM catalog_variants v WHERE v.id = ?",
            (variant_id,))
        if not rows:
            return None
        row = rows[0]
        current = self._price(variant_id)
        window = 30
        units = self._sold_by_variant(variant_id, window)
        ratio = self.config_max_change_ratio()
        history = [
            PriceHistoryEntry(
                amount_minor=int(entry["amount_minor"]), price_kind=entry["price_kind"],
                valid_from=str(entry["valid_from"]),
                valid_to=str(entry["valid_to"]) if entry["valid_to"] else None)
            for entry in self.store.rows(
                "SELECT amount_minor,compare_at_minor,price_kind,valid_from,valid_to "
                "FROM variant_prices WHERE variant_id = ? ORDER BY valid_from DESC LIMIT 8",
                (variant_id,))
        ]
        compare_at = self.store.rows(
            "SELECT compare_at_minor FROM variant_prices WHERE variant_id = ? "
            "AND compare_at_minor IS NOT NULL ORDER BY valid_from DESC LIMIT 1", (variant_id,))
        return PricingContext(
            variant_id=variant_id,
            product_id=row["product_id"],
            title=row["title"],
            current_price_minor=current,
            compare_at_minor=int(compare_at[0]["compare_at_minor"]) if compare_at else None,
            history=history,
            units_sold=units,
            revenue_minor=units * current,
            window_days=window,
            sellable=self._sellable(variant_id),
            max_change_ratio=ratio,
            floor_minor=int(round(current * (1 - ratio))),
            ceiling_minor=int(round(current * (1 + ratio))),
            limitations=[
                "Cost and margin are not recorded in Cartisan, so no floor here is a "
                "margin floor: the bounds below are policy limits on how far one change "
                "may move a price, and nothing more.",
                "Revenue over the window is units sold at the current price, not at the "
                "prices those units actually sold at.",
            ])

    def config_max_change_ratio(self) -> float:
        from marketplace_backend.merchant_changes import POLICY_BOUNDS

        return float(POLICY_BOUNDS["price_update"]["max_change_ratio"])

    async def get_campaign_performance(
        self,
        session: MerchantSessionContext,
        campaign_id: str | None = None,
        window_days: int = 30,
    ) -> list[CampaignPerformance]:
        clause, params = ("WHERE c.id = ?", (campaign_id,)) if campaign_id else ("", ())
        rows = self.store.rows(
            "SELECT c.id, c.name, c.channel, c.status, c.budget_minor, c.spend_minor, "
            "c.promotion_id, c.starts_at, c.ends_at, "
            "pr.code AS promotion_code, pr.description AS promotion_description "
            "FROM campaigns c LEFT JOIN promotions pr ON pr.id = c.promotion_id "
            f"{clause} ORDER BY c.id", params)
        results: list[CampaignPerformance] = []
        for row in rows:
            spend, budget = int(row["spend_minor"]), int(row["budget_minor"])
            claims = [
                Claim(key=f"campaign_spend:{row['id']}", value=spend, unit="INR paise",
                      claim_kind=OBSERVED, basis="campaigns.spend_minor as recorded",
                      inputs={"campaign_id": row["id"], "budget_minor": budget}),
                Claim(key=f"campaign_budget_used:{row['id']}",
                      value=round(spend / budget, 4) if budget else None, unit="ratio",
                      claim_kind=OBSERVED, basis="spend_minor / budget_minor",
                      inputs={"spend_minor": spend, "budget_minor": budget}),
            ]
            limitations = ["Spend and budget are observed."]
            if row["promotion_id"]:
                claims.extend(self._redemption_claims(row))
                limitations.extend([
                    "Attribution is by redemption, not by exposure: these are orders that "
                    "redeemed the campaign's promotion code while it ran. Cartisan records "
                    "no impression or click, so a customer who saw the campaign and bought "
                    "without the code is not counted, and one who found the code elsewhere "
                    "is.",
                    "Descriptive, not causal: it does not claim these orders would not have "
                    "happened without the campaign. Do not call it incremental revenue, "
                    "lift, or ROI.",
                    "Attributed revenue is the whole order total, so a basket that also "
                    "held items outside the promotion's scope is counted in full.",
                ])
            else:
                limitations.append(
                    "This campaign carries no promotion code, so nothing links an order to "
                    "it and attributed orders and revenue are NOT available. Any figure for "
                    "them would be invented; do not state one.")
            results.append(CampaignPerformance(
                campaign_id=row["id"], name=row["name"], channel=row["channel"],
                status=row["status"], budget_minor=budget, spend_minor=spend,
                promotion_code=row["promotion_code"],
                promotion_description=row["promotion_description"],
                claims=claims, limitations=limitations))
        return results

    def _redemption_claims(self, campaign: dict) -> list[Claim]:
        """Orders that redeemed this campaign's promotion inside its own window.

        The window is the campaign's, not an arbitrary trailing period: a redemption
        before the campaign started belongs to whatever else was running then.
        """
        window = "AND o.created_at >= ?" + (" AND o.created_at < ?" if campaign["ends_at"] else "")
        params: tuple = ((campaign["promotion_id"], campaign["starts_at"], campaign["ends_at"])
                         if campaign["ends_at"]
                         else (campaign["promotion_id"], campaign["starts_at"]))
        paid = self.store.rows(
            "SELECT COUNT(*) AS n, COALESCE(SUM(o.total_minor),0) AS revenue, "
            "COALESCE(SUM(o.discount_minor),0) AS discount FROM commerce_orders o "
            f"WHERE o.promotion_id = ? AND o.status = 'paid' {window}", params)[0]
        redeemed = int(self.store.rows(
            "SELECT COUNT(*) AS n FROM commerce_orders o "
            f"WHERE o.promotion_id = ? {window}", params)[0]["n"])
        orders, revenue = int(paid["n"]), int(paid["revenue"])
        window_note = (f"{campaign['starts_at']} to "
                       f"{campaign['ends_at'] or 'now (campaign still running)'}")
        shared = {"campaign_id": campaign["id"], "promotion_code": campaign["promotion_code"],
                  "window": window_note, "orders_redeeming_any_status": redeemed}
        return [
            Claim(key=f"campaign_attributed_orders:{campaign['id']}", value=orders, unit="orders",
                  claim_kind=OBSERVED,
                  basis="COUNT(paid commerce_orders whose promotion_id is this campaign's "
                        "promotion, created inside the campaign window)",
                  inputs=shared),
            Claim(key=f"campaign_attributed_revenue:{campaign['id']}", value=revenue,
                  unit="INR paise", claim_kind=OBSERVED,
                  basis="SUM(total_minor) over those same paid orders",
                  inputs=shared | {"attributed_orders": orders}),
            Claim(key=f"campaign_discount_given:{campaign['id']}", value=int(paid["discount"]),
                  unit="INR paise", claim_kind=OBSERVED,
                  basis="SUM(discount_minor) over those same paid orders — what the promotion "
                        "cost in forgone revenue, separate from campaign spend",
                  inputs=shared | {"spend_minor": int(campaign["spend_minor"])}),
        ]

    # -- staged changes --------------------------------------------------------

    async def get_pending_changes(
        self, session: MerchantSessionContext, limit: int = 10
    ) -> list[StagedChange]:
        return [_change(row) for row in self.changes.pending(limit=max(1, limit))]

    async def read_change(
        self, session: MerchantSessionContext, change_id: str
    ) -> StagedChange | None:
        try:
            return _change(self.changes.read(change_id))
        except ValueError:
            return None

    async def stage_change(
        self,
        session: MerchantSessionContext,
        *,
        kind: str,
        target_type: str,
        target_id: str | None,
        before: dict,
        after: dict,
        rationale: str,
    ) -> StagedChange:
        try:
            row = self.changes.stage(
                operator_id=session.operator_id, kind=kind, target_type=target_type,
                target_id=target_id, before=before, after=after, rationale=rationale,
                conversation_id=session.conversation_id,
                correlation=Correlation(
                    correlation_id=getattr(session, "correlation_id", None) or Correlation().correlation_id,
                    turn_id=getattr(session, "turn_id", None),
                    demo_run_id=session.demo_run_id))
        except PolicyViolation as exc:
            # A bound is a business refusal, not an outage: the model should propose
            # something inside the bound rather than retry the same call.
            raise BusinessRefusal(
                f"That change was not staged: {exc}. Nothing was recorded and nothing "
                "changed. Propose one that fits the bound, or tell the operator what the "
                "bound is.", gate="merchant_policy") from exc
        except ValueError as exc:
            raise BusinessRefusal(f"That change was not staged: {exc}") from exc
        return _change(row)

    # -- the current-state reader the host revalidates against -------------------

    # -- payments and financing -------------------------------------------------

    async def get_payment_health(self, session: MerchantSessionContext, window_days: int = 7) -> dict:
        return merchant_finance.payment_health(self.store, _window(window_days))

    async def get_restock_financing(self, session: MerchantSessionContext, horizon_days: int = 21,
                                    demand_multiplier: float = 1.0) -> dict:
        return merchant_finance.restock_financing(
            self.store, max(7, min(int(horizon_days or 21), 60)), demand_multiplier)

    async def get_recovery_policy(self, session: MerchantSessionContext) -> dict:
        return {"active": active_policy(self.store), "bounds": POLICY_BOUNDS["recovery_policy"]}

    def current_before(self, change: dict) -> dict:
        """The live `before` document for a staged change's target.

        `MerchantChangeRepository.apply` calls this at application time, so an
        approval of a proposal whose underlying record has since moved is refused
        instead of applied to a different starting point.
        """
        kind, target = change["kind"], change["target_id"]
        if kind == "price_update" and target:
            return {"amount_minor": self._price(target)}
        if kind == "inventory_action" and target:
            rows = self.store.rows(
                "SELECT COALESCE(SUM(on_hand),0) AS on_hand, COALESCE(SUM(reserved),0) AS reserved "
                "FROM inventory_levels WHERE variant_id = ?", (target,))
            return {"on_hand": int(rows[0]["on_hand"]), "reserved": int(rows[0]["reserved"])}
        if kind == "listing_update" and target:
            rows = self.store.rows(
                "SELECT title,description,status FROM catalog_products WHERE id = ?", (target,))
            if not rows:
                return {}
            live = rows[0]
            # Only the fields the proposal spoke about are compared, so an edit to the
            # description does not go stale because someone renamed the product.
            return {key: live[key] for key in change["before"] if key in live}
        if kind == "promotion" and target:
            rows = self.store.rows(
                "SELECT code,discount_kind,discount_value,min_subtotal_minor,status "
                "FROM promotions WHERE id = ?", (target,))
            return dict(rows[0]) if rows else {}
        if kind == "campaign" and target:
            rows = self.store.rows(
                "SELECT name,channel,budget_minor,spend_minor,status FROM campaigns WHERE id = ?",
                (target,))
            return dict(rows[0]) if rows else {}
        return dict(change["before"])

    # -- internals -------------------------------------------------------------

    def _listing(self, row: dict) -> Listing:
        aggregate = self.store.rows(
            "SELECT COUNT(*) AS variants FROM catalog_variants WHERE product_id = ?",
            (row["product_id"],))
        variants = self._variants(row["product_id"])
        prices = [variant.price_minor for variant in variants if variant.price_minor > 0]
        return Listing(
            product_id=row["product_id"], title=row["title"], brand=row["brand"],
            category=row.get("category"), status=row["status"],
            origin=row.get("origin") or "seeded",
            from_price_minor=min(prices) if prices else 0,
            total_sellable=sum(variant.sellable for variant in variants),
            variant_count=int(aggregate[0]["variants"]) if aggregate else len(variants))

    def _variants(self, product_id: str) -> list[ListingVariant]:
        variants: list[ListingVariant] = []
        for row in self.store.rows(
            "SELECT id,sku,title,options,status FROM catalog_variants WHERE product_id = ? "
            "ORDER BY id", (product_id,)):
            levels = self.store.rows(
                "SELECT COALESCE(SUM(on_hand),0) AS on_hand, COALESCE(SUM(reserved),0) AS reserved "
                "FROM inventory_levels WHERE variant_id = ?", (row["id"],))
            on_hand = int(levels[0]["on_hand"]) if levels else 0
            reserved = int(levels[0]["reserved"]) if levels else 0
            compare_at = self.store.rows(
                "SELECT compare_at_minor FROM variant_prices WHERE variant_id = ? "
                "AND compare_at_minor IS NOT NULL ORDER BY valid_from DESC LIMIT 1", (row["id"],))
            variants.append(ListingVariant(
                variant_id=row["id"], sku=row["sku"], title=row["title"],
                options=self.store.load(row["options"]) if row.get("options") else {},
                status=row["status"], price_minor=self._price(row["id"]),
                compare_at_minor=int(compare_at[0]["compare_at_minor"]) if compare_at else None,
                sellable=max(0, on_hand - reserved), on_hand=on_hand, reserved=reserved))
        return variants

    def _price(self, variant_id: str) -> int:
        rows = self.store.rows(
            "SELECT amount_minor FROM variant_prices WHERE variant_id = ? AND valid_from <= ? "
            "AND (valid_to IS NULL OR valid_to > ?) "
            "ORDER BY CASE price_kind WHEN 'promotional' THEN 0 ELSE 1 END, valid_from DESC "
            "LIMIT 1",
            (variant_id, iso_now(), iso_now()))
        return int(rows[0]["amount_minor"]) if rows else 0

    def _sellable(self, variant_id: str) -> int:
        rows = self.store.rows(
            "SELECT on_hand, reserved FROM inventory_levels WHERE variant_id = ?", (variant_id,))
        return sum(max(0, int(row["on_hand"]) - int(row["reserved"])) for row in rows)

    def _sold_by_variant(self, variant_id: str, window_days: int) -> int:
        placeholders = ",".join("?" for _ in self.origins)
        rows = self.store.rows(
            "SELECT COALESCE(SUM(l.quantity),0) AS units FROM commerce_order_lines l "
            "JOIN commerce_orders o ON o.id = l.order_id "
            f"WHERE l.variant_id = ? AND o.status='paid' AND o.origin IN ({placeholders}) "
            "AND o.created_at >= ?",
            (variant_id, *self.origins, _cutoff(window_days)))
        return int(rows[0]["units"]) if rows else 0

    def _sold_by_product(self, product_id: str, window_days: int) -> dict[str, int]:
        placeholders = ",".join("?" for _ in self.origins)
        rows = self.store.rows(
            "SELECT COALESCE(SUM(l.quantity),0) AS units, COALESCE(SUM(l.amount_minor),0) AS revenue "
            "FROM commerce_order_lines l JOIN commerce_orders o ON o.id = l.order_id "
            "JOIN catalog_variants v ON v.id = l.variant_id "
            f"WHERE v.product_id = ? AND o.status='paid' AND o.origin IN ({placeholders}) "
            "AND o.created_at >= ?",
            (product_id, *self.origins, _cutoff(window_days)))
        return {"units": int(rows[0]["units"]), "revenue_minor": int(rows[0]["revenue"])}


# -- module helpers -------------------------------------------------------------


def _window(days: Any) -> int:
    return max(1, min(int(days or 7), 90))


def _cutoff(window_days: int) -> str:
    return (datetime.now(UTC) - timedelta(days=window_days)).isoformat()


def _number(value: Any) -> float | int:
    number = float(value or 0)
    return int(number) if number.is_integer() else number


def _tokens(text: str) -> list[str]:
    return "".join(c.lower() if c.isalnum() else " " for c in text).split()


def _claim(metric: Any) -> Claim:
    """A `metrics.Metric` as a `Claim`. Every figure the repository computes is a
    measurement over the event log, so it is `observed` by construction; nothing in
    this direction can produce an estimate or a causal claim."""
    return Claim(
        key=metric.key, value=metric.value, unit=metric.unit, claim_kind=OBSERVED,
        basis=metric.basis, inputs=metric.inputs, limitations=list(metric.limitations))


def _all_time(claim: Claim) -> Claim:
    """A claim computed over all recorded history, marked as such so it is not read as
    belonging to the window it is displayed beside."""
    claim.inputs["window_days"] = None
    claim.inputs["covers"] = "all recorded history"
    claim.limitations = [
        "Covers all recorded history, not the window the other figures use, so it is "
        "not a share of them and the two must not be divided.",
        *claim.limitations,
    ]
    return claim


def _movement(current: Claim, previous: float | int | None, window: int) -> Claim | None:
    """This window against the one before it. A difference between two observed
    figures, reported as a difference — never as something the store did."""
    if previous is None or current.value is None:
        return None
    delta = current.value - previous
    return Claim(
        key=f"{current.key}:movement",
        value=round(delta / previous, 4) if previous else None,
        unit="ratio",
        claim_kind=OBSERVED,
        basis=f"(this {window}-day window - the {window} days before it) / the {window} days before it",
        inputs={"current": current.value, "previous": previous, "absolute_change": delta,
                "window_days": window},
        limitations=[
            "A change between two windows, not a cause: nothing here identifies why it "
            "moved, and no experiment supports attributing it to any action.",
        ])


def _change(row: dict) -> StagedChange:
    return StagedChange(
        change_id=row["id"], kind=row["kind"], target_type=row["target_type"],
        target_id=row["target_id"], status=row["status"], before=row["before"],
        after=row["after"], rationale=row["rationale"], created_at=str(row["created_at"]),
        decided_at=str(row["decided_at"]) if row.get("decided_at") else None,
        applied_at=str(row["applied_at"]) if row.get("applied_at") else None)


__all__ = ["CoreMerchantPort", "TARGET_COVER_DAYS", "price_change_ratio", "restock_quantity",
           "stockout_exposure"]
