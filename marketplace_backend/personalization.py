"""'Based on what you explored' and the welcome-back card (Q13, Q14).

Memory says what a shopper is interested in; the catalogue says what exists. So the
brief only steers *which* rows the catalogue query returns and in what order —
every card here is an active variant read now, priced now and in stock now, and
every reason chip is computed from the brief, not written by a model.
"""

from __future__ import annotations

from typing import Any, Protocol

from .store import Store

PRICE_BAND_SLACK = 0.3


class CatalogReads(Protocol):
    def sellable(self, variant_id: str) -> int: ...

    def current_price(self, variant_id: str) -> int: ...


class Personalizer:
    def __init__(self, store: Store, port: CatalogReads) -> None:
        self.store, self.port = store, port

    def suggestions(self, brief: dict, limit: int = 8) -> dict:
        explored = brief.get("explored") or []
        if not explored and not brief.get("top_categories") and not brief.get("liked_brands"):
            return {"items": [], "basis": "no_history"}

        exclude = set(brief.get("purchased_variant_ids") or [])
        exclude |= {v["variant_id"] for v in explored}
        explored_products = {v["product_id"] for v in explored}
        avoided = {b.lower() for b in brief.get("avoided_brands") or []}
        rejected = set(brief.get("rejected") or [])
        liked = [b.lower() for b in brief.get("liked_brands") or []]
        categories = list(brief.get("top_categories") or [])
        anchor_by_category: dict[str, dict] = {}
        for item in explored:
            if item.get("category"):
                anchor_by_category.setdefault(item["category"], item)
                if item["category"] not in categories:
                    categories.append(item["category"])
        band = brief.get("price_band_minor")

        rows = self.store.rows(
            "SELECT v.id AS variant_id, v.title AS variant_title, p.id AS product_id, "
            "p.title, p.brand, c.name AS category FROM catalog_variants v "
            "JOIN catalog_products p ON p.id=v.product_id "
            "LEFT JOIN catalog_categories c ON c.id=p.category_id "
            "WHERE v.status='active' AND p.status='active' ORDER BY v.id")

        scored: list[tuple[float, dict]] = []
        seen_products: set[str] = set()
        for row in rows:
            if row["variant_id"] in exclude or row["brand"].lower() in avoided:
                continue
            if row["title"].lower() in rejected or row["product_id"] in explored_products:
                continue
            score, reason = 0.0, None
            if row["category"] in categories:
                score += 3.0 - categories.index(row["category"]) * 0.5
                anchor = anchor_by_category.get(row["category"])
                reason = (f"Because you explored {anchor['title']}" if anchor
                          else f"You've been browsing {row['category']}")
            if row["brand"].lower() in liked:
                score += 2.0 - liked.index(row["brand"].lower()) * 0.4
                reason = reason or f"From {row['brand']}, a brand you keep coming back to"
            if score <= 0:
                continue
            sellable = self.port.sellable(row["variant_id"])
            if sellable <= 0:
                continue
            price = self.port.current_price(row["variant_id"])
            if band:
                low, high = band[0] * (1 - PRICE_BAND_SLACK), band[1] * (1 + PRICE_BAND_SLACK)
                if low <= price <= high:
                    score += 1.0
                    if reason is None:
                        reason = "Fits the price range you've been looking at"
            scored.append((score, {
                "variant_id": row["variant_id"], "product_id": row["product_id"],
                "title": row["variant_title"] or row["title"], "brand": row["brand"],
                "category": row["category"], "price_minor": price, "in_stock": True,
                "reason": reason,
            }))

        items = []
        for _, item in sorted(scored, key=lambda s: (-s[0], s[1]["variant_id"])):
            # One card per product: variants of the same thing are not a recommendation.
            if item["product_id"] in seen_products:
                continue
            seen_products.add(item["product_id"])
            items.append(item)
            if len(items) >= limit:
                break
        return {"items": items, "basis": "memory"}

    def welcome(self, brief: dict, cart: Any = None) -> dict | None:
        """The one card shown on return. Live price and stock for everything on it;
        nothing when there is nothing worth saying."""
        still_interested = []
        for item in (brief.get("explored") or [])[:2]:
            sellable = self.port.sellable(item["variant_id"])
            still_interested.append({
                "variant_id": item["variant_id"], "title": item["title"], "brand": item["brand"],
                "price_minor": self.port.current_price(item["variant_id"]),
                "in_stock": sellable > 0,
            })
        cart_lines = 0
        if isinstance(cart, dict):
            cart_lines = len(cart.get("lines") or cart.get("items") or [])
        devices = [f["value"] for f in brief.get("facts", []) if f["type"] == "device_owned"]
        picks = self.suggestions(brief, limit=3)["items"]
        if not (still_interested or cart_lines or picks):
            return None
        return {
            "cart_line_count": cart_lines,
            "still_interested": still_interested,
            "picks": picks,
            "devices": devices,
        }
