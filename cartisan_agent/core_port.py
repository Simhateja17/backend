"""The `CommercePort` adapter over the normalized commerce core.

Reads are real: they go to `catalog_variants`, `variant_specs`, `variant_requirements`,
`variant_prices`, `inventory_levels`, and `commerce_orders`, so catalogue grounding and
structured compatibility are proven against the Phase 3 merchant rather than a stub.
Writes are deliberately thin — the cart lines and the priced preview, and nothing
beyond them — because reservation, internal orders, and the Razorpay handoff are
Phase 5's, behind the customer's confirmation (ADR 0012, ADR 0013).

The cart is variant-keyed. The storefront UI still reads the legacy flat `products`
table, so until Phase 5 migrates it the browser cart panel and this cart can hold
different lines for the same customer; that divergence is the recorded carry-over, not
a second cart authority — both write the same `customer_carts` row and the same
`state_version`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from marketplace_backend.carts import ConflictError, IdempotencyLedger
from marketplace_backend.checkout import CheckoutRepository
from marketplace_backend.evidence import CommerceEventLog
from marketplace_backend.store import Store
from marketplace_backend.timeutil import now as iso_now

from .config import CartisanAgentConfig
from .outcomes import Conflict, Unavailable
from .ports import CommercePort
from .types import (
    Cart,
    CartLine,
    CompatibilityFinding,
    CompatibilityVerdict,
    FulfillmentOption,
    Order,
    OrderLine,
    Policy,
    Preferences,
    SearchFilters,
    SessionContext,
    StagedCheckout,
    Variant,
    VariantDetails,
    inr,
)

_STOPWORDS = frozenset(
    {
        "a", "an", "and", "the", "for", "with", "want", "need", "show", "some", "under",
        "this", "that", "please", "can", "you", "could", "would", "me", "my", "of", "to",
        "in", "on", "is", "are", "it", "i", "looking", "something", "good", "best",
        "product", "products", "catalog", "catalogue", "range", "selection", "kind",
        "kinds", "type", "types", "everything", "carry", "sell", "stock", "have",
    }
)

_VARIANT_SELECT = """
SELECT v.id AS variant_id, v.product_id, v.sku, v.title, v.options,
       p.title AS product_title, p.brand, p.origin, c.name AS category
FROM catalog_variants v
JOIN catalog_products p ON p.id = v.product_id
LEFT JOIN catalog_categories c ON c.id = p.category_id
"""


class CoreCommercePort(CommercePort):
    def __init__(
        self,
        store: Store,
        *,
        checkout: CheckoutRepository,
        config: CartisanAgentConfig | None = None,
    ) -> None:
        self.store = store
        self.checkout = checkout
        self.config = config or CartisanAgentConfig()
        self.idempotency = IdempotencyLedger(store)
        self.events = CommerceEventLog(store)
        # The host may attach cart recovery: given a customer and a subtotal it returns
        # the discount and promotion their live, approved offer grants. Computed from
        # stored terms at staging — no client or model supplies a discount.
        self.stage_discount: Callable[[str, int], tuple[int, str | None]] | None = None

    # -- catalogue ------------------------------------------------------------

    async def search_products(
        self,
        session: SessionContext,
        query: str,
        filters: SearchFilters | None = None,
        limit: int = 8,
    ) -> list[Variant]:
        filters = filters or SearchFilters()
        terms = [t for t in _tokens(query) if t not in _STOPWORDS][:6]
        clauses: list[str] = ["v.status = 'active'", "p.status = 'active'"]
        params: list[Any] = []
        if terms:
            # Any term may match, and the count of matching terms is the ranking. An
            # AND across every token would answer "usb-c charger" with nothing, and an
            # empty result reads to the model as proof the catalogue lacks the item.
            ors = " OR ".join(
                "(" + " OR ".join(
                    "(lower(v.title) LIKE ? OR lower(p.title) LIKE ? OR lower(p.brand) LIKE ? "
                    "OR lower(p.description) LIKE ?)"
                    for _ in _term_forms(term)
                ) + ")"
                for term in terms
            )
            clauses.append(f"({ors})")
            for term in terms:
                for form in _term_forms(term):
                    params += [f"%{form}%"] * 4
        if filters.category:
            category_ids = self._resolve_category_ids(filters.category)
            if category_ids:
                clauses.append(f"p.category_id IN ({','.join('?' for _ in category_ids)})")
                params += category_ids
            # A label with no match in the real taxonomy (a model guessing "Earbuds"
            # when the category is "Personal Audio") drops the constraint rather than
            # filtering to nothing: an exact-match miss here used to manufacture an
            # empty result the model then read as proof the catalogue lacks the item,
            # even though the free-text query terms already found it.
        if filters.brand:
            clauses.append("lower(p.brand) = ?")
            params.append(filters.brand.lower())
        rows = self.store.rows(
            f"{_VARIANT_SELECT} WHERE {' AND '.join(clauses)} ORDER BY v.id LIMIT ?",
            (*params, max(limit, 1) * 6),
        )
        scored: list[tuple[int, Variant]] = []
        for row in rows:
            variant = self._variant(row)
            if filters.in_stock_only and not variant.in_stock:
                continue
            if filters.min_price_minor is not None and variant.price_minor < filters.min_price_minor:
                continue
            if filters.max_price_minor is not None and variant.price_minor > filters.max_price_minor:
                continue
            haystack = " ".join(
                str(row.get(key) or "")
                for key in ("title", "product_title", "brand", "category")
            ).lower()
            score = sum(_matches_search_term(term, haystack) for term in terms)
            if terms and score == 0:
                continue
            scored.append((score, variant))
        if filters.sort == "price_asc":
            scored.sort(key=lambda pair: pair[1].price_minor)
        elif filters.sort == "price_desc":
            scored.sort(key=lambda pair: -pair[1].price_minor)
        else:
            scored.sort(key=lambda pair: (-pair[0], pair[1].variant_id))
        return [variant for _, variant in scored[:limit]]

    async def record_unmet_demand(
        self, session: SessionContext, query: str, filters: SearchFilters | None = None
    ) -> None:
        normalized = " ".join(_tokens(query))[:160]
        if not normalized:
            return
        applied = filters or SearchFilters()
        self.events.append(
            event_type="catalog_search_no_results", subject_type="catalog_query",
            subject_id=normalized, customer_id=session.customer_id, origin="live_app",
            correlation=session.correlation(),
            detail={"query": normalized, "filters": applied.model_dump(exclude_none=True)},
        )

    def _resolve_category_ids(self, category: str) -> list[str]:
        """Match a `filters.category` label against the real taxonomy by word
        overlap rather than exact string equality.

        A tool caller sees category names nowhere but in prior search results, so it
        sometimes guesses one ("Earbuds") that isn't in the taxonomy at all (the real
        category is "Personal Audio"). Word overlap still resolves an exact name and
        a plausible partial ("Audio" against "Personal Audio", "Home Audio"), and
        returns every category that shares a word rather than picking one arbitrarily.
        """
        wanted = set(_tokens(category))
        if not wanted:
            return []
        rows = self.store.rows("SELECT id, name FROM catalog_categories")
        return [row["id"] for row in rows if wanted & set(_tokens(row["name"]))]

    async def get_product_details(
        self, session: SessionContext, variant_id: str
    ) -> VariantDetails | None:
        rows = self.store.rows(f"{_VARIANT_SELECT} WHERE v.id = ?", (variant_id,))
        if not rows:
            return None
        row = rows[0]
        base = self._variant(row)
        description = self.store.rows(
            "SELECT description FROM catalog_products WHERE id = ?", (row["product_id"],)
        )
        siblings = [
            self._variant(sibling)
            for sibling in self.store.rows(
                f"{_VARIANT_SELECT} WHERE v.product_id = ? AND v.id <> ? AND v.status = 'active' "
                "ORDER BY v.id",
                (row["product_id"], variant_id),
            )
        ]
        return VariantDetails(
            **base.model_dump(),
            description=description[0]["description"] if description else "",
            capabilities=self._capabilities(variant_id),
            requirements=[
                requirement["explanation"]
                for requirement in self.store.rows(
                    "SELECT explanation FROM variant_requirements WHERE variant_id = ? "
                    "ORDER BY id",
                    (variant_id,),
                )
            ],
            siblings=siblings,
        )

    async def check_compatibility(
        self, session: SessionContext, base_variant_id: str, candidate_variant_id: str
    ) -> CompatibilityVerdict:
        for variant_id in (base_variant_id, candidate_variant_id):
            if not self.store.rows("SELECT id FROM catalog_variants WHERE id = ?", (variant_id,)):
                raise Unavailable(
                    f"variant_id {variant_id} is not in the catalogue, so compatibility "
                    "cannot be decided. Look the item up first."
                )
        # The base states what it needs; the candidate states what it offers. Both
        # directions are evaluated, because a requirement may sit on either item.
        findings = [
            *self._evaluate(base_variant_id, candidate_variant_id),
            *self._evaluate(candidate_variant_id, base_variant_id),
        ]
        blocking = [f for f in findings if not f.satisfied and f.severity == "blocking"]
        return CompatibilityVerdict(
            base_variant_id=base_variant_id,
            candidate_variant_id=candidate_variant_id,
            compatible=not blocking,
            findings=findings,
        )

    # -- cart -----------------------------------------------------------------

    async def get_cart(self, session: SessionContext) -> Cart:
        return self._render(self._active_cart_id(session.customer_id))

    async def add_to_cart(
        self,
        session: SessionContext,
        variant_id: str,
        quantity: int,
        *,
        expected_state_version: int | None = None,
        idempotency_key: str | None = None,
    ) -> Cart:
        if not self._sellable(variant_id):
            raise Unavailable(
                f"variant_id {variant_id} has no sellable stock, so nothing was added."
            )
        cart_id = self._active_cart_id(session.customer_id)
        request = {"variant_id": variant_id, "quantity": quantity,
                   "expected_state_version": expected_state_version}

        def effect() -> dict:
            cap = self.config.max_quantity_per_item
            with self.store.transaction() as tx:
                self._guard_version(tx, cart_id, expected_state_version)
                current = self._line_quantity(tx, cart_id, variant_id)
                if current == 0 and self._line_count(tx, cart_id) >= self.config.max_cart_lines:
                    raise Unavailable("The cart is full.")
                target = min(current + max(1, quantity), cap)
                self._write_line(tx, cart_id, variant_id, target)
                self._bump(tx, cart_id)
            return self._render(cart_id).model_dump()

        result = self.idempotency.run(
            principal_id=session.customer_id, operation="add_to_cart", key=idempotency_key,
            request=request, effect=effect)
        return Cart.model_validate(result)

    async def update_cart_item(
        self,
        session: SessionContext,
        variant_id: str,
        quantity: int,
        *,
        expected_state_version: int | None = None,
        idempotency_key: str | None = None,
    ) -> Cart:
        cart_id = self._active_cart_id(session.customer_id)
        request = {"variant_id": variant_id, "quantity": quantity,
                   "expected_state_version": expected_state_version}

        def effect() -> dict:
            with self.store.transaction() as tx:
                self._guard_version(tx, cart_id, expected_state_version)
                if self._line_quantity(tx, cart_id, variant_id) == 0:
                    raise Unavailable(f"The cart has no line for variant_id {variant_id}.")
                self._write_line(
                    tx, cart_id, variant_id,
                    min(max(1, quantity), self.config.max_quantity_per_item))
                self._bump(tx, cart_id)
            return self._render(cart_id).model_dump()

        result = self.idempotency.run(
            principal_id=session.customer_id, operation="update_cart_item", key=idempotency_key,
            request=request, effect=effect)
        return Cart.model_validate(result)

    async def remove_from_cart(
        self,
        session: SessionContext,
        variant_id: str,
        *,
        expected_state_version: int | None = None,
        idempotency_key: str | None = None,
    ) -> Cart:
        cart_id = self._active_cart_id(session.customer_id)
        request = {"variant_id": variant_id, "expected_state_version": expected_state_version}

        def effect() -> dict:
            with self.store.transaction() as tx:
                self._guard_version(tx, cart_id, expected_state_version)
                tx.execute(
                    "DELETE FROM cart_lines WHERE cart_id = ? AND product_id = ?",
                    (cart_id, variant_id))
                self._bump(tx, cart_id)
            return self._render(cart_id).model_dump()

        result = self.idempotency.run(
            principal_id=session.customer_id, operation="remove_from_cart", key=idempotency_key,
            request=request, effect=effect)
        return Cart.model_validate(result)

    async def stage_checkout(
        self,
        session: SessionContext,
        *,
        fulfillment_option: str,
        note: str | None = None,
        idempotency_key: str | None = None,
    ) -> StagedCheckout:
        cart = await self.get_cart(session)
        if not cart.lines:
            raise Unavailable(
                "The cart is empty, so there is nothing to stage. Nothing was created."
            )
        for line in cart.lines:
            if self._sellable(line.variant_id) < line.quantity:
                raise Unavailable(
                    f"variant_id {line.variant_id} no longer has {line.quantity} units "
                    "available, so the checkout was not staged."
                )
        request = {"cart_id": cart.cart_id, "cart_state_version": cart.state_version,
                   "fulfillment_option": fulfillment_option, "note": note}

        subtotal = sum(line.quantity * line.unit_price_minor for line in cart.lines)
        discount, promotion_id = (
            self.stage_discount(session.customer_id, subtotal) if self.stage_discount else (0, None))

        def effect() -> dict:
            stage = self.checkout.stage(
                discount_minor=discount,
                promotion_id=promotion_id,
                customer_id=session.customer_id,
                cart_id=cart.cart_id,
                cart_state_version=cart.state_version,
                lines=[{
                    "variant_id": line.variant_id,
                    "quantity": line.quantity,
                    "unit_price_minor": line.unit_price_minor,
                } for line in cart.lines],
                fulfillment_option=fulfillment_option,
                constraints_note=note,
                correlation=session.correlation(),
            )
            return self._stage(stage).model_dump()

        result = self.idempotency.run(
            principal_id=session.customer_id, operation="stage_checkout", key=idempotency_key,
            request=request, effect=effect)
        return StagedCheckout.model_validate(result)

    async def read_stage(self, session: SessionContext, stage_id: str) -> StagedCheckout | None:
        rows = self.store.rows(
            "SELECT id FROM checkout_stages WHERE id = ? AND customer_id = ?",
            (stage_id, session.customer_id),
        )
        if not rows:
            return None
        return self._stage(self.checkout.read_stage(stage_id))

    # -- customer context, orders, policies, fulfillment -----------------------

    async def get_preferences(self, session: SessionContext) -> Preferences:
        rows = self.store.rows(
            "SELECT id,email,display_name FROM customers WHERE id = ?", (session.customer_id,)
        )
        row = rows[0] if rows else {}
        return Preferences(
            customer_id=session.customer_id,
            display_name=row.get("display_name"),
            email=row.get("email"),
        )

    async def get_orders(self, session: SessionContext, limit: int = 5) -> list[Order]:
        rows = self.store.rows(
            "SELECT id,status,created_at,currency,total_minor,amount_paid_minor "
            "FROM commerce_orders WHERE customer_id = ? ORDER BY created_at DESC LIMIT ?",
            (session.customer_id, max(1, limit)),
        )
        return [self._order(row) for row in rows]

    async def get_order(self, session: SessionContext, order_id: str) -> Order | None:
        rows = self.store.rows(
            "SELECT id,status,created_at,currency,total_minor,amount_paid_minor "
            "FROM commerce_orders WHERE id = ? AND customer_id = ?",
            (order_id, session.customer_id),
        )
        return self._order(rows[0]) if rows else None

    async def search_policies(self, session: SessionContext, query: str) -> list[Policy]:
        # Cartisan has no policy corpus yet. The tool stays registered and answers
        # `unavailable`, which is what the executor turns this into: a store term the
        # agent cannot source is one it must decline to state, not one it may invent.
        raise Unavailable(
            "Cartisan's terms are not readable from this conversation yet, so no return "
            "window, refund timing, warranty, or fee can be stated here."
        )

    async def get_fulfillment_options(
        self, session: SessionContext, variant_ids: list[str]
    ) -> list[FulfillmentOption]:
        known = [
            variant_id
            for variant_id in variant_ids[:20]
            if self.store.rows("SELECT id FROM catalog_variants WHERE id = ?", (variant_id,))
        ]
        if not known:
            raise Unavailable("None of those variant_ids are in the catalogue.")
        locations = self.store.rows(
            "SELECT code,name,region FROM inventory_locations ORDER BY code LIMIT 1"
        )
        options = [FulfillmentOption(method="delivery", eta="2-4 business days", fee_minor=0)]
        if locations:
            options.append(
                FulfillmentOption(
                    method="pickup",
                    eta="same day",
                    fee_minor=0,
                    location=f"{locations[0]['name']} ({locations[0]['region']})",
                )
            )
        return options

    # -- catalogue reads the host shares with the agent -------------------------
    # The REST catalogue and the agent's tool results must quote the same price and
    # the same stock, so both go through these rather than through a second query.

    def sellable(self, variant_id: str) -> int:
        """Units that can be sold right now: on hand, less what is already held."""
        return self._sellable(variant_id)

    def current_price(self, variant_id: str) -> int:
        """The price in effect now, promotional first, in paise."""
        return self._price(variant_id)

    # -- internals ------------------------------------------------------------

    def _variant(self, row: dict) -> Variant:
        variant_id = row["variant_id"]
        sellable = self._sellable(variant_id)
        return Variant(
            variant_id=variant_id,
            product_id=row["product_id"],
            sku=row["sku"],
            title=row["title"] or row["product_title"],
            brand=row["brand"],
            category=row.get("category"),
            price_minor=self._price(variant_id),
            in_stock=sellable > 0,
            sellable=sellable,
            options=self.store.load(row["options"]) if row.get("options") else {},
            origin=row.get("origin") or "seeded",
            specs=self._specs(variant_id),
        )

    def _price(self, variant_id: str) -> int:
        rows = self.store.rows(
            "SELECT amount_minor FROM variant_prices WHERE variant_id = ? AND valid_from <= ? "
            "AND (valid_to IS NULL OR valid_to > ?) "
            "ORDER BY CASE price_kind WHEN 'promotional' THEN 0 ELSE 1 END, valid_from DESC "
            "LIMIT 1",
            (variant_id, iso_now(), iso_now()),
        )
        return int(rows[0]["amount_minor"]) if rows else 0

    def _sellable(self, variant_id: str) -> int:
        rows = self.store.rows(
            "SELECT on_hand, reserved FROM inventory_levels WHERE variant_id = ?", (variant_id,)
        )
        return sum(max(0, int(row["on_hand"]) - int(row["reserved"])) for row in rows)

    def _specs(self, variant_id: str) -> dict[str, str]:
        specs: dict[str, str] = {}
        for row in self.store.rows(
            "SELECT spec_key,value_text,value_numeric,value_unit,value_bool FROM variant_specs "
            "WHERE variant_id = ? ORDER BY spec_key",
            (variant_id,),
        ):
            specs[row["spec_key"]] = _spec_value(row)
        return specs

    def _capabilities(self, variant_id: str) -> dict[str, str]:
        capabilities: dict[str, str] = {}
        for row in self.store.rows(
            "SELECT c.label AS label, vc.value_text, vc.value_numeric, vc.value_bool "
            "FROM variant_capabilities vc JOIN capabilities c ON c.id = vc.capability_id "
            "WHERE vc.variant_id = ? ORDER BY c.label",
            (variant_id,),
        ):
            capabilities[row["label"]] = _capability_value(row)
        return capabilities

    def _evaluate(self, requiring_id: str, offering_id: str) -> list[CompatibilityFinding]:
        """Every requirement `requiring_id` states, checked against what `offering_id`
        offers. The verdict is these rows and nothing else (ADR 0006)."""
        findings: list[CompatibilityFinding] = []
        for rule in self.store.rows(
            "SELECT r.capability_id, r.operator, r.value_text, r.value_numeric, r.severity, "
            "r.explanation, c.label AS label FROM variant_requirements r "
            "JOIN capabilities c ON c.id = r.capability_id WHERE r.variant_id = ? ORDER BY r.id",
            (requiring_id,),
        ):
            offered = self.store.rows(
                "SELECT value_text,value_numeric,value_bool FROM variant_capabilities "
                "WHERE variant_id = ? AND capability_id = ?",
                (offering_id, rule["capability_id"]),
            )
            observed = _capability_value(offered[0]) if offered else None
            expected = _rule_value(rule)
            findings.append(
                CompatibilityFinding(
                    capability=rule["label"],
                    operator=rule["operator"],
                    expected=expected,
                    observed=observed,
                    severity=rule["severity"],
                    satisfied=_satisfies(rule, offered[0] if offered else None),
                    explanation=rule["explanation"],
                )
            )
        return findings

    def _active_cart_id(self, customer_id: str) -> str:
        rows = self.store.rows(
            "SELECT id FROM customer_carts WHERE customer_id = ? AND status = 'active'",
            (customer_id,),
        )
        if rows:
            return rows[0]["id"]
        from uuid import uuid4

        cart_id = f"cart_{uuid4().hex[:12]}"
        try:
            self.store.execute(
                "INSERT INTO customer_carts (id,customer_id,status,state_version) "
                "VALUES (?,?,'active',0)",
                (cart_id, customer_id),
            )
        except Exception:
            # The partial unique index makes a concurrent create fail rather than
            # produce a second active cart; on that loss we re-read the winner.
            rows = self.store.rows(
                "SELECT id FROM customer_carts WHERE customer_id = ? AND status = 'active'",
                (customer_id,),
            )
            if not rows:
                raise
            return rows[0]["id"]
        return cart_id

    def _render(self, cart_id: str) -> Cart:
        version = self.store.rows(
            "SELECT state_version FROM customer_carts WHERE id = ?", (cart_id,)
        )
        lines: list[CartLine] = []
        for row in self.store.rows(
            "SELECT product_id AS variant_id, quantity FROM cart_lines WHERE cart_id = ? "
            "ORDER BY product_id",
            (cart_id,),
        ):
            named = self.store.rows(
                "SELECT title FROM catalog_variants WHERE id = ?", (row["variant_id"],)
            )
            if not named:
                # A line the catalogue no longer knows is dropped from the rendering
                # rather than priced at zero; the stage gate refuses it too.
                continue
            unit = self._price(row["variant_id"])
            lines.append(
                CartLine(
                    variant_id=row["variant_id"],
                    title=named[0]["title"],
                    quantity=int(row["quantity"]),
                    unit_price_minor=unit,
                    amount_minor=unit * int(row["quantity"]),
                )
            )
        return Cart(
            cart_id=cart_id,
            state_version=int(version[0]["state_version"]) if version else 0,
            lines=lines,
            subtotal_minor=sum(line.amount_minor for line in lines),
        )

    def _stage(self, stage: dict) -> StagedCheckout:
        lines = []
        for line in stage.get("lines", []):
            named = self.store.rows(
                "SELECT title FROM catalog_variants WHERE id = ?", (line["variant_id"],)
            )
            lines.append(
                CartLine(
                    variant_id=line["variant_id"],
                    title=named[0]["title"] if named else line["variant_id"],
                    quantity=int(line["quantity"]),
                    unit_price_minor=int(line["unit_price_minor"]),
                    amount_minor=int(line["amount_minor"]),
                )
            )
        return StagedCheckout(
            stage_id=stage["id"],
            cart_id=stage["cart_id"],
            cart_state_version=int(stage["cart_state_version"]),
            state=stage["state"],
            currency=stage.get("currency") or "INR",
            lines=lines,
            subtotal_minor=int(stage["subtotal_minor"]),
            shipping_minor=int(stage["shipping_minor"]),
            tax_minor=int(stage["tax_minor"]),
            discount_minor=int(stage["discount_minor"]),
            total_minor=int(stage["total_minor"]),
            fulfillment_option=stage["fulfillment_option"],
            constraints_note=stage.get("constraints_note"),
            expires_at=str(stage["expires_at"]),
        )

    def _order(self, row: dict) -> Order:
        lines = [
            OrderLine(
                variant_id=line["variant_id"],
                title=line.get("title") or line["variant_id"],
                quantity=int(line["quantity"]),
                unit_price_minor=int(line["unit_price_minor"]),
            )
            for line in self.store.rows(
                "SELECT l.variant_id, l.quantity, l.unit_price_minor, v.title AS title "
                "FROM commerce_order_lines l "
                "LEFT JOIN catalog_variants v ON v.id = l.variant_id WHERE l.order_id = ?",
                (row["id"],),
            )
        ]
        paid = int(row["amount_paid_minor"]) >= int(row["total_minor"]) and row["status"] == "paid"
        return Order(
            order_id=row["id"],
            status=row["status"],
            placed_at=row["created_at"],
            currency=row.get("currency") or "INR",
            total_minor=int(row["total_minor"]),
            lines=lines,
            # Anything short of a verified full payment reads as not yet paid (ADR 0013).
            payment_state="paid" if paid else row["status"],
        )

    @staticmethod
    def _guard_version(tx: Any, cart_id: str, expected: int | None) -> None:
        if expected is None:
            return
        rows = tx.rows("SELECT state_version FROM customer_carts WHERE id = ?", (cart_id,))
        actual = int(rows[0]["state_version"]) if rows else 0
        if actual != expected:
            raise Conflict(
                f"The cart moved since you read it: you sent state_version {expected}, "
                f"it is now {actual}. Call get_cart again and redo the change against "
                "the fresh cart."
            )

    @staticmethod
    def _line_quantity(tx: Any, cart_id: str, variant_id: str) -> int:
        rows = tx.rows(
            "SELECT quantity FROM cart_lines WHERE cart_id = ? AND product_id = ?",
            (cart_id, variant_id),
        )
        return int(rows[0]["quantity"]) if rows else 0

    @staticmethod
    def _line_count(tx: Any, cart_id: str) -> int:
        rows = tx.rows("SELECT COUNT(*) AS lines FROM cart_lines WHERE cart_id = ?", (cart_id,))
        return int(rows[0]["lines"]) if rows else 0

    @staticmethod
    def _write_line(tx: Any, cart_id: str, variant_id: str, quantity: int) -> None:
        tx.execute(
            "INSERT INTO cart_lines (cart_id,product_id,quantity) VALUES (?,?,?) "
            "ON CONFLICT(cart_id,product_id) DO UPDATE SET quantity=?",
            (cart_id, variant_id, quantity, quantity),
        )

    @staticmethod
    def _bump(tx: Any, cart_id: str) -> None:
        # `updated_at` is when the shopper last touched the cart; cart recovery reads it
        # to decide a cart was left, so every mutation moves it.
        tx.execute(
            "UPDATE customer_carts SET state_version = state_version + 1, updated_at = ? "
            "WHERE id = ?", (iso_now(), cart_id)
        )


def _tokens(text: str) -> list[str]:
    return [token for token in "".join(c.lower() if c.isalnum() else " " for c in text).split()]


def _term_forms(term: str) -> tuple[str, ...]:
    """The query spelling and a conservative English singular for token matching."""
    forms = [term]
    if len(term) > 3 and term.endswith("ies"):
        forms.append(f"{term[:-3]}y")
    elif len(term) > 3 and term.endswith("s") and not term.endswith("ss"):
        forms.append(term[:-1])
    return tuple(dict.fromkeys(forms))


def _matches_search_term(term: str, haystack: str) -> int:
    """Match complete tokens so `phone` cannot leak through `headphones`, and rank an
    exact match above one that only landed through the conservative singular fallback.

    Without this split, a query for "earbuds" scores an "Earbud Case" (matching only
    the stemmed singular "earbud") the same as an actual "Wireless Earbuds" product
    (matching the term verbatim) — a tie that the id-ordered tiebreak in
    `search_products` can resolve in the accessory's favor, crowding the real product
    out of a small result page entirely.
    """
    words = set(_tokens(haystack))
    forms = _term_forms(term)
    if forms[0] in words:
        return 2
    return 1 if any(form in words for form in forms[1:]) else 0


def _spec_value(row: dict) -> str:
    if row.get("value_text") is not None:
        return str(row["value_text"])
    if row.get("value_bool") is not None:
        return "yes" if row["value_bool"] else "no"
    numeric = _number(row.get("value_numeric"))
    unit = row.get("value_unit")
    return f"{numeric} {unit}".strip() if unit else numeric


def _capability_value(row: dict | None) -> str:
    if row is None:
        return "not offered"
    if row.get("value_text") is not None:
        return str(row["value_text"])
    if row.get("value_bool") is not None:
        return "yes" if row["value_bool"] else "no"
    return _number(row.get("value_numeric"))


def _rule_value(rule: dict) -> str:
    if rule["operator"] == "is_true":
        return "yes"
    if rule.get("value_text") is not None:
        return str(rule["value_text"])
    return _number(rule.get("value_numeric"))


def _number(value: Any) -> str:
    if value is None:
        return "unknown"
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


def _satisfies(rule: dict, offered: dict | None) -> bool:
    """One requirement row against one capability row. An absent capability satisfies
    nothing: the rule says what the other item must offer, and silence is not an offer."""
    operator = rule["operator"]
    if offered is None:
        return False
    if operator == "is_true":
        return bool(offered.get("value_bool"))
    if operator in {"eq", "neq", "in"}:
        expected = rule.get("value_text")
        observed = offered.get("value_text")
        if expected is None or observed is None:
            return False
        if operator == "eq":
            return observed == expected
        if operator == "neq":
            return observed != expected
        return observed in [part.strip() for part in str(expected).split(",")]
    expected_number, observed_number = rule.get("value_numeric"), offered.get("value_numeric")
    if expected_number is None or observed_number is None:
        return False
    if operator == "gte":
        return float(observed_number) >= float(expected_number)
    return float(observed_number) <= float(expected_number)


__all__ = ["CoreCommercePort", "ConflictError", "inr"]
