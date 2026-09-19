"""Server-issued presentation references, and the components that carry them.

ADR 0020: a conversational phrase such as "the best one" must resolve to something the
customer was actually shown. The model never mints a reference. When a presentation
renders, the server writes a `presentations` row and one `presentation_items` row per
card — each pinned to a catalogue variant and the exact price shown — and hands the
model back opaque `item_ref` values. `add_to_cart` takes one of those and nothing else,
so an unpresented product cannot reach the cart even if the model names it.

References expire. A reference resolved after its presentation's `expires_at` is
refused with the reason, because the price on the card is no longer the price the
catalogue would charge.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field

from commerce_common.presentation import (
    CHIPS_COMPONENT,
    CHIPS_TOOL,
    EnrichmentContext,
    PresentationComponent,
    PresentationPayload,
    PresentationRefused,
    PresentSuggestionsPayload,
)
from marketplace_backend.store import Store
from marketplace_backend.timeutil import as_datetime, now as iso_now

from .config import CartisanAgentConfig
from .types import SessionContext, SessionState, Variant, inr

PROVENANCE_GATE = "provenance"
REFERENCE_GATE = "presentation_reference"
CROSS_SELL_GATE = "cross_sell_bound"
OWNERSHIP_GATE = "ownership"
SETUP_GATE = "setup_budget"


class PresentationLedger:
    """Issues and resolves the references. One instance per process; every row it
    writes carries the conversation and the authenticated customer, so a reference
    issued in one customer's session cannot be redeemed in another's."""

    def __init__(self, store: Store, config: CartisanAgentConfig | None = None) -> None:
        self.store = store
        self.config = config or CartisanAgentConfig()

    def issue(
        self,
        session: SessionContext,
        kind: str,
        items: list[tuple[str, int]],
        *,
        turn_id: str | None = None,
    ) -> tuple[str, list[str]]:
        """Record one presentation and return `(presentation_id, item_refs)`, refs in
        the order the cards are shown."""
        presentation_id = f"pres_{uuid4().hex[:12]}"
        expires = (
            datetime.now(UTC) + timedelta(minutes=self.config.presentation_ttl_minutes)
        ).isoformat()
        refs: list[str] = []
        with self.store.transaction() as tx:
            tx.execute(
                "INSERT INTO presentations (id,conversation_id,customer_id,kind,turn_id,"
                "created_at,expires_at) VALUES (?,?,?,?,?,?,?)",
                (
                    presentation_id,
                    session.conversation_id,
                    session.customer_id,
                    kind,
                    turn_id,
                    iso_now(),
                    expires,
                ),
            )
            for position, (variant_id, unit_price_minor) in enumerate(items):
                item_ref = f"item_{uuid4().hex[:12]}"
                tx.execute(
                    "INSERT INTO presentation_items (id,presentation_id,position,variant_id,"
                    "unit_price_minor) VALUES (?,?,?,?,?)",
                    (item_ref, presentation_id, position, variant_id, unit_price_minor),
                )
                refs.append(item_ref)
        return presentation_id, refs

    def resolve(self, session: SessionContext, item_ref: str) -> dict[str, Any]:
        """The variant and price behind a reference, or a refusal naming the reason."""
        rows = self.store.rows(
            "SELECT i.variant_id, i.unit_price_minor, p.customer_id, p.conversation_id, "
            "p.expires_at FROM presentation_items i "
            "JOIN presentations p ON p.id = i.presentation_id WHERE i.id = ?",
            (item_ref,),
        )
        if not rows:
            raise PresentationRefused(
                f"item_ref {item_ref} was not issued in this session. Present the options "
                "first, then add the one the customer chooses; a variant_id is not an "
                "item_ref.",
                gate=REFERENCE_GATE,
            )
        row = rows[0]
        if row["customer_id"] != session.customer_id or (
            row["conversation_id"] != session.conversation_id
        ):
            raise PresentationRefused(
                f"item_ref {item_ref} belongs to another session.", gate=OWNERSHIP_GATE
            )
        expires = as_datetime(row["expires_at"])
        if expires is not None and expires <= datetime.now(UTC):
            raise PresentationRefused(
                f"item_ref {item_ref} has expired, so the price on that card is no longer "
                "current. Show the item again and add it from the fresh card.",
                gate=REFERENCE_GATE,
            )
        return {
            "variant_id": row["variant_id"],
            "unit_price_minor": int(row["unit_price_minor"]),
        }


# -- payloads ----------------------------------------------------------------------


class Pick(PresentationPayload):
    variant_id: str
    reason: str | None = None
    is_cross_sell: bool = False


class ProductsPayload(PresentationPayload):
    title: str | None = None
    layout: str = "carousel"
    purpose: Literal["shortlist", "setup"] = "shortlist"
    budget_minor: int | None = Field(default=None, ge=0)
    picks: list[Pick] = Field(min_length=1, max_length=12)


class ComparisonEntry(PresentationPayload):
    variant_id: str
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    best_for: str | None = None


class ComparisonPayload(PresentationPayload):
    title: str | None = None
    entries: list[ComparisonEntry] = Field(min_length=2, max_length=4)
    recommended_variant_id: str | None = None


class CartPayload(PresentationPayload):
    title: str | None = None
    note: str | None = None


class CheckoutPayload(PresentationPayload):
    stage_id: str
    note: str | None = None


class OrderStatusPayload(PresentationPayload):
    order_id: str
    summary: str


class GuideSection(PresentationPayload):
    heading: str
    body: str


class GuidePayload(PresentationPayload):
    title: str
    sections: list[GuideSection] = Field(min_length=1, max_length=8)
    related_variant_ids: list[str] = Field(default_factory=list)


# -- enrichment --------------------------------------------------------------------


def _services(context: EnrichmentContext) -> Any:
    return context.backend


def _state(context: EnrichmentContext) -> SessionState:
    return context.state


def _grounded(context: EnrichmentContext, variant_id: str) -> Variant:
    """The canonical record for an id the model named, or a refusal. This is the
    catalogue-grounding gate on the presentation side: a card can only show something
    a tool returned this session, and it shows the server's title, price, and stock,
    never the model's."""
    variant = _state(context).seen_variants.get(variant_id)
    if variant is None:
        raise PresentationRefused(
            f"variant_id {variant_id} was not returned by a catalogue tool in this "
            "session. Call get_product_details with this exact id, or find it with "
            "search_products, then present it from those results.",
            gate=PROVENANCE_GATE,
        )
    return variant


def _card(variant: Variant) -> dict[str, Any]:
    return {
        "variant_id": variant.variant_id,
        "title": variant.title,
        "brand": variant.brand,
        "price_minor": variant.price_minor,
        "price": variant.price,
        "currency": variant.currency,
        "in_stock": variant.in_stock,
        "options": variant.options,
        "specifications": variant.specs,
    }


async def _enrich_products(payload: ProductsPayload, context: EnrichmentContext) -> dict[str, Any]:
    config: CartisanAgentConfig = context.config
    session: SessionContext = context.session
    services = _services(context)

    cross_sells = [pick for pick in payload.picks if pick.is_cross_sell]
    if len(cross_sells) > config.max_cross_sells_per_turn:
        raise PresentationRefused(
            f"At most {config.max_cross_sells_per_turn} card may be a cross-sell; this "
            f"call marked {len(cross_sells)}. Show the rest as ordinary results.",
            gate=CROSS_SELL_GATE,
        )
    cart = await services.port.get_cart(session)
    in_cart = {line.variant_id for line in cart.lines}
    cards: list[dict[str, Any]] = []
    for pick in payload.picks:
        variant = _grounded(context, pick.variant_id)
        if pick.is_cross_sell:
            # A cross-sell pairs with something the customer already has, must be
            # buyable, and is never added on their behalf (ADR 0007).
            if not in_cart:
                raise PresentationRefused(
                    "A cross-sell needs something already in the cart to pair with; the "
                    "cart is empty.",
                    gate=CROSS_SELL_GATE,
                )
            if variant.variant_id in in_cart:
                raise PresentationRefused(
                    f"variant_id {variant.variant_id} is already in the cart, so it is "
                    "not a cross-sell.",
                    gate=CROSS_SELL_GATE,
                )
            if not variant.in_stock:
                raise PresentationRefused(
                    f"variant_id {variant.variant_id} is out of stock and cannot be "
                    "offered as a cross-sell.",
                    gate=CROSS_SELL_GATE,
                )
        cards.append({**_card(variant), "reason": pick.reason, "is_cross_sell": pick.is_cross_sell})

    setup_total = sum(card["price_minor"] for card in cards)
    if payload.purpose == "setup":
        if payload.budget_minor is None:
            raise PresentationRefused(
                "A setup needs the customer's stated total budget so the server can verify the total.",
                gate=SETUP_GATE,
            )
        if cross_sells:
            raise PresentationRefused("A setup cannot count an optional cross-sell as one of its parts.", gate=SETUP_GATE)
        if any(not card["in_stock"] for card in cards):
            raise PresentationRefused("Every product in a setup must currently be in stock.", gate=SETUP_GATE)
        if setup_total > payload.budget_minor:
            raise PresentationRefused(
                f"The setup total {inr(setup_total)} exceeds the customer's "
                f"{inr(payload.budget_minor)} budget. Present a working set within budget.",
                gate=SETUP_GATE,
            )
        anchor_id = cards[0]["variant_id"]
        for card in cards[1:]:
            key = f"{anchor_id}:{card['variant_id']}"
            verdict = _state(context).compatibility_verdicts.get(key)
            if verdict is None:
                raise PresentationRefused(
                    f"{card['variant_id']} was not checked for compatibility with setup anchor {anchor_id}.",
                    gate=SETUP_GATE,
                )
            if not verdict.compatible:
                raise PresentationRefused(
                    f"{card['variant_id']} is not compatible with setup anchor {anchor_id}.",
                    gate=SETUP_GATE,
                )
            card["fit_checks"] = [finding.explanation for finding in verdict.findings if finding.satisfied]

    presentation_id, refs = services.presentations.issue(
        session,
        "products",
        [(card["variant_id"], card["price_minor"]) for card in cards],
        turn_id=getattr(session, "turn_id", None),
    )
    for card, ref in zip(cards, refs, strict=True):
        card["item_ref"] = ref
    _state(context).issued_items.update(
        {card["item_ref"]: card["variant_id"] for card in cards}
    )
    _state(context).last_presented_variant_ids = [card["variant_id"] for card in cards]
    # The references have to reach the model, not just the host: they are what a later
    # add_to_cart names. That costs this round its clean close — a presenting round with
    # a note does not end the turn on its own — and the trade is deliberate, because the
    # alternative is a model that has been shown cards it cannot act on.
    context.notes.append(
        "Cards issued, in order: "
        + "; ".join(f"{card['variant_id']} -> {card['item_ref']}" for card in cards)
        + ". add_to_cart takes the item_ref of the one the customer chooses."
    )
    return {
        "presentation_id": presentation_id,
        "title": payload.title,
        "layout": payload.layout,
        "purpose": payload.purpose,
        "budget_minor": payload.budget_minor,
        "budget": inr(payload.budget_minor) if payload.budget_minor is not None else None,
        "total_minor": setup_total,
        "total": inr(setup_total),
        "remaining_budget_minor": (
            payload.budget_minor - setup_total
            if payload.budget_minor is not None else None
        ),
        "remaining_budget": (
            inr(payload.budget_minor - setup_total)
            if payload.budget_minor is not None and payload.budget_minor >= setup_total
            else None
        ),
        "items": cards,
    }


async def _enrich_comparison(
    payload: ComparisonPayload, context: EnrichmentContext
) -> dict[str, Any]:
    services = _services(context)
    entries: list[dict[str, Any]] = []
    for entry in payload.entries:
        variant = _grounded(context, entry.variant_id)
        entries.append(
            {
                **_card(variant),
                "pros": entry.pros,
                "cons": entry.cons,
                "best_for": entry.best_for,
            }
        )
    if payload.recommended_variant_id:
        _grounded(context, payload.recommended_variant_id)
    presentation_id, refs = services.presentations.issue(
        context.session,
        "comparison",
        [(entry["variant_id"], entry["price_minor"]) for entry in entries],
        turn_id=getattr(context.session, "turn_id", None),
    )
    for entry, ref in zip(entries, refs, strict=True):
        entry["item_ref"] = ref
    _state(context).issued_items.update(
        {entry["item_ref"]: entry["variant_id"] for entry in entries}
    )
    _state(context).last_presented_variant_ids = [entry["variant_id"] for entry in entries]
    context.notes.append(
        "Entries issued: "
        + "; ".join(f"{entry['variant_id']} -> {entry['item_ref']}" for entry in entries)
        + "."
    )
    return {
        "presentation_id": presentation_id,
        "title": payload.title,
        "entries": entries,
        "recommended_variant_id": payload.recommended_variant_id,
    }


async def _enrich_cart(payload: CartPayload, context: EnrichmentContext) -> dict[str, Any]:
    cart = await _services(context).port.get_cart(context.session)
    return {
        "title": payload.title,
        "note": payload.note,
        "cart_id": cart.cart_id,
        "state_version": cart.state_version,
        "currency": cart.currency,
        "lines": [
            {**line.model_dump(), "amount": inr(line.amount_minor)} for line in cart.lines
        ],
        "subtotal_minor": cart.subtotal_minor,
        "subtotal": inr(cart.subtotal_minor),
    }


async def _enrich_checkout(payload: CheckoutPayload, context: EnrichmentContext) -> dict[str, Any]:
    stage = await _services(context).port.read_stage(context.session, payload.stage_id)
    if stage is None:
        raise PresentationRefused(
            f"stage_id {payload.stage_id} is not one of this customer's staged "
            "checkouts. Call stage_checkout, then present the stage it returns.",
            gate=OWNERSHIP_GATE,
        )
    if stage.state != "staged":
        raise PresentationRefused(
            f"That checkout preview is {stage.state}, so it cannot be confirmed. Stage "
            "the cart again.",
            gate=REFERENCE_GATE,
        )
    return {
        "note": payload.note,
        "stage_id": stage.stage_id,
        "state": stage.state,
        "currency": stage.currency,
        "lines": [
            {**line.model_dump(), "amount": inr(line.amount_minor)} for line in stage.lines
        ],
        "subtotal_minor": stage.subtotal_minor,
        "shipping_minor": stage.shipping_minor,
        "tax_minor": stage.tax_minor,
        "discount_minor": stage.discount_minor,
        "total_minor": stage.total_minor,
        "total": inr(stage.total_minor),
        "fulfillment_option": stage.fulfillment_option,
        "constraints_note": stage.constraints_note,
        "expires_at": stage.expires_at,
        # The card is a preview. Confirmation, reservation, the internal order and the
        # Paytm handoff are the host's, after the customer taps (ADR 0005, ADR 0012).
        "confirm_action": "host_confirm_checkout",
    }


async def _enrich_order_status(
    payload: OrderStatusPayload, context: EnrichmentContext
) -> dict[str, Any]:
    order = await _services(context).port.get_order(context.session, payload.order_id)
    if order is None:
        raise PresentationRefused(
            f"order_id {payload.order_id} is not one of this customer's orders.",
            gate=OWNERSHIP_GATE,
        )
    return {
        "summary": payload.summary,
        "order_id": order.order_id,
        "status": order.status,
        "payment_state": order.payment_state,
        "placed_at": str(order.placed_at),
        "currency": order.currency,
        "total_minor": order.total_minor,
        "total": inr(order.total_minor),
        "lines": [line.model_dump() for line in order.lines],
    }


async def _enrich_guide(payload: GuidePayload, context: EnrichmentContext) -> dict[str, Any]:
    related = [_card(_grounded(context, vid)) for vid in payload.related_variant_ids]
    return {
        "title": payload.title,
        "sections": [section.model_dump() for section in payload.sections],
        "related": related,
    }


async def _enrich_suggestions(
    payload: PresentSuggestionsPayload, context: EnrichmentContext
) -> dict[str, Any]:
    """Treat a Browse/Shop chip as an inventory claim, not decorative prose.

    Exact catalogue category labels are intentionally required. A product result may
    mention a phone case, but that is not authority to widen the claim to "phones".
    """
    categories = {
        _catalogue_label(variant.category)
        for variant in _state(context).seen_variants.values()
        if variant.category
    }
    for chip in payload.suggestions:
        match = re.match(r"^\s*(?:browse|shop)\s+(.+?)\s*$", chip, re.IGNORECASE)
        if match is None:
            continue
        requested = _catalogue_label(match.group(1))
        if requested not in categories:
            available = ", ".join(sorted(categories)) or "none read this turn"
            raise PresentationRefused(
                f"{chip!r} claims a catalogue category that was not returned by a "
                f"catalogue read. Use an exact returned category label; available: "
                f"{available}.",
                gate=PROVENANCE_GATE,
            )
    return payload.model_dump(exclude_none=True)


def _catalogue_label(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


PRESENTATION_COMPONENTS: dict[str, PresentationComponent] = {
    "present_products": PresentationComponent(
        name="present_products",
        component="products",
        payload_model=ProductsPayload,
        enrich=_enrich_products,
    ),
    "present_comparison": PresentationComponent(
        name="present_comparison",
        component="comparison",
        payload_model=ComparisonPayload,
        enrich=_enrich_comparison,
    ),
    "present_cart": PresentationComponent(
        name="present_cart",
        component="cart",
        payload_model=CartPayload,
        enrich=_enrich_cart,
    ),
    "present_checkout": PresentationComponent(
        name="present_checkout",
        component="checkout",
        payload_model=CheckoutPayload,
        enrich=_enrich_checkout,
    ),
    "present_order_status": PresentationComponent(
        name="present_order_status",
        component="order_status",
        payload_model=OrderStatusPayload,
        enrich=_enrich_order_status,
    ),
    "present_guide": PresentationComponent(
        name="present_guide",
        component="guide",
        payload_model=GuidePayload,
        enrich=_enrich_guide,
    ),
    CHIPS_TOOL: PresentationComponent(
        name=CHIPS_TOOL,
        component=CHIPS_COMPONENT,
        payload_model=PresentSuggestionsPayload,
        enrich=_enrich_suggestions,
    ),
}
