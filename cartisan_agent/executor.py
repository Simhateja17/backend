"""One handler per boundary tool, over `commerce_common`'s executor frame.

`execute` never raises: every path out of a tool is one of the five typed outcomes
(`outcomes.Outcome`), so an expected business refusal reaches the model as something it
can recover from and reaches the ledger as something a judge can read. The frame owns
dispatch, the `status` line, skills, presentation, and memory; this class owns Cartisan's
handlers, its refusal wording, and the gates that sit in front of a write.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from commerce_common.execution import BaseToolExecutor, Handler, clamp_limit, parse_argument
from commerce_common.memory import MEMORY_EXTRACTION_TEMPLATE, MemoryRuntime
from commerce_common.presentation import PresentationExtension, PresentationRefused
from commerce_common.skills import SkillRegistry
from commerce_common.streaming import AgentEvent, ToolOutcome
from marketplace_backend.carts import ConflictError

from .config import FORBIDDEN_TOOLS, CartisanAgentConfig
from .fences import CARTISAN_FENCE
from .gates import (
    CHECKOUT_PRECEDENCE_GATE,
    CHECKOUT_TURN_REFUSED,
    FORBIDDEN_GATE,
    cart_lock,
    checkout_precedence_error,
    forbidden_error,
)
from .outcomes import BusinessRefusal, Conflict, Outcome, Unavailable, refusal, tag
from .ports import CommercePort
from .presentations import PRESENTATION_COMPONENTS, PresentationLedger
from .types import Cart, SearchFilters, SessionContext, SessionState, inr

MAX_ORDERS = 20
MAX_FULFILLMENT_IDS = 20

CARTISAN_MEMORY_EXTRACTION_PROMPT = MEMORY_EXTRACTION_TEMPLATE.format(
    keeper="an electronics and smart-lifestyle retailer",
    subject="one customer",
    occasions="visits",
    speaker="the customer",
    qualifies=(
        "a preference, a limit, or standing context the customer stated themselves: a "
        "budget ceiling for a category, the phone or laptop they already own and buy "
        "accessories for, a connector or platform they need everything to match, a room "
        "they are kitting out."
    ),
    standalone_example=(
        '"under 5,000" tells a future reader nothing, while "keeps accessory purchases '
        'under 5,000 rupees each" tells them everything'
    ),
    live_key_rule=(
        "Keep the customer's one live project (a desk setup, a home theatre, a gift) as a "
        'single fact under the key "current_project", naming what it is for and its '
        "budget; a new project replaces it. Every other key is one of these typed keys, "
        "and a fact that fits none of them is not recorded: device_owned:<kind> (phone, "
        "laptop, tablet, watch, tv, console), brand_affinity:<brand> (value says liked or "
        "avoided and why), budget:<category>, use_case (gym, travel, work from home), "
        "gift_context (recipient type and occasion, never a name), rejected_item:<product>. "
        'When the customer says a fact is no longer true ("I sold my Pixel"), record its '
        'key with the value "none".'
    ),
    excluded=(
        "anything that came from catalogue records, results, or Cartisan's own terms; the "
        "mechanics of this visit; your own guesses; and payment, identity, or health "
        "details in every case."
    ),
)


@dataclass
class CommerceServices:
    """What the handlers and the presentation hooks work through. The frame calls this
    the `backend`; splitting it keeps the port free of anything to do with rendering."""

    port: CommercePort
    presentations: PresentationLedger


def build_memory(config: CartisanAgentConfig, store: Any = None) -> MemoryRuntime:
    return MemoryRuntime.build(
        config,
        store,
        fence=CARTISAN_FENCE,
        extraction_prompt=CARTISAN_MEMORY_EXTRACTION_PROMPT,
    )


class CartisanToolExecutor(BaseToolExecutor):
    fence = CARTISAN_FENCE
    components = PRESENTATION_COMPONENTS
    displayed_text = "Displayed to the customer."
    unavailable_text = (
        "{name} is temporarily unavailable. Work with what you already have, or tell the "
        "customer plainly that you cannot check it right now."
    )
    absent_text = (
        "{name} is not something you can do. Say what you can do instead, and do not "
        "describe the action as done."
    )

    def __init__(
        self,
        *,
        backend: CommerceServices,
        config: CartisanAgentConfig,
        skills: SkillRegistry,
        session: SessionContext,
        state: SessionState,
        memory: MemoryRuntime | None = None,
        extensions: Sequence[PresentationExtension] = (),
    ) -> None:
        super().__init__(
            backend=backend,
            config=config,
            skills=skills,
            session=session,
            state=state,
            memory=memory or build_memory(config),
            extensions=extensions,
        )

    @property
    def memory_subject(self) -> str:
        return self._session.customer_id

    @property
    def port(self) -> CommercePort:
        return self._backend.port

    def domain_error(self, error: Exception) -> ToolOutcome | None:
        """Cartisan's own refusals, typed. Anything else falls through to the frame's
        ladder and is reported as the tool being unavailable, and logged."""
        if isinstance(error, BusinessRefusal):
            return refusal(error)
        if isinstance(error, PresentationRefused):
            # A reference the ledger would not resolve. `run_presentation` catches this
            # inside a presentation call; raised from a handler — `add_to_cart` resolving
            # an item_ref — it must land as the same held outcome and not as a failure,
            # or a gate doing its job would read to the model as an outage to retry.
            return tag(
                ToolOutcome.held(error.gate or "presentation", str(error)), Outcome.BLOCKED
            )
        if isinstance(error, ConflictError):
            return refusal(Conflict(str(error)))
        return None

    async def dispatch(self, name: str, tool_input: dict[str, Any]) -> ToolOutcome:
        """Checkout precedence sits ahead of every other gate, because on a checkout
        turn the question is not whether a search would be well-formed (ADR 0021)."""
        if self._state.checkout_turn and name in CHECKOUT_TURN_REFUSED:
            return tag(
                ToolOutcome.held(CHECKOUT_PRECEDENCE_GATE, checkout_precedence_error(name)),
                Outcome.BLOCKED,
            )
        # A forbidden capability is refused in its own words, so the model learns it is
        # not a thing it has rather than a thing that is down (ADR 0015).
        if name in FORBIDDEN_TOOLS:
            return tag(ToolOutcome.held(FORBIDDEN_GATE, forbidden_error(name)), Outcome.BLOCKED)
        return await super().dispatch(name, tool_input)

    def handlers(self) -> dict[str, Handler]:
        return {
            "search_products": self._search_products,
            "get_product_details": self._get_product_details,
            "check_compatibility": self._check_compatibility,
            "get_cart": self._get_cart,
            "add_to_cart": self._add_to_cart,
            "update_cart_item": self._update_cart_item,
            "remove_from_cart": self._remove_from_cart,
            "stage_checkout": self._stage_checkout,
            "get_preferences": self._get_preferences,
            "get_orders": self._get_orders,
            "get_order_status": self._get_order_status,
            "search_policies": self._search_policies,
            "get_fulfillment_options": self._get_fulfillment_options,
        }

    # -- catalogue -----------------------------------------------------------------

    async def _search_products(self, tool_input: dict[str, Any]) -> ToolOutcome:
        query = self._sanitize(tool_input.get("query", ""), 300)
        filters = (
            parse_argument(SearchFilters, tool_input["filters"])
            if tool_input.get("filters")
            else None
        )
        comparison_anchor = None
        anchor_id = self._state.cheaper_anchor_variant_id
        anchor = self._state.seen_variants.get(anchor_id) if anchor_id else None
        if anchor is not None:
            # "Cheaper than this" is a strict catalogue constraint, not prompt advice.
            # Preserve any narrower ceiling the model supplied, but never let an
            # omitted or looser filter support a false claim that no cheaper item exists.
            filters = filters or SearchFilters()
            strict_ceiling = max(0, anchor.price_minor - 1)
            filters.max_price_minor = min(
                filters.max_price_minor
                if filters.max_price_minor is not None
                else strict_ceiling,
                strict_ceiling,
            )
            filters.sort = "price_asc"
            comparison_anchor = {
                "variant_id": anchor.variant_id,
                "title": anchor.title,
                "price_minor": anchor.price_minor,
                "price": anchor.price,
            }
        variants = await self.port.search_products(
            self._session, query, filters, self._search_limit(tool_input.get("limit"))
        )
        self._state.remember_variants(variants)
        if not variants:
            await self.port.record_unmet_demand(self._session, query, filters)
            return ToolOutcome(
                f"No active catalogue variants matched {query!r}. Nothing was found; do "
                "not describe an item the catalogue did not return."
            )
        return self._fenced(
            {
                "query": query,
                "comparison_anchor": comparison_anchor,
                "count": len(variants),
                "results": [_variant_payload(variant) for variant in variants],
            }
        )

    async def _get_product_details(self, tool_input: dict[str, Any]) -> ToolOutcome:
        variant_id = str(tool_input.get("variant_id", ""))
        details = await self.port.get_product_details(self._session, variant_id)
        if details is None:
            return ToolOutcome.error(f"No catalogue variant with id {variant_id}.")
        # The siblings enter provenance with the record, so a switch of capacity or
        # colour needs no second search.
        self._state.remember_variants([details, *details.siblings])
        return self._fenced(
            {
                **_variant_payload(details),
                "description": details.description,
                "capabilities": details.capabilities,
                "requirements": details.requirements,
                "siblings": [_variant_payload(sibling) for sibling in details.siblings],
            }
        )

    async def _check_compatibility(self, tool_input: dict[str, Any]) -> ToolOutcome:
        verdict = await self.port.check_compatibility(
            self._session,
            str(tool_input.get("base_variant_id", "")),
            str(tool_input.get("candidate_variant_id", "")),
        )
        self._state.remember_compatibility(verdict)
        return self._fenced(
            {
                **verdict.model_dump(),
                "note": (
                    "This verdict is the catalogue's structured rules evaluated. State it "
                    "as it stands; do not soften a blocking finding or add a compatibility "
                    "claim of your own."
                ),
            }
        )

    # -- cart ----------------------------------------------------------------------

    async def _get_cart(self, _: dict[str, Any]) -> ToolOutcome:
        return self._fenced(_cart_payload(await self.port.get_cart(self._session)))

    async def _add_to_cart(self, tool_input: dict[str, Any]) -> ToolOutcome:
        item_ref = str(tool_input.get("item_ref", ""))
        # The reference gate: this resolves a server-issued ref or refuses (ADR 0020).
        resolved = self._backend.presentations.resolve(self._session, item_ref)
        variant_id = resolved["variant_id"]
        requested = max(1, int(tool_input.get("quantity") or 1))
        cap = self._config.max_quantity_per_item
        async with cart_lock(self._session):
            current = await self.port.get_cart(self._session)
            held = next((line for line in current.lines if line.variant_id == variant_id), None)
            allowed = min(requested, max(0, cap - (held.quantity if held else 0)))
            if allowed <= 0:
                raise Unavailable(f"That item is already at the per-line limit of {cap}.")
            cart = await self.port.add_to_cart(
                self._session,
                variant_id,
                allowed,
                expected_state_version=_version(tool_input),
                idempotency_key=self._idempotency_key("add_to_cart", item_ref, allowed),
            )
        capped = f" (capped at the per-line limit of {cap})" if allowed < requested else ""
        return self._written(
            f"Added {variant_id} x{allowed}{capped}. {_cart_summary(cart)}", cart
        )

    async def _update_cart_item(self, tool_input: dict[str, Any]) -> ToolOutcome:
        variant_id = str(tool_input.get("variant_id", ""))
        quantity = min(
            max(1, int(tool_input.get("quantity") or 1)), self._config.max_quantity_per_item
        )
        async with cart_lock(self._session):
            cart = await self.port.update_cart_item(
                self._session,
                variant_id,
                quantity,
                expected_state_version=_version(tool_input),
                idempotency_key=self._idempotency_key("update_cart_item", variant_id, quantity),
            )
        return self._written(f"Set {variant_id} to {quantity}. {_cart_summary(cart)}", cart)

    async def _remove_from_cart(self, tool_input: dict[str, Any]) -> ToolOutcome:
        variant_id = str(tool_input.get("variant_id", ""))
        async with cart_lock(self._session):
            cart = await self.port.remove_from_cart(
                self._session,
                variant_id,
                expected_state_version=_version(tool_input),
                idempotency_key=self._idempotency_key("remove_from_cart", variant_id, 0),
            )
        return self._written(f"Removed {variant_id}. {_cart_summary(cart)}", cart)

    async def _stage_checkout(self, tool_input: dict[str, Any]) -> ToolOutcome:
        option = str(tool_input.get("fulfillment_option") or "delivery")
        note = self._sanitize(tool_input.get("note"), 300) or None
        async with cart_lock(self._session):
            stage = await self.port.stage_checkout(
                self._session,
                fulfillment_option=option,
                note=note,
                idempotency_key=self._idempotency_key("stage_checkout", option, 0),
            )
        return self._fenced(
            {
                "stage_id": stage.stage_id,
                "state": stage.state,
                "currency": stage.currency,
                "lines": [line.model_dump() for line in stage.lines],
                "subtotal_minor": stage.subtotal_minor,
                "total_minor": stage.total_minor,
                "total": inr(stage.total_minor),
                "fulfillment_option": stage.fulfillment_option,
                "expires_at": stage.expires_at,
                "note": (
                    "This is a preview. No order exists, no stock is held, no payment link "
                    "was created and no money moved. Present it with present_checkout and "
                    "let the customer confirm in the app."
                ),
            }
        )

    # -- context, orders, policies, fulfillment ------------------------------------

    async def _get_preferences(self, _: dict[str, Any]) -> ToolOutcome:
        preferences = await self.port.get_preferences(self._session)
        return self._fenced(preferences.model_dump(exclude_none=True))

    async def _get_orders(self, tool_input: dict[str, Any]) -> ToolOutcome:
        orders = await self.port.get_orders(
            self._session, clamp_limit(tool_input.get("limit"), 5, MAX_ORDERS)
        )
        return self._fenced({"orders": [_order_payload(order) for order in orders]})

    async def _get_order_status(self, tool_input: dict[str, Any]) -> ToolOutcome:
        order_id = str(tool_input.get("order_id", ""))
        order = await self.port.get_order(self._session, order_id)
        if order is None:
            return ToolOutcome.error(f"No order with id {order_id} belongs to this customer.")
        return self._fenced(_order_payload(order))

    async def _search_policies(self, tool_input: dict[str, Any]) -> ToolOutcome:
        query = self._sanitize(tool_input.get("query", ""), 200)
        policies = await self.port.search_policies(self._session, query)
        if not policies:
            raise Unavailable(
                f"No Cartisan policy passage matched {query!r}, so that term cannot be "
                "stated in this conversation."
            )
        return self._fenced({"policies": [policy.model_dump() for policy in policies]})

    async def _get_fulfillment_options(self, tool_input: dict[str, Any]) -> ToolOutcome:
        variant_ids = [str(vid) for vid in tool_input.get("variant_ids") or []][
            :MAX_FULFILLMENT_IDS
        ]
        options = await self.port.get_fulfillment_options(self._session, variant_ids)
        return self._fenced(
            {
                "options": [
                    {**option.model_dump(), "fee": inr(option.fee_minor)} for option in options
                ]
            }
        )

    # -- helpers --------------------------------------------------------------------

    def _written(self, text: str, cart: Cart) -> ToolOutcome:
        return tag(
            ToolOutcome(text, [AgentEvent.cart_update(_cart_payload(cart))]), Outcome.APPLIED
        )

    def _idempotency_key(self, operation: str, subject: str, quantity: int) -> str:
        """One key per (conversation, turn, operation, subject, quantity). A retried
        model round producing the same call is the same effect; a different quantity is
        a different one (ADR 0029)."""
        turn = getattr(self._session, "turn_id", None) or self._session.conversation_id
        return f"{turn}:{operation}:{subject}:{quantity}"


def _version(tool_input: dict[str, Any]) -> int | None:
    raw = tool_input.get("expected_state_version")
    return int(raw) if raw is not None else None


def _variant_payload(variant: Any) -> dict[str, Any]:
    return {
        "variant_id": variant.variant_id,
        "title": variant.title,
        "brand": variant.brand,
        "category": variant.category,
        "price_minor": variant.price_minor,
        "price": variant.price,
        "currency": variant.currency,
        "in_stock": variant.in_stock,
        "options": variant.options,
        "specifications": variant.specs,
    }


def _order_payload(order: Any) -> dict[str, Any]:
    return {
        "order_id": order.order_id,
        "status": order.status,
        # Kept distinct from `status` on purpose: an order is paid only when a verified
        # provider outcome says so, never because a browser came back (ADR 0013).
        "payment_state": order.payment_state,
        "placed_at": str(order.placed_at),
        "currency": order.currency,
        "total_minor": order.total_minor,
        "total": inr(order.total_minor),
        "lines": [line.model_dump() for line in order.lines],
    }


def _cart_payload(cart: Cart) -> dict[str, Any]:
    return {
        "cart_id": cart.cart_id,
        "state_version": cart.state_version,
        "currency": cart.currency,
        "lines": [
            {**line.model_dump(), "amount": inr(line.amount_minor)} for line in cart.lines
        ],
        "subtotal_minor": cart.subtotal_minor,
        "subtotal": inr(cart.subtotal_minor),
    }


def _cart_summary(cart: Cart) -> str:
    if not cart.lines:
        return "The cart is now empty."
    return (
        f"The cart now has {cart.item_count} item(s) across {len(cart.lines)} line(s), "
        f"subtotal {inr(cart.subtotal_minor)}, state_version {cart.state_version}."
    )
