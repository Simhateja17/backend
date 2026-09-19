"""The tool contracts, in a fixed order.

The names are the architecture document's "Tool boundary" section, verbatim, and are
the contract: `tests/test_runtime_contracts.py` pins them, so renaming one is a
deliberate act with a version bump beside it. The list depends only on the deployment
config, never on the request, because it is part of the cached prefix (ADR 0028);
whether a capability can serve a particular call is decided in the executor and the
gates, not by adding or removing a tool.

Two boundary rules are expressed in the schemas themselves rather than in prose the
model may reinterpret:

* `add_to_cart` takes an `item_ref`, never a raw catalogue id. A reference is issued
  by the server when it renders a presentation, so "add the best one" resolves to
  something the customer was actually shown (ADR 0020).
* `stage_checkout` takes no cart. It reads the authenticated customer's authoritative
  cart server-side, so neither the model nor the transcript can substitute one
  (ADR 0021).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from commerce_common.execution import LOAD_SKILL, with_status

from .config import CartisanAgentConfig, MerchantAgentConfig

# The boundary, as named in CARTISAN_COMMERCE_ARCHITECTURE.md § Tool boundary.
SHOPPING_READS: tuple[str, ...] = (
    "search_products", "get_product_details", "check_compatibility", "get_cart",
    "get_orders", "get_order_status", "search_policies", "get_fulfillment_options",
    "get_preferences", "recall_memories", "load_skill",
)
SHOPPING_MUTATIONS: tuple[str, ...] = (
    "add_to_cart", "update_cart_item", "remove_from_cart", "stage_checkout",
)
SHOPPING_PRESENTATION: tuple[str, ...] = (
    "present_products", "present_comparison", "present_cart", "present_checkout",
    "present_order_status", "present_guide", "present_suggestions",
)
MERCHANT_READS: tuple[str, ...] = (
    "get_business_snapshot", "query_metrics", "search_listings", "get_listing",
    "get_inventory_alerts", "get_unmet_demand", "get_pricing_context", "get_campaign_performance",
    "get_pending_changes", "load_skill", "get_payment_health", "check_restock_financing",
    "get_recovery_policy", "recall_memories",
)
MERCHANT_STAGING: tuple[str, ...] = (
    "stage_inventory_action", "stage_price_update", "stage_promotion", "stage_campaign",
    "stage_listing_update", "stage_loan_request", "stage_recovery_policy",
)
MERCHANT_PRESENTATION: tuple[str, ...] = (
    "present_digest", "present_metrics", "present_change_preview", "present_suggestions",
)

_CUSTOMER = "the customer"
_OPERATOR = "the operator"

_VARIANT_ID = "A variant_id a catalogue tool returned this session."
_ITEM_REF = (
    "An item_ref from a presentation you rendered this session. This is the only way "
    "to name something for the cart: it proves the customer was shown this exact "
    "variant at this exact price."
)


def _string(description: str, **extra: Any) -> dict[str, Any]:
    return {"type": "string", "description": description, **extra}


def _object(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


def _empty() -> dict[str, Any]:
    return _object({})


def _title(what: str) -> dict[str, Any]:
    return {"type": "string", "maxLength": 80, "description": f"Short heading for the {what}."}


def _reason(what: str = "pick") -> dict[str, Any]:
    return {
        "type": "string",
        "maxLength": 140,
        "description": f"One clause tying the {what} to a need the customer stated.",
    }


def _load_skill(skill_names: Sequence[str]) -> dict[str, Any]:
    return {
        "name": LOAD_SKILL,
        "description": (
            "Load the rules of the flow whose entry in the skill index the request "
            "matches; they are not in your prompt. Call it in the same round as the "
            "flow's first read and follow them for the rest of the flow."
        ),
        "input_schema": _object(
            {
                "skill_name": _string(
                    "Name of the skill as listed in the index.", enum=sorted(skill_names)
                )
            },
            ["skill_name"],
        ),
    }


def _filters_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "description": "Constraints the customer stated; leave guesses in the query.",
        "properties": {
            "category": _string("Catalogue category name."),
            "brand": _string("Brand name, when the customer named one."),
            "min_price_minor": {
                "type": "integer",
                "minimum": 0,
                "description": "Lowest acceptable price, in paise (₹1 = 100).",
            },
            "max_price_minor": {
                "type": "integer",
                "minimum": 0,
                "description": "Price ceiling the customer stated, in paise (₹1 = 100).",
            },
            "in_stock_only": {
                "type": "boolean",
                "description": "Leave unset; out-of-stock variants are excluded by default.",
            },
            "sort": _string(
                "Result order; relevance unless the customer asked otherwise.",
                enum=["relevance", "price_asc", "price_desc"],
            ),
        },
        "additionalProperties": False,
    }


def build_shopping_tools(
    config: CartisanAgentConfig, skill_names: Sequence[str] = ()
) -> list[dict[str, Any]]:
    """The shopping surface, built once per deployment."""

    reads: list[dict[str, Any]] = [
        _load_skill(skill_names),
        {
            "name": "search_products",
            "description": (
                "Search the catalogue; returns variants with variant_id, title, brand, "
                "price and stock. Cartisan sells variants, so every result is something "
                "that can be bought as it stands. Use a specific query, put stated "
                "constraints in filters, and run one search per distinct item a request "
                "names."
            ),
            "input_schema": _object(
                {
                    "query": _string("What to look for, in the catalogue's vocabulary."),
                    "filters": _filters_schema(),
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": config.max_search_results,
                        "description": "Maximum results to return.",
                    },
                },
                ["query"],
            ),
        },
        {
            "name": "get_product_details",
            "description": (
                "Full record for one variant: description, typed specifications, the "
                "capabilities it offers, the requirements it places on other items, and "
                "its sibling variants. Use it for a question about one item, before "
                "comparing finalists, and for any reference shaped like a catalogue id."
            ),
            "input_schema": _object(
                {"variant_id": _string("Catalogue variant_id to look up.")}, ["variant_id"]
            ),
        },
        {
            "name": "check_compatibility",
            "description": (
                "Decide whether two items work together. This is the only source of a "
                "compatibility claim: it evaluates the catalogue's structured "
                "requirement rules and returns each finding with the rule's own "
                "explanation. Never infer compatibility from specifications, titles, or "
                "your own knowledge, and never soften a blocking finding."
            ),
            "input_schema": _object(
                {
                    "base_variant_id": _string(
                        "The item the customer already has or has chosen. " + _VARIANT_ID
                    ),
                    "candidate_variant_id": _string(
                        "The item being checked against it. " + _VARIANT_ID
                    ),
                },
                ["base_variant_id", "candidate_variant_id"],
            ),
        },
        {
            "name": "get_cart",
            "description": (
                "The customer's authoritative cart: lines, quantities, subtotal, and the "
                "state version a mutation must carry."
            ),
            "input_schema": _empty(),
        },
        {
            "name": "get_orders",
            "description": (
                "Recent orders with status and payment state. Use it for a status "
                "question that names no order, and to buy something again."
            ),
            "input_schema": _object(
                {
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "description": "Maximum orders to return.",
                    }
                }
            ),
        },
        {
            "name": "get_order_status",
            "description": (
                "Status, lines, and payment state for one order the customer named. An "
                "order is paid only when this says so; a redirect from Razorpay does not."
            ),
            "input_schema": _object(
                {"order_id": _string("Order id from get_orders.")}, ["order_id"]
            ),
        },
        {
            "name": "search_policies",
            "description": (
                "Search Cartisan's own terms and help content: returns, refunds, "
                "warranty, shipping, and fees. Every statement about the store's terms "
                "comes from a result of this tool in this conversation."
            ),
            "input_schema": _object(
                {"query": _string("The term or topic to look up.")}, ["query"]
            ),
        },
        {
            "name": "get_fulfillment_options",
            "description": (
                "Delivery and pickup options, with their dates and fees, for given "
                "variants at the customer's location."
            ),
            "input_schema": _object(
                {
                    "variant_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 20,
                        "description": "variant_ids to quote options for.",
                    }
                },
                ["variant_ids"],
            ),
        },
        {
            "name": "get_preferences",
            "description": (
                "The customer's profile and saved preferences. Usually already in the "
                "Session context block; call this only when it is missing there."
            ),
            "input_schema": _empty(),
        },
        {
            "name": "recall_memories",
            "description": (
                "Search the customer's saved facts that are not in the Session context "
                "block: older preferences, the devices they already own, recurring "
                "needs. Use it when such a fact would change your recommendation."
            ),
            "input_schema": _object(
                {
                    "topic": {
                        "type": "string",
                        "maxLength": 100,
                        "description": "Topic to search for, in a few words.",
                    }
                },
                ["topic"],
            ),
        },
    ]

    mutations: list[dict[str, Any]] = [
        {
            "name": "add_to_cart",
            "description": (
                "Add a variant the customer chose from something you presented. It takes "
                "an item_ref, not a variant_id: present the options first, then add the "
                "one they picked. If the result says the item is unavailable, say so and "
                "offer an alternative; add that only once the customer chooses it."
            ),
            "input_schema": _object(
                {
                    "item_ref": _string(_ITEM_REF),
                    "quantity": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": config.max_quantity_per_item,
                        "description": "Units to add; omit for one.",
                    },
                    "expected_state_version": {
                        "type": "integer",
                        "minimum": 0,
                        "description": (
                            "The cart state_version you last read. Send it so a cart that "
                            "moved underneath you comes back as a conflict instead of "
                            "overwriting the customer's own change."
                        ),
                    },
                },
                ["item_ref"],
            ),
        },
        {
            "name": "update_cart_item",
            "description": "Set the quantity of a line already in the cart.",
            "input_schema": _object(
                {
                    "variant_id": _string("variant_id of a line already in the cart."),
                    "quantity": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": config.max_quantity_per_item,
                        "description": "New quantity for the line.",
                    },
                    "expected_state_version": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "The cart state_version you last read.",
                    },
                },
                ["variant_id", "quantity"],
            ),
        },
        {
            "name": "remove_from_cart",
            "description": "Remove a line from the cart.",
            "input_schema": _object(
                {
                    "variant_id": _string("variant_id of the line to remove."),
                    "expected_state_version": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "The cart state_version you last read.",
                    },
                },
                ["variant_id"],
            ),
        },
        {
            "name": "stage_checkout",
            "description": (
                "Stage the customer's cart as an immutable, expiring preview they "
                "confirm in the app. It takes no cart and no totals: it reads the "
                "authoritative cart server-side. It places no order, reserves no stock, "
                "creates no payment link, and moves no money, and your text must not "
                "suggest otherwise. Use it as soon as the customer asks to check out, "
                "pay, or complete the purchase."
            ),
            "input_schema": _object(
                {
                    "fulfillment_option": _string(
                        "The method the customer chose, when they chose one.",
                        enum=["delivery", "pickup", "shipping"],
                    ),
                    "note": {
                        "type": "string",
                        "maxLength": 300,
                        "description": "Anything the customer should check before confirming.",
                    },
                }
            ),
        },
    ]

    presentation: list[dict[str, Any]] = [
        {
            "name": "present_products",
            "description": (
                "Show variants from this session's results as cards; the UI fills in "
                "title, price, stock, and image from the catalogue. The server issues an "
                "item_ref for each card, which is what a later add_to_cart names. At most "
                "one card may be marked as a cross-sell, and only for something already "
                "in the cart."
            ),
            "input_schema": _object(
                {
                    "title": _title("set of cards"),
                    "layout": _string(
                        "Card layout; carousel when omitted.",
                        enum=["carousel", "grid", "list"],
                    ),
                    "purpose": _string(
                        "Use setup only when every card is one chosen part of a complete goal-based setup.",
                        enum=["shortlist", "setup"],
                    ),
                    "budget_minor": {
                        "type": "integer", "minimum": 0,
                        "description": "The customer's stated total setup budget in paise. Required for purpose=setup.",
                    },
                    "picks": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 12,
                        "description": "Variants to show, recommended pick first.",
                        "items": _object(
                            {
                                "variant_id": _string(_VARIANT_ID),
                                "reason": _reason(),
                                "is_cross_sell": {
                                    "type": "boolean",
                                    "description": (
                                        "True for the one optional pairing for something "
                                        "already in the cart. Never for a main result."
                                    ),
                                },
                            },
                            ["variant_id"],
                        ),
                    },
                },
                ["picks"],
            ),
        },
        {
            "name": "present_comparison",
            "description": (
                "Compare 2-4 finalists side by side, with pros, cons, and what each is "
                "best for. Use it once the customer has narrowed to them; a fresh "
                "shortlist goes through present_products."
            ),
            "input_schema": _object(
                {
                    "title": _title("comparison"),
                    "entries": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 4,
                        "description": "The finalists being compared.",
                        "items": _object(
                            {
                                "variant_id": _string(_VARIANT_ID),
                                "pros": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "maxItems": 4,
                                    "description": "Short advantages, from tool results.",
                                },
                                "cons": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "maxItems": 3,
                                    "description": "Short drawbacks, from tool results.",
                                },
                                "best_for": {
                                    "type": "string",
                                    "maxLength": 80,
                                    "description": "Who or what this option suits best.",
                                },
                            },
                            ["variant_id"],
                        ),
                    },
                    "recommended_variant_id": _string("The entry you recommend."),
                },
                ["entries"],
            ),
        },
        {
            "name": "present_cart",
            "description": (
                "Show the customer's cart. The UI fills in every line and the subtotal "
                "from the authoritative cart; you supply only the framing."
            ),
            "input_schema": _object(
                {
                    "title": _title("cart card"),
                    "note": {
                        "type": "string",
                        "maxLength": 240,
                        "description": "One line of framing, when the cart needs any.",
                    },
                }
            ),
        },
        {
            "name": "present_checkout",
            "description": (
                "Show the staged checkout preview for confirmation. The UI fills in the "
                "exact items, fulfillment, totals, constraints, and expiry from the "
                "stage; call it in the same round as stage_checkout. It confirms nothing "
                "by itself — the customer does that in the app."
            ),
            "input_schema": _object(
                {
                    "stage_id": _string("stage_id returned by stage_checkout."),
                    "note": {
                        "type": "string",
                        "maxLength": 300,
                        "description": "Anything the customer should check before confirming.",
                    },
                },
                ["stage_id"],
            ),
        },
        {
            "name": "present_order_status",
            "description": (
                "Show the status card for one order; the UI fills in the order and its "
                "payment state. Every answer about where an order stands goes through it."
            ),
            "input_schema": _object(
                {
                    "order_id": _string("Order id from get_orders or get_order_status."),
                    "summary": {
                        "type": "string",
                        "maxLength": 300,
                        "description": "Current state and what happens next, in a sentence.",
                    },
                },
                ["order_id", "summary"],
            ),
        },
        {
            "name": "present_guide",
            "description": (
                "Show know-how as a card of short sections: how to choose between two "
                "kinds of product, what a specification means, how a Cartisan process "
                "works. Use it for guidance that is not a list of products."
            ),
            "input_schema": _object(
                {
                    "title": _title("guide"),
                    "sections": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 8,
                        "description": "Sections of one to three sentences each.",
                        "items": _object(
                            {
                                "heading": {"type": "string", "maxLength": 80},
                                "body": {"type": "string", "maxLength": 600},
                            },
                            ["heading", "body"],
                        ),
                    },
                    "related_variant_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 8,
                        "description": "Variants from this session the guide relates to.",
                    },
                },
                ["title", "sections"],
            ),
        },
        _present_suggestions(_CUSTOMER),
    ]

    return _assemble(config, reads, mutations, presentation)


def build_merchant_tools(
    config: MerchantAgentConfig, skill_names: Sequence[str] = ()
) -> list[dict[str, Any]]:
    """The merchant surface: evidence-backed reads and staged proposals. Nothing here
    applies a change — application is the host's, behind an operator's approval
    (ADR 0016). The reads and staging bodies land in Phase 6; the contract is fixed now
    so the prompt, the cached prefix, and the evaluations have something stable to
    hold."""

    reads: list[dict[str, Any]] = [
        _load_skill(skill_names),
        {
            "name": "get_business_snapshot",
            "description": (
                "Today's observed position: sales, paid orders, average order value, and "
                "the movement against the comparison window. Every figure is observed, "
                "not estimated."
            ),
            "input_schema": _object(
                {
                    "window_days": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 90,
                        "description": "Days to cover; 7 when omitted.",
                    }
                }
            ),
        },
        {
            "name": "query_metrics",
            "description": (
                "One metric over a window, derived from the commerce event log. Returns "
                "the series, the total, and the origins it covers. State a figure only "
                "with the window it came from."
            ),
            "input_schema": _object(
                {
                    "metric": _string(
                        "The metric to read.",
                        enum=[
                            "revenue", "orders", "units", "conversion",
                            "refund_rate", "cart_abandonment",
                        ],
                    ),
                    "window_days": {"type": "integer", "minimum": 1, "maximum": 90},
                    "group_by": _string(
                        "Optional breakdown. Use product or variant to rank best sellers.",
                        enum=["day", "category", "brand", "origin", "product", "variant"],
                    ),
                },
                ["metric"],
            ),
        },
        {
            "name": "search_listings",
            "description": "Search the merchant's own listings by title, brand, or category.",
            "input_schema": _object(
                {
                    "query": _string("What to look for."),
                    "limit": {"type": "integer", "minimum": 1, "maximum": 25},
                },
                ["query"],
            ),
        },
        {
            "name": "get_listing",
            "description": (
                "One listing in full: variants, prices, stock by location, and recent "
                "performance."
            ),
            "input_schema": _object(
                {"product_id": _string("Catalogue product_id.")}, ["product_id"]
            ),
        },
        {
            "name": "get_inventory_alerts",
            "description": (
                "Variants at or below their reorder point, with sellable stock and the "
                "recent sales rate behind each alert."
            ),
            "input_schema": _object(
                {"limit": {"type": "integer", "minimum": 1, "maximum": 50}}
            ),
        },
        {
            "name": "get_unmet_demand",
            "description": (
                "Observed live shopper searches that returned no active catalog result, "
                "grouped by query with request and unique-customer counts. Read this before "
                "claiming shoppers want something the store does not currently offer."
            ),
            "input_schema": _object({
                "window_days": {"type": "integer", "minimum": 1, "maximum": 90},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            }),
        },
        {
            "name": "get_pricing_context",
            "description": (
                "What a price change would be working against for one variant: current "
                "and compare-at price, price history, margin inputs, and recent units."
            ),
            "input_schema": _object(
                {"variant_id": _string("Catalogue variant_id.")}, ["variant_id"]
            ),
        },
        {
            "name": "get_campaign_performance",
            "description": (
                "Campaigns with spend, attributed orders, and attributed revenue. "
                "Attribution here is observed lineage, not a causal claim."
            ),
            "input_schema": _object(
                {
                    "campaign_id": _string("One campaign; omit for all running campaigns."),
                    "window_days": {"type": "integer", "minimum": 1, "maximum": 90},
                }
            ),
        },
        {
            "name": "get_pending_changes",
            "description": "Staged changes awaiting an operator's decision, newest first.",
            "input_schema": _object(
                {"limit": {"type": "integer", "minimum": 1, "maximum": 50}}
            ),
        },
        {
            "name": "get_payment_health",
            "description": (
                "Paytm payment health over a window: verified collections, paid orders, "
                "orders still awaiting payment or verification, stuck orders, and failed "
                "payment attempts by reason. Use it for today's orders, collections, "
                "settlements, or payment problems."
            ),
            "input_schema": _object(
                {"window_days": {"type": "integer", "minimum": 1, "maximum": 90}}
            ),
        },
        {
            "name": "check_restock_financing",
            "description": (
                "Whether the store can afford to restock fast-moving items for the coming "
                "days from its verified Paytm collections: estimated restock cost, cash "
                "from the last 7 days, the shortfall, and, when there is one, a Paytm "
                "merchant loan sized to cover it. Use it when the operator asks whether "
                "they can afford a restock or is preparing for a busy season."
            ),
            "input_schema": _object(
                {"horizon_days": {"type": "integer", "minimum": 7, "maximum": 60,
                                  "description": "Days of sales the restock should cover."},
                 "demand_multiplier": {"type": "number", "minimum": 1, "maximum": 3,
                                       "description": (
                                           "Expected demand against the normal rate, from "
                                           "the operator's words: 2 for a festival they "
                                           "expect to sell double. Default 1; ask when a "
                                           "season is named without a figure, or use 1.5 "
                                           "and say so.")}}
            ),
        },
        {
            "name": "get_recovery_policy",
            "description": (
                "The cart recovery policy in force, if any, and the bounds a new one "
                "must stay within."
            ),
            "input_schema": _empty(),
        },
        {
            "name": "recall_memories",
            "description": (
                "Search the store's saved lessons from the operators' own past decisions, "
                "e.g. which promotion sizes they approve or reject and why. Use it before "
                "proposing a promotion, price change, or recovery offer."
            ),
            "input_schema": _object(
                {"topic": {"type": "string", "maxLength": 100,
                           "description": "Topic to search for, in a few words."}}
            ),
        },
    ]

    def _staging(name: str, what: str, fields: dict[str, Any], required: list[str]) -> dict[str, Any]:
        return {
            "name": name,
            "description": (
                f"Stage {what} for the operator to approve or reject. Staging changes "
                "nothing: the proposal is recorded with its before and after documents "
                "and shown on the approval surface, and only the operator's decision "
                "applies it. Say plainly that it is queued for approval."
            ),
            "input_schema": _object(
                {
                    **fields,
                    "rationale": {
                        "type": "string",
                        "maxLength": 400,
                        "description": (
                            "Why, naming the evidence: the metric, window, and figures "
                            "a read returned this turn."
                        ),
                    },
                },
                [*required, "rationale"],
            ),
        }

    staging: list[dict[str, Any]] = [
        _staging(
            "stage_inventory_action", "a stock movement",
            {
                "variant_id": _string("Catalogue variant_id."),
                "location_id": _string("Inventory location the movement applies to."),
                "action": _string("What to do.", enum=["restock", "adjust", "write_off"]),
                "quantity": {"type": "integer", "minimum": 1, "description": "Units."},
            },
            ["variant_id", "location_id", "action", "quantity"],
        ),
        _staging(
            "stage_price_update", "a price change",
            {
                "variant_id": _string("Catalogue variant_id."),
                "new_price_minor": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Proposed price in paise (₹1 = 100).",
                },
            },
            ["variant_id", "new_price_minor"],
        ),
        _staging(
            "stage_promotion", "a promotion",
            {
                "code": _string("Promotion code.", maxLength=32),
                "description": _string("What the promotion offers.", maxLength=200),
                "discount_kind": _string("How it discounts.", enum=["percentage", "fixed_minor"]),
                "discount_value": {"type": "integer", "minimum": 1},
                "min_subtotal_minor": {"type": "integer", "minimum": 0},
            },
            ["code", "description", "discount_kind", "discount_value"],
        ),
        _staging(
            "stage_campaign", "a campaign",
            {
                "name": _string("Campaign name.", maxLength=80),
                "channel": _string("Where it runs.", maxLength=40),
                "budget_minor": {"type": "integer", "minimum": 0},
                "promotion_code": _string("Promotion it carries, when it carries one."),
            },
            ["name", "channel", "budget_minor"],
        ),
        _staging(
            "stage_listing_update", "a listing edit",
            {
                "product_id": _string("Catalogue product_id."),
                "title": _string("Proposed title.", maxLength=140),
                "description": _string("Proposed description.", maxLength=1200),
                "status": _string("Proposed status.", enum=["draft", "active", "discontinued"]),
            },
            ["product_id"],
        ),
        _staging(
            "stage_loan_request", "a Paytm merchant loan request",
            {
                "amount_minor": {"type": "integer", "minimum": 1,
                                 "description": "Loan amount in paise, at most the eligible "
                                                "limit check_restock_financing returned."},
                "tenure_months": {"type": "integer", "enum": [3, 6, 9, 12],
                                  "description": "Default 6 unless the operator names one."},
                "purpose": _string("What the loan pays for.", maxLength=120),
            },
            ["amount_minor", "tenure_months", "purpose"],
        ),
        _staging(
            "stage_recovery_policy", "a cart recovery policy",
            {
                "abandon_after_minutes": {"type": "integer", "minimum": 30, "maximum": 10080},
                "min_cart_minor": {"type": "integer", "minimum": 0},
                "discount_percentage": {"type": "integer", "minimum": 1, "maximum": 20},
                "max_discount_minor": {"type": "integer", "minimum": 1, "maximum": 200000},
                "cooldown_days": {"type": "integer", "minimum": 7, "maximum": 365},
                "monthly_budget_minor": {"type": "integer", "minimum": 0, "maximum": 5000000},
                "offer_valid_hours": {"type": "integer", "minimum": 1, "maximum": 168},
            },
            ["min_cart_minor", "discount_percentage", "max_discount_minor", "monthly_budget_minor"],
        ),
    ]

    presentation: list[dict[str, Any]] = [
        {
            "name": "present_digest",
            "description": (
                "Show the operator's digest: what moved, what needs attention, and what "
                "you staged. Each line names the read it came from."
            ),
            "input_schema": _object(
                {
                    "title": _title("digest"),
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 8,
                        "items": _object(
                            {
                                "heading": {"type": "string", "maxLength": 80},
                                "body": {"type": "string", "maxLength": 400},
                                "claim_kind": _string(
                                    "What kind of claim this is; observed unless a "
                                    "deterministic estimate or an accepted experiment "
                                    "supports otherwise.",
                                    enum=["observed", "estimated", "causal"],
                                ),
                            },
                            ["heading", "body", "claim_kind"],
                        ),
                    },
                },
                ["title", "items"],
            ),
        },
        {
            "name": "present_metrics",
            "description": (
                "Show a metric the operator asked for; the UI renders the series and the "
                "window from the query_metrics result you name."
            ),
            "input_schema": _object(
                {
                    "title": _title("metrics card"),
                    "metric": _string("The metric, as query_metrics named it."),
                    "window_days": {"type": "integer", "minimum": 1, "maximum": 90},
                    "reading": {
                        "type": "string",
                        "maxLength": 300,
                        "description": "What the series shows, without a causal claim.",
                    },
                },
                ["metric", "window_days", "reading"],
            ),
        },
        {
            "name": "present_change_preview",
            "description": (
                "Show a staged change for approval: the exact before and after documents "
                "as the server recorded them, and the policy checks it passed. The UI "
                "renders the change from its id; you supply only the framing."
            ),
            "input_schema": _object(
                {
                    "change_id": _string("change_id returned by a stage_* tool."),
                    "note": {
                        "type": "string",
                        "maxLength": 300,
                        "description": "What the operator should weigh before deciding.",
                    },
                },
                ["change_id"],
            ),
        },
        _present_suggestions(_OPERATOR),
    ]

    return _assemble(config, reads, staging, presentation)


def _present_suggestions(reader: str) -> dict[str, Any]:
    return {
        "name": "present_suggestions",
        "description": (
            f"Give the turn its 1-4 chips; it ends the reply. Call it in the same round "
            f"as the turn's last component, without waiting for that component's result. "
            f"Alone, after the text, only on a turn with no component. Each chip is "
            f"something {reader} taps instead of typing."
        ),
        "input_schema": _object(
            {
                "suggestions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 4,
                    "description": (
                        "1-4 chips, each a brief imperative and each a different kind of "
                        "step; leave out anything this turn already displayed."
                    ),
                }
            },
            ["suggestions"],
        ),
    }


def _assemble(
    config: CartisanAgentConfig,
    reads: list[dict[str, Any]],
    writes: list[dict[str, Any]],
    presentation: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reads and writes carry the optional `status` line the person waiting sees;
    presentation tools do not, because their card is the status. Anything the config
    switches off, and anything forbidden outright, is left out here as well as refused
    in the executor."""
    absent = config.absent_tools()
    tools = [
        with_status(tool, _OPERATOR if isinstance(config, MerchantAgentConfig) else _CUSTOMER)
        for tool in (*reads, *writes)
        if tool["name"] not in absent
    ]
    tools += [tool for tool in presentation if tool["name"] not in absent]
    return tools


def tool_names(tools: Sequence[dict[str, Any]]) -> list[str]:
    return [str(tool["name"]) for tool in tools]
