"""One handler per merchant boundary tool, over `commerce_common`'s executor frame.

The shape mirrors the shopping executor, and so does the guarantee: `execute`
never raises, and every path out of a tool is one of the five typed outcomes, so
a bound that refuses a proposal reaches the model as something it can work with
and reaches the ledger as something a judge can read.

What is different is the ceiling. Every write handler here ends at
`MerchantPort.stage_change`, which produces a `pending` row and nothing else. The
executor asserts that before it returns: a staging that came back in any other
status is treated as a failure rather than reported as success, because the one
thing this surface must never do is tell an operator a change is live when the
host has not applied it (ADR 0016).

Each staging also carries its own evidence forward. The `before` document is not
the model's idea of the current value — it is read from the port at staging time,
so the proposal the operator approves states what the record actually said when
it was written, and the host can tell at application time whether that is still
true.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from commerce_common.execution import BaseToolExecutor, Handler, clamp_limit
from commerce_common.memory import MEMORY_EXTRACTION_TEMPLATE, MemoryRuntime
from commerce_common.presentation import PresentationExtension, PresentationRefused
from commerce_common.skills import SkillRegistry
from commerce_common.streaming import AgentEvent, ToolOutcome

from .config import FORBIDDEN_TOOLS, MerchantAgentConfig
from .fences import MERCHANT_FENCE
from .gates import FORBIDDEN_GATE, forbidden_error
from .merchant_estimates import (
    price_change_ratio,
    restock_quantity,
    stockout_exposure,
)
from .merchant_gates import (
    LISTING_PROVENANCE_GATE,
    listing_provenance_error,
)
from .merchant_ports import MerchantPort
from .merchant_presentations import MERCHANT_PRESENTATION_COMPONENTS
from .merchant_types import (
    MerchantSessionContext,
    MerchantSessionState,
    StagedChange,
)
from .outcomes import BusinessRefusal, Outcome, Unavailable, refusal, tag
from .types import inr
from marketplace_backend import merchant_finance

# A payments-only merchant (Paytm QR / Soundbox) has no catalogue or stock on Paytm:
# a payment carries an amount, not the items sold. These tools need Paytm POS.
POS_ONLY_TOOLS = frozenset({
    "search_listings", "get_listing", "get_inventory_alerts", "get_pricing_context",
    "stage_inventory_action", "stage_price_update", "stage_listing_update",
    "check_restock_financing",
})
POS_GATE = "paytm_pos_required"
POS_REQUIRED_TEXT = (
    "This store uses Paytm payments only (QR / Soundbox), so there is no catalogue or "
    "stock data to read. Nothing was read or staged. Say so plainly, answer from payment "
    "data where it helps, and mention that connecting Paytm POS unlocks stock alerts, "
    "restocks and pricing."
)

MAX_LISTINGS = 25
MAX_ALERTS = 50
MAX_CHANGES = 50

MERCHANT_MEMORY_EXTRACTION_PROMPT = MEMORY_EXTRACTION_TEMPLATE.format(
    keeper="an apparel and fashion retailer",
    subject="one merchant operator",
    occasions="working sessions",
    speaker="the operator",
    qualifies=(
        "a standing operating preference or constraint the operator stated themselves: a "
        "margin or discount limit they will not go past, a supplier lead time they plan "
        "around, a category they are deliberately running down, a reporting window they "
        "always want."
    ),
    standalone_example=(
        '"never below 15" tells a future reader nothing, while "will not discount audio '
        'accessories by more than 15 percent" tells them everything'
    ),
    live_key_rule=(
        "Keep the operator's one live objective (clearing a category, protecting margin "
        'through a season, launching a line) as a single fact under the key '
        '"current_objective", naming what it is and its horizon; a new objective replaces it.'
    ),
    excluded=(
        "anything that came from the store's own records, metrics, or catalogue; the "
        "mechanics of this session; your own guesses; and any figure a read produced, "
        "which belongs in the records and not in memory."
    ),
)


@dataclass
class MerchantServices:
    """What the handlers and the presentation hooks work through. One field today,
    kept as a container so a later surface (an analysis delegate, a memory store)
    lands beside the port rather than changing every handler's signature."""

    port: MerchantPort


def build_merchant_memory(config: MerchantAgentConfig, store: Any = None) -> MemoryRuntime:
    return MemoryRuntime.build(
        config,
        store,
        fence=MERCHANT_FENCE,
        extraction_prompt=MERCHANT_MEMORY_EXTRACTION_PROMPT,
    )


class MerchantToolExecutor(BaseToolExecutor):
    fence = MERCHANT_FENCE
    components = MERCHANT_PRESENTATION_COMPONENTS
    displayed_text = "Displayed to the operator."
    unavailable_text = (
        "{name} is temporarily unavailable. Work with what you already have, or tell the "
        "operator plainly that you cannot read it right now; do not estimate around it."
    )
    absent_text = (
        "{name} is not something you can do. Say what you can do instead, and do not "
        "describe the action as done or as queued."
    )

    def __init__(
        self,
        *,
        backend: MerchantServices,
        config: MerchantAgentConfig,
        skills: SkillRegistry,
        session: MerchantSessionContext,
        state: MerchantSessionState,
        memory: MemoryRuntime | None = None,
        extensions: Sequence[PresentationExtension] = (),
    ) -> None:
        super().__init__(
            backend=backend,
            config=config,
            skills=skills,
            session=session,
            state=state,
            memory=memory or build_merchant_memory(config),
            extensions=extensions,
        )

    @property
    def memory_subject(self) -> str:
        return self._session.operator_id

    @property
    def port(self) -> MerchantPort:
        return self._backend.port

    def domain_error(self, error: Exception) -> ToolOutcome | None:
        if isinstance(error, BusinessRefusal):
            return refusal(error)
        if isinstance(error, PresentationRefused):
            return tag(
                ToolOutcome.held(error.gate or "presentation", str(error)), Outcome.BLOCKED
            )
        return None

    async def dispatch(self, name: str, tool_input: dict[str, Any]) -> ToolOutcome:
        """A forbidden capability is refused in its own words on this surface too. The
        merchant list is the one that matters most: `apply_change`, `approve_change` and
        `set_price` are exactly the names a model reaches for when it wants to finish
        the job itself (ADR 0016)."""
        if name in FORBIDDEN_TOOLS:
            return tag(ToolOutcome.held(FORBIDDEN_GATE, forbidden_error(name)), Outcome.BLOCKED)
        if name in POS_ONLY_TOOLS and self._session.paytm_plan != "pos":
            return tag(ToolOutcome.held(POS_GATE, POS_REQUIRED_TEXT), Outcome.BLOCKED)
        return await super().dispatch(name, tool_input)

    def handlers(self) -> dict[str, Handler]:
        return {
            "get_business_snapshot": self._get_business_snapshot,
            "query_metrics": self._query_metrics,
            "search_listings": self._search_listings,
            "get_listing": self._get_listing,
            "get_inventory_alerts": self._get_inventory_alerts,
            "get_unmet_demand": self._get_unmet_demand,
            "get_pricing_context": self._get_pricing_context,
            "get_campaign_performance": self._get_campaign_performance,
            "get_pending_changes": self._get_pending_changes,
            "stage_inventory_action": self._stage_inventory_action,
            "stage_price_update": self._stage_price_update,
            "stage_promotion": self._stage_promotion,
            "stage_campaign": self._stage_campaign,
            "stage_listing_update": self._stage_listing_update,
            "get_payment_health": self._get_payment_health,
            "check_restock_financing": self._check_restock_financing,
            "stage_loan_request": self._stage_loan_request,
            "get_recovery_policy": self._get_recovery_policy,
            "stage_recovery_policy": self._stage_recovery_policy,
        }

    # -- payments, financing and recovery -----------------------------------------

    async def _get_payment_health(self, tool_input: dict[str, Any]) -> ToolOutcome:
        health = await self.port.get_payment_health(
            self._session, int(tool_input.get("window_days") or 7))
        return self._fenced({**health, "note": (
            "Only verified Paytm payments are collected money. Name stuck orders and failed "
            "attempts plainly; an order awaiting verification is not revenue yet.")})

    async def _check_restock_financing(self, tool_input: dict[str, Any]) -> ToolOutcome:
        financing = await self.port.get_restock_financing(
            self._session, int(tool_input.get("horizon_days") or 21),
            float(tool_input.get("demand_multiplier") or 1.0))
        self._state.financing = financing
        return self._fenced({**financing, "note": (
            "Say which figures are observed and which are estimated, as claim_kinds states. "
            "When loan is not null, you may offer a Paytm merchant loan of the suggested "
            "amount and stage it with stage_loan_request if the operator wants it; never "
            "apply for it yourself. When loan is null, say the restock can be paid from "
            "collections.")})

    async def _stage_loan_request(self, tool_input: dict[str, Any]) -> ToolOutcome:
        financing = self._state.financing
        if not financing or not financing.get("loan"):
            return tag(ToolOutcome.held(
                "loan_provenance",
                "No loan has been sized this session. Call check_restock_financing first; "
                "a loan is only staged against a limit it returned. Nothing was staged."),
                Outcome.BLOCKED)
        loan = financing["loan"]
        return await self._stage(
            kind="loan_request",
            target_type="paytm_loan",
            target_id=None,
            before={
                "eligible_limit_minor": loan["eligible_limit_minor"],
                "restock_cost_minor": financing["restock_cost_minor"],
                "cash_last_7_days_minor": financing["cash_last_7_days_minor"],
            },
            after=self._loan_after(tool_input, loan),
            rationale=self._sanitize(tool_input.get("rationale"), 400),
        )

    def _loan_after(self, tool_input: dict[str, Any], loan: dict) -> dict[str, Any]:
        amount = int(tool_input.get("amount_minor") or loan["suggested_amount_minor"])
        tenure = int(tool_input.get("tenure_months") or 6)
        return {
            "amount_minor": amount,
            "tenure_months": tenure,
            "monthly_repayment_estimate_minor": merchant_finance.monthly_repayment(amount, tenure),
            "purpose": self._sanitize(tool_input.get("purpose") or "Restock ahead of demand", 120),
        }

    async def _get_recovery_policy(self, tool_input: dict[str, Any]) -> ToolOutcome:
        return self._fenced(await self.port.get_recovery_policy(self._session))

    async def _stage_recovery_policy(self, tool_input: dict[str, Any]) -> ToolOutcome:
        current = (await self.port.get_recovery_policy(self._session)).get("active")
        after = {
            "abandon_after_minutes": int(tool_input.get("abandon_after_minutes") or 120),
            "min_cart_minor": int(tool_input.get("min_cart_minor") or 0),
            "discount_percentage": int(tool_input.get("discount_percentage") or 0),
            "max_discount_minor": int(tool_input.get("max_discount_minor") or 0),
            "cooldown_days": int(tool_input.get("cooldown_days") or 14),
            "monthly_budget_minor": int(tool_input.get("monthly_budget_minor") or 0),
            "offer_valid_hours": int(tool_input.get("offer_valid_hours") or 48),
        }
        return await self._stage(
            kind="recovery_policy",
            target_type="recovery_policy",
            target_id=None,
            before={key: current[key] for key in after} if current else {},
            after=after,
            rationale=self._sanitize(tool_input.get("rationale"), 400),
        )

    # -- reads ---------------------------------------------------------------------

    async def _get_business_snapshot(self, tool_input: dict[str, Any]) -> ToolOutcome:
        window = int(tool_input.get("window_days") or self._config.default_snapshot_days)
        snapshot = await self.port.get_business_snapshot(self._session, window)
        self._state.remember_claims([*snapshot.claims, *snapshot.movements])
        return self._fenced(
            {
                "window_days": snapshot.window_days,
                "currency": snapshot.currency,
                "origins": snapshot.origins,
                "claims": [claim.payload() for claim in snapshot.claims],
                "movements": [claim.payload() for claim in snapshot.movements],
                "limitations": snapshot.limitations,
                "note": (
                    "Every figure here is observed: it was computed from the event log by "
                    "the formula in its basis, over the inputs beside it. A movement is a "
                    "difference between two windows and is not a cause."
                ),
            }
        )

    async def _query_metrics(self, tool_input: dict[str, Any]) -> ToolOutcome:
        metric = str(tool_input.get("metric", ""))
        window = int(tool_input.get("window_days") or self._config.default_metric_days)
        group_by = tool_input.get("group_by")
        series = await self.port.query_metrics(
            self._session, metric, window, str(group_by) if group_by else None
        )
        self._state.remember_series(series)
        return self._fenced(
            {
                **series.model_dump(),
                "total_label": (
                    inr(int(series.total))
                    if series.unit == "INR paise" and series.total is not None
                    else None
                ),
                "note": (
                    f"Present this with present_metrics using metric={series.metric!r} and "
                    f"window_days={series.window_days}; the card renders these points. State "
                    "the figure with the window it came from."
                ),
            }
        )

    async def _search_listings(self, tool_input: dict[str, Any]) -> ToolOutcome:
        query = self._sanitize(tool_input.get("query", ""), 200)
        listings = await self.port.search_listings(
            self._session, query, clamp_limit(tool_input.get("limit"), 10, MAX_LISTINGS)
        )
        for listing in listings:
            self._state.remember_listing(listing)
        if not listings:
            return ToolOutcome(
                f"No listing matched {query!r}. Nothing was found; do not describe a "
                "listing the catalogue did not return."
            )
        return self._fenced(
            {
                "query": query,
                "count": len(listings),
                "results": [_listing_payload(listing) for listing in listings],
            }
        )

    async def _get_listing(self, tool_input: dict[str, Any]) -> ToolOutcome:
        product_id = str(tool_input.get("product_id", ""))
        listing = await self.port.get_listing(self._session, product_id)
        if listing is None:
            return ToolOutcome.error(f"No listing with product_id {product_id}.")
        self._state.remember_listing(listing)
        self._state.remember_variants(listing.variants)
        return self._fenced(
            {
                **_listing_payload(listing),
                "description": listing.description,
                "units_sold": listing.units_sold,
                "revenue_minor": listing.revenue_minor,
                "revenue": inr(listing.revenue_minor),
                "window_days": listing.window_days,
                "variants": [_variant_payload(variant) for variant in listing.variants],
                "note": (
                    "A listing has no price or stock of its own; its variants do. Price and "
                    "restock by variant_id."
                ),
            }
        )

    async def _get_inventory_alerts(self, tool_input: dict[str, Any]) -> ToolOutcome:
        alerts = await self.port.get_inventory_alerts(
            self._session, clamp_limit(tool_input.get("limit"), 10, MAX_ALERTS)
        )
        payloads = []
        for alert in alerts:
            # The estimates that go with an alert are computed here, once, and remembered:
            # a restock the model proposes has to be sized by the same arithmetic the card
            # showed, not by a number it liked the look of.
            restock = restock_quantity(
                variant_id=alert.variant_id, sellable=alert.sellable,
                units_sold=alert.units_sold, window_days=alert.window_days)
            claims = [claim for claim in (alert.days_of_cover, restock) if claim is not None]
            self._state.remember_claims(claims)
            # An alert is a catalogue read: it names the variant and carries its real
            # levels, which is precisely what a staged restock needs behind it.
            self._state.remember_variants([_alert_variant(alert)])
            payloads.append(
                {
                    "variant_id": alert.variant_id,
                    "product_id": alert.product_id,
                    "title": alert.title,
                    "sellable": alert.sellable,
                    "on_hand": alert.on_hand,
                    "reserved": alert.reserved,
                    "location_id": alert.location_id,
                    "units_sold": alert.units_sold,
                    "window_days": alert.window_days,
                    "estimates": [claim.payload() for claim in claims],
                }
            )
        if not payloads:
            return ToolOutcome(
                "No active variant is below its cover threshold. Nothing needs restocking "
                "on this reading; say so rather than looking for something to flag."
            )
        return self._fenced(
            {
                "alerts": payloads,
                "note": (
                    "Stock and units sold are observed. Days of cover and the restock "
                    "quantity are estimates: they are the formula in each `basis` applied "
                    "to the inputs beside it, and they carry the limitations shown. Say "
                    "which is which."
                ),
            }
        )

    async def _get_unmet_demand(self, tool_input: dict[str, Any]) -> ToolOutcome:
        window = int(tool_input.get("window_days") or 30)
        signals = await self.port.get_unmet_demand(
            self._session, window, clamp_limit(tool_input.get("limit"), 10, 50)
        )
        return self._fenced({
            "window_days": window,
            "origins": ["live_app"],
            "signals": [signal.model_dump() for signal in signals],
            "basis": "Count of authoritative catalog searches that returned no active result, grouped by normalized query.",
            "limitations": [
                "A no-result search records unmet interest, not a promised sale.",
                "Different wording for the same need may appear as separate signals.",
            ],
        })

    async def _get_pricing_context(self, tool_input: dict[str, Any]) -> ToolOutcome:
        variant_id = str(tool_input.get("variant_id", ""))
        context = await self.port.get_pricing_context(self._session, variant_id)
        if context is None:
            return ToolOutcome.error(f"No catalogue variant with id {variant_id}.")
        # A pricing context is what a staging is written against, so the variant it
        # describes counts as read for the provenance gate.
        self._state.seen_variants.setdefault(variant_id, _stub_variant(context))
        exposure = stockout_exposure(
            variant_id=variant_id, price_minor=context.current_price_minor,
            units_sold=context.units_sold, window_days=context.window_days,
            sellable=context.sellable)
        self._state.remember_claims([exposure])
        return self._fenced(
            {
                "variant_id": context.variant_id,
                "product_id": context.product_id,
                "title": context.title,
                "currency": context.currency,
                "current_price_minor": context.current_price_minor,
                "current_price": inr(context.current_price_minor),
                "compare_at_minor": context.compare_at_minor,
                "history": [entry.model_dump() for entry in context.history],
                "units_sold": context.units_sold,
                "window_days": context.window_days,
                "sellable": context.sellable,
                "bounds": {
                    "max_change_ratio": context.max_change_ratio,
                    "floor_minor": context.floor_minor,
                    "ceiling_minor": context.ceiling_minor,
                    "floor": inr(context.floor_minor),
                    "ceiling": inr(context.ceiling_minor),
                },
                "estimates": [exposure.payload()],
                "limitations": context.limitations,
                "note": (
                    "A price outside the floor and ceiling above will be refused when you "
                    "stage it. These are policy bounds on the size of one change, not "
                    "margin floors: Cartisan records no cost."
                ),
            }
        )

    async def _get_campaign_performance(self, tool_input: dict[str, Any]) -> ToolOutcome:
        campaign_id = tool_input.get("campaign_id")
        window = int(tool_input.get("window_days") or self._config.default_metric_days)
        campaigns = await self.port.get_campaign_performance(
            self._session, str(campaign_id) if campaign_id else None, window
        )
        if not campaigns:
            raise Unavailable(
                "No campaign matched. Nothing was found, so no campaign result can be "
                "stated here."
            )
        for campaign in campaigns:
            self._state.remember_claims(campaign.claims)
        return self._fenced(
            {
                "campaigns": [
                    {
                        "campaign_id": campaign.campaign_id,
                        "name": campaign.name,
                        "channel": campaign.channel,
                        "status": campaign.status,
                        "budget_minor": campaign.budget_minor,
                        "budget": inr(campaign.budget_minor),
                        "spend_minor": campaign.spend_minor,
                        "spend": inr(campaign.spend_minor),
                        "promotion_code": campaign.promotion_code,
                        "promotion_description": campaign.promotion_description,
                        "claims": [claim.payload() for claim in campaign.claims],
                        "limitations": campaign.limitations,
                    }
                    for campaign in campaigns
                ],
                "note": (
                    "Spend and budget are observed. Attribution is by promotion redemption: "
                    "an attributed order is one that redeemed this campaign's promotion code "
                    "inside the campaign's own window, which Cartisan records. It is not "
                    "recorded anywhere in Cartisan which orders the campaign *caused*: there "
                    "is no impression or click, so never describe attributed revenue as lift, "
                    "incremental revenue, ROI, or a change the campaign caused. A campaign "
                    "with no promotion code has no attribution at all — say the orders are "
                    "not connected to it rather than reporting zero for it."
                ),
            }
        )

    async def _get_pending_changes(self, tool_input: dict[str, Any]) -> ToolOutcome:
        changes = await self.port.get_pending_changes(
            self._session, clamp_limit(tool_input.get("limit"), 10, MAX_CHANGES)
        )
        for change in changes:
            self._state.remember_change(change)
        return self._fenced(
            {
                "pending": [_change_payload(change) for change in changes],
                "note": (
                    "These are waiting on the operator. You cannot approve, reject, or "
                    f"apply one — that happens on {self._config.approval_surface}."
                ),
            }
        )

    # -- staging: the ceiling ------------------------------------------------------

    async def _stage_inventory_action(self, tool_input: dict[str, Any]) -> ToolOutcome:
        variant_id = self._known_variant("stage_inventory_action", tool_input.get("variant_id"))
        action = str(tool_input.get("action") or "restock")
        quantity = int(tool_input.get("quantity") or 0)
        if quantity <= 0:
            raise BusinessRefusal(
                "An inventory action needs a positive quantity. Nothing was staged.",
                gate="merchant_policy")
        # A write-off or a downward adjustment removes units, so the signed figure the
        # policy checks is derived here rather than taken from the model.
        units = quantity if action == "restock" else -quantity
        variant = self._state.seen_variants[variant_id]
        if not variant.levels_known:
            raise BusinessRefusal(
                f"Stock for {variant_id} has not been read in this conversation, so a "
                "staged movement would state a starting level it does not know. Call "
                "get_inventory_alerts or get_listing first. Nothing was staged.",
                gate=LISTING_PROVENANCE_GATE)
        return await self._stage(
            kind="inventory_action",
            target_type="catalog_variant",
            target_id=variant_id,
            before={"on_hand": variant.on_hand, "reserved": variant.reserved},
            after={"units": units, "action": action,
                   "location_id": str(tool_input.get("location_id") or "")},
            rationale=self._sanitize(tool_input.get("rationale"), 400),
        )

    async def _stage_price_update(self, tool_input: dict[str, Any]) -> ToolOutcome:
        variant_id = self._known_variant("stage_price_update", tool_input.get("variant_id"))
        proposed = int(tool_input.get("new_price_minor") or 0)
        if proposed <= 0:
            raise BusinessRefusal(
                "A price update needs a positive price in paise. Nothing was staged.",
                gate="merchant_policy")
        # The current price is read now, not taken from the call: the proposal has to
        # state what the record actually said when it was written, because that is what
        # the host revalidates against at application time.
        context = await self.port.get_pricing_context(self._session, variant_id)
        if context is None:
            raise Unavailable(f"variant_id {variant_id} is no longer in the catalogue.")
        ratio = price_change_ratio(
            current_minor=context.current_price_minor, proposed_minor=proposed)
        self._state.remember_claims([ratio])
        return await self._stage(
            kind="price_update",
            target_type="catalog_variant",
            target_id=variant_id,
            before={"amount_minor": context.current_price_minor},
            after={"amount_minor": proposed},
            rationale=self._sanitize(tool_input.get("rationale"), 400),
            extra={"change_ratio": ratio.payload()},
        )

    async def _stage_promotion(self, tool_input: dict[str, Any]) -> ToolOutcome:
        return await self._stage(
            kind="promotion",
            target_type="promotion",
            target_id=None,
            before={},
            after={
                "code": self._sanitize(tool_input.get("code"), 32),
                "description": self._sanitize(tool_input.get("description"), 200),
                "discount_kind": str(tool_input.get("discount_kind") or ""),
                "discount_value": int(tool_input.get("discount_value") or 0),
                "min_subtotal_minor": int(tool_input.get("min_subtotal_minor") or 0),
            },
            rationale=self._sanitize(tool_input.get("rationale"), 400),
        )

    async def _stage_campaign(self, tool_input: dict[str, Any]) -> ToolOutcome:
        return await self._stage(
            kind="campaign",
            target_type="campaign",
            target_id=None,
            before={},
            after={
                "name": self._sanitize(tool_input.get("name"), 80),
                "channel": self._sanitize(tool_input.get("channel"), 40),
                "budget_minor": int(tool_input.get("budget_minor") or 0),
                "promotion_code": self._sanitize(tool_input.get("promotion_code"), 32) or None,
            },
            rationale=self._sanitize(tool_input.get("rationale"), 400),
        )

    async def _stage_listing_update(self, tool_input: dict[str, Any]) -> ToolOutcome:
        product_id = str(tool_input.get("product_id", ""))
        listing = self._state.seen_listings.get(product_id)
        if listing is None:
            raise BusinessRefusal(
                listing_provenance_error("stage_listing_update", product_id or "no product_id"),
                gate=LISTING_PROVENANCE_GATE)
        details = await self.port.get_listing(self._session, product_id)
        if details is None:
            raise Unavailable(f"product_id {product_id} is no longer in the catalogue.")
        after: dict[str, Any] = {}
        before: dict[str, Any] = {}
        for field, live, limit in (
            ("title", details.title, 140),
            ("description", details.description, 1200),
        ):
            if tool_input.get(field) is not None:
                after[field] = self._sanitize(tool_input.get(field), limit)
                before[field] = live
        if tool_input.get("status") is not None:
            after["status"] = str(tool_input["status"])
            before["status"] = details.status
        if not after:
            raise BusinessRefusal(
                "A listing update has to change the title, the description, or the "
                "status. Nothing was staged.", gate="merchant_policy")
        return await self._stage(
            kind="listing_update",
            target_type="catalog_product",
            target_id=product_id,
            before=before,
            after=after,
            rationale=self._sanitize(tool_input.get("rationale"), 400),
        )

    # -- helpers -------------------------------------------------------------------

    def _known_variant(self, name: str, raw: Any) -> str:
        """A variant id a catalogue read returned this session, or a refusal. This is
        the provenance gate on the staging side: a change cannot be proposed against a
        record the model has not looked at."""
        variant_id = str(raw or "")
        if variant_id not in self._state.seen_variants:
            raise BusinessRefusal(
                listing_provenance_error(name, variant_id or "no variant_id"),
                gate=LISTING_PROVENANCE_GATE)
        return variant_id

    async def _stage(
        self,
        *,
        kind: str,
        target_type: str,
        target_id: str | None,
        before: dict,
        after: dict,
        rationale: str,
        extra: dict[str, Any] | None = None,
    ) -> ToolOutcome:
        if not rationale.strip():
            raise BusinessRefusal(
                "A staged change needs a rationale naming the evidence behind it — the "
                "metric, the window, and the figures a read returned this turn. Nothing "
                "was staged.", gate="merchant_policy")
        change = await self.port.stage_change(
            self._session, kind=kind, target_type=target_type, target_id=target_id,
            before=before, after=after, rationale=rationale)
        # The one invariant this surface exists to hold. A port that returned anything
        # but `pending` has applied something, and the right answer is to fail loudly
        # rather than to narrate it as queued (ADR 0016).
        if change.status != "pending":
            raise RuntimeError(
                f"staging produced status {change.status!r}; staging may only ever "
                "produce 'pending'")
        self._state.remember_change(change)
        payload = {
            **_change_payload(change),
            "note": (
                "Queued for approval and nothing more: no price, stock level, promotion, "
                "campaign, or listing has changed. The operator decides on "
                f"{self._config.approval_surface}, and Cartisan re-checks the bounds "
                "against current figures before anything is applied."
            ),
        }
        if extra:
            payload.update(extra)
        # The approval surface is a different pane from the conversation, so it is told
        # directly rather than waiting for the operator to reload it.
        return tag(
            self._fenced(payload, [AgentEvent.change_update(_change_payload(change))]),
            Outcome.APPLIED,
        )


def _listing_payload(listing: Any) -> dict[str, Any]:
    return {
        "product_id": listing.product_id,
        "title": listing.title,
        "brand": listing.brand,
        "category": listing.category,
        "status": listing.status,
        "origin": listing.origin,
        "from_price_minor": listing.from_price_minor,
        "from_price": inr(listing.from_price_minor),
        "total_sellable": listing.total_sellable,
        "variant_count": listing.variant_count,
    }


def _variant_payload(variant: Any) -> dict[str, Any]:
    return {
        "variant_id": variant.variant_id,
        "sku": variant.sku,
        "title": variant.title,
        "options": variant.options,
        "status": variant.status,
        "price_minor": variant.price_minor,
        "price": inr(variant.price_minor),
        "compare_at_minor": variant.compare_at_minor,
        "sellable": variant.sellable,
        "on_hand": variant.on_hand,
        "reserved": variant.reserved,
    }


def _change_payload(change: StagedChange) -> dict[str, Any]:
    return {
        "change_id": change.change_id,
        "kind": change.kind,
        "target_type": change.target_type,
        "target_id": change.target_id,
        "status": change.status,
        "before": change.before,
        "after": change.after,
        "rationale": change.rationale,
        "created_at": change.created_at,
        "decided_at": change.decided_at,
        "applied_at": change.applied_at,
    }


def _alert_variant(alert: Any) -> Any:
    """The variant an inventory alert describes, in the shape the provenance map holds.
    Its levels are the ones the alert reported, so a movement staged from it states a
    starting level that was actually read."""
    from .merchant_types import ListingVariant

    return ListingVariant(
        variant_id=alert.variant_id, sku=alert.variant_id, title=alert.title, options={},
        status="active", price_minor=0, sellable=alert.sellable, on_hand=alert.on_hand,
        reserved=alert.reserved)


def _stub_variant(context: Any) -> Any:
    """The variant a pricing context describes, in the shape the provenance map holds.
    Built from the context rather than re-queried: it is the same read."""
    from .merchant_types import ListingVariant

    return ListingVariant(
        variant_id=context.variant_id, sku=context.variant_id, title=context.title,
        options={}, status="active", price_minor=context.current_price_minor,
        compare_at_minor=context.compare_at_minor, sellable=context.sellable,
        on_hand=context.sellable, reserved=0, levels_known=False)
