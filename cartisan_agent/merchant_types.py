"""The merchant runtime's domain models.

Two things separate these from the shopping types. The principal is an operator,
not a customer, and every number carries the kind of claim it is (ADR 0017): a
figure the event log measured is `observed`, a figure a deterministic formula
produced from observed inputs is `estimated`, and `causal` is reserved for an
accepted experiment. Nothing in Cartisan produces the last one, so nothing in
this module can construct one — the enum exists so a model's attempt to make a
causal claim has a name to be refused under.

Money is integer minor units (paise), as everywhere else; the rupee label rides
alongside so the model quotes it rather than doing the arithmetic.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .types import SessionContext, inr

ClaimKind = Literal["observed", "estimated", "causal"]

OBSERVED: ClaimKind = "observed"
ESTIMATED: ClaimKind = "estimated"
CAUSAL: ClaimKind = "causal"


class MerchantSessionContext(SessionContext):
    """One operator's turn. The id comes from a verified `merchant_operator`
    principal, and no tool takes an operator id from the model."""

    surface: Literal["shopping", "merchant"] = "merchant"

    @property
    def operator_id(self) -> str:
        return self.customer_id


class Claim(BaseModel):
    """A number with everything needed to check it.

    `basis` is the formula in words, `inputs` are the operands it was computed
    from, and `limitations` are what the figure cannot support. A claim that
    cannot show its inputs is not a claim this runtime will make.
    """

    key: str
    value: float | int | None
    unit: str
    claim_kind: ClaimKind = OBSERVED
    basis: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)

    @property
    def label(self) -> str | None:
        """The rupee label, for a paise figure only."""
        return inr(int(self.value)) if self.unit == "INR paise" and self.value is not None else None

    def payload(self) -> dict[str, Any]:
        body = self.model_dump()
        if (label := self.label) is not None:
            body["value_label"] = label
        return body


class MetricPoint(BaseModel):
    date: str
    value: float | int
    orders: int | None = None
    units: int | None = None
    # Grouped series use ``date`` as their display label for compatibility with
    # the chart component.  Catalogue breakdowns also carry the stable id so an
    # agent can identify the winning product/variant without guessing from text.
    bucket_id: str | None = None


class MetricSeries(BaseModel):
    """One metric over one window, with the origins it covers spelled out: a seeded
    ninety-day history and a live demo purchase are never silently summed (ADR 0032)."""

    metric: str
    window_days: int
    group_by: str | None = None
    unit: str
    origins: list[str]
    points: list[MetricPoint] = Field(default_factory=list)
    total: float | int | None = None
    claim_kind: ClaimKind = OBSERVED
    basis: str
    limitations: list[str] = Field(default_factory=list)


class BusinessSnapshot(BaseModel):
    window_days: int
    currency: str = "INR"
    origins: list[str]
    claims: list[Claim] = Field(default_factory=list)
    comparison_window_days: int | None = None
    movements: list[Claim] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class ListingVariant(BaseModel):
    variant_id: str
    sku: str
    title: str
    options: dict[str, str] = Field(default_factory=dict)
    status: str
    price_minor: int
    compare_at_minor: int | None = None
    sellable: int
    on_hand: int
    reserved: int
    reorder_point: int | None = None
    # False when this record came from a read that does not carry stock (a pricing
    # context). A staged inventory action needs real levels in its `before` document,
    # so it refuses a variant whose stock was never actually read.
    levels_known: bool = True

    @property
    def price(self) -> str:
        return inr(self.price_minor)


class Listing(BaseModel):
    """A catalogue product as the operator works on it. Price and stock are the
    lowest and the sum across its variants, because a product is not the thing that
    has either — a variant is."""

    product_id: str
    title: str
    brand: str
    category: str | None = None
    status: str
    origin: str = "seeded"
    from_price_minor: int
    total_sellable: int
    variant_count: int


class ListingDetails(Listing):
    description: str = ""
    variants: list[ListingVariant] = Field(default_factory=list)
    units_sold: int = 0
    revenue_minor: int = 0
    window_days: int = 30


class InventoryAlert(BaseModel):
    variant_id: str
    product_id: str
    title: str
    sellable: int
    on_hand: int
    reserved: int
    location_id: str | None = None
    units_sold: int = 0
    window_days: int = 30
    days_of_cover: Claim | None = None


class UnmetDemandSignal(BaseModel):
    query: str
    requests: int
    unique_customers: int
    first_seen: str
    last_seen: str
    window_days: int
    origin: str = "live_app"


class PriceHistoryEntry(BaseModel):
    amount_minor: int
    price_kind: str
    valid_from: str
    valid_to: str | None = None


class PricingContext(BaseModel):
    variant_id: str
    product_id: str
    title: str
    currency: str = "INR"
    current_price_minor: int
    compare_at_minor: int | None = None
    history: list[PriceHistoryEntry] = Field(default_factory=list)
    units_sold: int = 0
    revenue_minor: int = 0
    window_days: int = 30
    sellable: int = 0
    # The bound the staging tool will enforce, stated up front so the model proposes
    # something that can pass rather than learning the limit from a refusal.
    max_change_ratio: float = 0.25
    floor_minor: int = 0
    ceiling_minor: int = 0
    limitations: list[str] = Field(default_factory=list)


class CampaignPerformance(BaseModel):
    campaign_id: str
    name: str
    channel: str
    status: str
    budget_minor: int
    spend_minor: int
    promotion_code: str | None = None
    promotion_description: str | None = None
    claims: list[Claim] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class StagedChange(BaseModel):
    """One row of `merchant_changes` as the operator and the model both see it. The
    status is the record's, never the model's: `stage_*` can only ever produce
    `pending` (ADR 0016)."""

    change_id: str
    kind: str
    target_type: str
    target_id: str | None = None
    status: str
    before: dict[str, Any] = Field(default_factory=dict)
    after: dict[str, Any] = Field(default_factory=dict)
    rationale: str
    created_at: str
    decided_at: str | None = None
    applied_at: str | None = None
    policy_checks: dict[str, Any] = Field(default_factory=dict)


class MerchantSessionState(BaseModel):
    """Per-conversation provenance the merchant gates read.

    Everything here answers one question: can this claim, this id, or this preview
    be traced to something the server produced in this session? A model-supplied
    value that is not in one of these maps has no lineage, and the gate refuses it.
    """

    seen_listings: dict[str, Listing] = Field(default_factory=dict)      # product_id
    seen_variants: dict[str, ListingVariant] = Field(default_factory=dict)
    read_metrics: dict[str, MetricSeries] = Field(default_factory=dict)  # f"{metric}:{days}"
    read_claims: dict[str, Claim] = Field(default_factory=dict)          # claim key
    staged_changes: dict[str, StagedChange] = Field(default_factory=dict)
    # Set by the loop when the operator's message is about business performance, so
    # the turn reads before it describes.
    performance_turn: bool = False

    def remember_listing(self, listing: Listing) -> None:
        self.seen_listings[listing.product_id] = listing

    def remember_variants(self, variants: list[ListingVariant]) -> None:
        for variant in variants:
            self.seen_variants[variant.variant_id] = variant

    def remember_series(self, series: MetricSeries) -> None:
        self.read_metrics[f"{series.metric}:{series.window_days}"] = series

    def remember_claims(self, claims: list[Claim]) -> None:
        for claim in claims:
            self.read_claims[claim.key] = claim

    def remember_change(self, change: StagedChange) -> None:
        self.staged_changes[change.change_id] = change
