"""The merchant runtime's one integration surface.

Every method acts for the operator in `session`, whose identity came from a
verified Supabase `merchant_operator` principal (ADR 0010). The shape of this
port is the whole of ADR 0016 expressed in types: there are reads, there is
`stage_change`, and there is nothing else. No method approves, applies,
refunds, sends a campaign, or writes a price — those verbs do not exist here to
be called, so no future executor bug can reach one.

Keeping it a port is what lets the transcript evaluations drive the real turn
loop, the real gates and the real change ledger against a scripted model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .outcomes import Unavailable

from .merchant_types import (
    BusinessSnapshot,
    CampaignPerformance,
    InventoryAlert,
    Listing,
    ListingDetails,
    MerchantSessionContext,
    MetricSeries,
    PricingContext,
    StagedChange,
    UnmetDemandSignal,
)


class MerchantPort(ABC):
    # -- payments and financing (optional; a port without them says so) --------

    async def get_payment_health(self, session: MerchantSessionContext, window_days: int = 7) -> dict:
        raise Unavailable("payment health is not available on this store")

    async def get_restock_financing(self, session: MerchantSessionContext, horizon_days: int = 21,
                                    demand_multiplier: float = 1.0) -> dict:
        raise Unavailable("restock financing is not available on this store")

    async def get_recovery_policy(self, session: MerchantSessionContext) -> dict:
        raise Unavailable("cart recovery is not available on this store")

    # -- evidence-backed reads -------------------------------------------------

    @abstractmethod
    async def get_business_snapshot(
        self, session: MerchantSessionContext, window_days: int = 7
    ) -> BusinessSnapshot:
        """Observed position over the window, each figure carrying its formula and
        operands, plus movement against the window before it."""

    @abstractmethod
    async def query_metrics(
        self,
        session: MerchantSessionContext,
        metric: str,
        window_days: int = 30,
        group_by: str | None = None,
    ) -> MetricSeries:
        """One metric derived from `commerce_events` (ADR 0018). Raises `Unavailable`
        for a metric the event log cannot support, rather than returning a zero."""

    @abstractmethod
    async def search_listings(
        self, session: MerchantSessionContext, query: str, limit: int = 10
    ) -> list[Listing]:
        """The merchant's own listings, including drafts and discontinued ones: the
        operator's catalogue is not the shopper's."""

    @abstractmethod
    async def get_listing(
        self, session: MerchantSessionContext, product_id: str
    ) -> ListingDetails | None:
        """One listing in full — variants, prices, stock by location, recent units —
        or None when the id is unknown."""

    @abstractmethod
    async def get_inventory_alerts(
        self, session: MerchantSessionContext, limit: int = 10
    ) -> list[InventoryAlert]:
        """Variants at or below cover, with the observed sales rate behind each one."""

    @abstractmethod
    async def get_unmet_demand(
        self, session: MerchantSessionContext, window_days: int = 30, limit: int = 10
    ) -> list[UnmetDemandSignal]:
        """Observed customer searches that returned no active catalogue result."""

    @abstractmethod
    async def get_pricing_context(
        self, session: MerchantSessionContext, variant_id: str
    ) -> PricingContext | None:
        """What a price change on one variant would be working against."""

    @abstractmethod
    async def get_campaign_performance(
        self,
        session: MerchantSessionContext,
        campaign_id: str | None = None,
        window_days: int = 30,
    ) -> list[CampaignPerformance]:
        """Campaigns with spend and budget observed, and attribution reported only as
        far as the records actually link it (ADR 0019)."""

    @abstractmethod
    async def get_pending_changes(
        self, session: MerchantSessionContext, limit: int = 10
    ) -> list[StagedChange]:
        """Staged changes awaiting an operator's decision."""

    @abstractmethod
    async def read_change(
        self, session: MerchantSessionContext, change_id: str
    ) -> StagedChange | None:
        """One staged change, for the preview card to render from the record."""

    # -- the only write, and it writes a proposal --------------------------------

    @abstractmethod
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
        """Record a proposal for the operator to decide on. The result is always
        `pending`. Raises `BusinessRefusal` when policy bounds refuse it; nothing in
        the catalogue, the inventory, or the price tables moves either way."""
