"""The merchant agent's turn loop (ADR 0014).

The loop is `runtime.AgentRuntime`, the same one shopping runs on. What is here
is what makes a turn a *merchant* turn: the frozen merchant tool array and its
prompt halves, the store snapshot prefetched into the context block, and the
deterministic rule that a question about how the business is doing reads before
it answers.

This is what replaces the pre-Phase-4 `/chat/portal` path. That endpoint asked a
model for one JSON object containing a reply and at most one proposal, then
revalidated the proposal in code. It worked, and it could not do the job this
phase needs: a single shot cannot read a metric, notice what the metric implies,
look up the listing behind it, and stage a change against the figures it just
read — and without those steps there is no evidence lineage to point at, only a
proposal and a promise.

    runtime = CartisanMerchantRuntime(services=..., store=...)
    async for event in runtime.stream_turn(messages, session, state):
        ...
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic

from commerce_common.grounding import first_forced_tool
from commerce_common.memory import MemoryRuntime
from commerce_common.skills import SkillRegistry
from marketplace_backend.store import Store

from .config import MerchantAgentConfig
from .contracts import build_merchant_tools
from .merchant_executor import (
    MerchantServices,
    MerchantToolExecutor,
    build_merchant_memory,
)
from .merchant_gates import MERCHANT_GROUNDING_RULES, is_performance_turn
from .merchant_presentations import MERCHANT_PRESENTATION_COMPONENTS
from .merchant_prompts import build_merchant_context, build_merchant_static_system
from .merchant_types import MerchantSessionContext, MerchantSessionState
from .runtime import AgentRuntime, fetched
from .turns import TurnStore
from .versions import MERCHANT_PROMPT_VERSION

MERCHANT_SKILLS_DIR = Path(__file__).resolve().parent / "merchant_skills"


class CartisanMerchantRuntime(AgentRuntime):
    prompt_version = MERCHANT_PROMPT_VERSION

    def __init__(
        self,
        *,
        services: MerchantServices,
        store: Store,
        config: MerchantAgentConfig | None = None,
        skills: SkillRegistry | None = None,
        skills_dir: Path | None = MERCHANT_SKILLS_DIR,
        client: AsyncAnthropic | None = None,
        memory: MemoryRuntime | None = None,
        turns: TurnStore | None = None,
    ) -> None:
        config = config or MerchantAgentConfig()
        skills = skills or (
            SkillRegistry.from_dir(skills_dir)
            if skills_dir and skills_dir.exists()
            else SkillRegistry([])
        )
        self.services = services
        super().__init__(
            store=store,
            config=config,
            skills=skills,
            components=MERCHANT_PRESENTATION_COMPONENTS,
            tools=build_merchant_tools(config, skills.names),
            static_system=build_merchant_static_system(config, skills),
            memory=memory or build_merchant_memory(config),
            client=client,
            turns=turns,
        )

    # -- what a merchant turn supplies --------------------------------------------

    def new_state(self) -> MerchantSessionState:
        return MerchantSessionState()

    def before_turn(self, state: MerchantSessionState, user_text: str) -> None:
        state.performance_turn = is_performance_turn(self.config, user_text)

    def build_executor(
        self, session: MerchantSessionContext, state: MerchantSessionState
    ) -> MerchantToolExecutor:
        return MerchantToolExecutor(
            backend=self.services,
            config=self.config,
            skills=self.skills,
            session=session,
            state=state,
            memory=self.memory,
        )

    async def dynamic_context(self, session: MerchantSessionContext) -> str:
        store_context, facts = await self._prefetch(session)
        return build_merchant_context(
            operator_name=None,
            paytm_plan=session.paytm_plan,
            channel=session.channel,
            store_context=store_context,
            memory_facts=list(facts or []),
            now=session.local_now(),
            max_chars=self.config.max_context_chars * 3,
        )

    def forced_tool(self, state: MerchantSessionState, user_text: str) -> str | None:
        return first_forced_tool(MERCHANT_GROUNDING_RULES, self.config, user_text, state)

    def target_type(self, name: str) -> str | None:
        if name.startswith("stage_") or name in {"get_pending_changes", "present_change_preview"}:
            return "merchant_change"
        if name in {"get_listing", "search_listings", "stage_listing_update"}:
            return "catalog_product"
        if name in {"get_pricing_context", "get_inventory_alerts"}:
            return "catalog_variant"
        if name in {"get_business_snapshot", "query_metrics", "get_unmet_demand", "present_metrics", "present_digest"}:
            return "metrics"
        if name == "get_campaign_performance":
            return "campaign"
        return None

    def target_id(self, arguments: dict[str, Any]) -> str | None:
        for key in ("variant_id", "product_id", "change_id", "campaign_id", "code", "metric"):
            if arguments.get(key):
                return str(arguments[key])
        return None

    # -- internals ----------------------------------------------------------------

    async def _prefetch(self, session: MerchantSessionContext) -> tuple[Any, list[Any]]:
        """The headline position and the operator's saved constraints, both optional.

        The snapshot goes into the context block so an opening line has something
        behind it, but it does not stand in for a read: the model still calls
        get_business_snapshot before quoting a figure, because the block carries no
        formula and no inputs, and a claim without those is not one this surface makes.
        """
        snapshot, facts = await asyncio.gather(
            fetched(
                self.services.port.get_business_snapshot(
                    session, self.config.default_snapshot_days
                )
            ),
            fetched(self.memory.tier_one(session.operator_id)),
        )
        if snapshot is None:
            return None, list(facts or [])
        return (
            {
                "window_days": snapshot.window_days,
                "origins": snapshot.origins,
                "headline": {
                    claim.key: claim.label or claim.value for claim in snapshot.claims
                },
                "limitations": snapshot.limitations,
                "note": (
                    "Orientation only. Quote a figure from a read, not from here: these "
                    "carry no formula, no inputs, and no window beyond the one named."
                ),
            },
            list(facts or []),
        )
