"""The shopping agent's turn loop (ADR 0014).

The loop itself is `runtime.AgentRuntime`, shared with the merchant surface. What
is here is what makes a turn a *shopping* turn: the tool array and prompt halves
it is built from, the cart and preferences prefetched into its context block, and
deterministic checkout precedence deciding the first round before the model is
asked anything (ADR 0021).

    runtime = CartisanShoppingRuntime(services=..., store=..., skills_dir=...)
    async for event in runtime.stream_turn(messages, session, state):
        ...

`messages` ends with the user's message and is extended in place with the turn's
assistant messages and tool results, so the host stores it as it stands. `state` is
the conversation's provenance and comes back on every turn.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic

from commerce_common.grounding import first_forced_tool
from commerce_common.memory import MemoryRuntime
from commerce_common.skills import SkillRegistry
from marketplace_backend.store import Store

from .config import CartisanAgentConfig
from .contracts import build_shopping_tools
from .executor import CartisanToolExecutor, CommerceServices, build_memory
from .gates import GROUNDING_RULES, is_checkout_turn
from .presentations import PRESENTATION_COMPONENTS
from .prompts import build_dynamic_context, build_static_system
from .runtime import AgentRuntime, fetched
from .turns import TurnStore
from .types import SessionContext, SessionState
from .versions import PROMPT_VERSION

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(__file__).resolve().parent / "skills"


class CartisanShoppingRuntime(AgentRuntime):
    prompt_version = PROMPT_VERSION

    def __init__(
        self,
        *,
        services: CommerceServices,
        store: Store,
        config: CartisanAgentConfig | None = None,
        skills: SkillRegistry | None = None,
        skills_dir: Path | None = SKILLS_DIR,
        client: AsyncAnthropic | None = None,
        memory: MemoryRuntime | None = None,
        turns: TurnStore | None = None,
        brief_reader: Callable[[str], dict[str, Any] | None] | None = None,
    ) -> None:
        config = config or CartisanAgentConfig()
        self.store = store
        # The customer's precomputed memory brief (host-owned, read from Postgres):
        # a read that fails leaves the turn without it, never without an answer.
        self.brief_reader = brief_reader
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
            components=PRESENTATION_COMPONENTS,
            tools=build_shopping_tools(config, skills.names),
            static_system=build_static_system(config, skills),
            memory=memory or build_memory(config),
            client=client,
            turns=turns,
        )

    # -- what a shopping turn supplies -------------------------------------------

    def new_state(self) -> SessionState:
        return SessionState()

    def before_turn(self, state: SessionState, user_text: str) -> None:
        """Checkout precedence is decided here, in code, before the model is asked
        anything at all (ADR 0021). It pins the first round and, through the gate in
        the executor, closes search and cart mutation for the whole turn."""
        state.checkout_turn = is_checkout_turn(self.config, user_text)
        lowered = user_text.lower()
        asks_for_cheaper = any(
            cue in lowered
            for cue in ("cheaper", "lower price", "less expensive", "more affordable")
        )
        state.cheaper_anchor_variant_id = (
            state.last_presented_variant_ids[0]
            if asks_for_cheaper and len(state.last_presented_variant_ids) == 1
            else None
        )

    def build_executor(
        self, session: SessionContext, state: SessionState
    ) -> CartisanToolExecutor:
        return CartisanToolExecutor(
            backend=self.services,
            config=self.config,
            skills=self.skills,
            session=session,
            state=state,
            memory=self.memory,
        )

    async def dynamic_context(self, session: SessionContext) -> str:
        preferences, cart, facts = await self._prefetch(session)
        return build_dynamic_context(
            preferences=preferences,
            memory_facts=facts,
            memory_brief=self._brief(session),
            catalogue_index=self._catalogue_index(),
            cart=cart,
            page=session.page,
            now=session.local_now(),
            channel=session.channel,
            max_chars=self.config.max_context_chars * 3,
        )

    def forced_tool(self, state: SessionState, user_text: str) -> str | None:
        """What the turn's first round is pinned to. Checkout outranks every grounding
        rule: a customer saying "complete my purchase" is not asking a policy question,
        whatever words they used to say it."""
        if state.checkout_turn:
            return "stage_checkout"
        return first_forced_tool(GROUNDING_RULES, self.config, user_text, state)

    def target_type(self, name: str) -> str | None:
        if name in {"add_to_cart", "update_cart_item", "remove_from_cart", "get_cart", "present_cart"}:
            return "cart"
        if name in {"stage_checkout", "present_checkout"}:
            return "checkout_stage"
        if name in {"get_order_status", "present_order_status"}:
            return "order"
        if name in {"search_products", "get_product_details", "check_compatibility"}:
            return "catalog_variant"
        return None

    def target_id(self, arguments: dict[str, Any]) -> str | None:
        for key in ("variant_id", "base_variant_id", "order_id", "stage_id", "item_ref"):
            if arguments.get(key):
                return str(arguments[key])
        return None

    # -- internals ----------------------------------------------------------------

    def _brief(self, session: SessionContext) -> dict[str, Any] | None:
        if self.brief_reader is None or not self.memory.enabled:
            return None
        try:
            return self.brief_reader(session.customer_id)
        except Exception:
            logger.warning("memory brief read failed; the turn runs without it", exc_info=True)
            return None

    def _catalogue_index(self) -> dict[str, list[str]]:
        """Return the active DB catalogue as a compact product-level map.

        It guides planning only. Variants, current prices, and stock still come from
        the normal catalogue tools before the agent can present or sell anything.
        """
        rows = self.store.rows(
            "SELECT COALESCE(c.name, 'Other') AS category, p.title "
            "FROM catalog_products p "
            "LEFT JOIN catalog_categories c ON c.id = p.category_id "
            "WHERE p.status = 'active' AND EXISTS ("
            "SELECT 1 FROM catalog_variants v "
            "WHERE v.product_id = p.id AND v.status = 'active') "
            "ORDER BY category, p.title"
        )
        catalogue: dict[str, list[str]] = {}
        for row in rows:
            titles = catalogue.setdefault(str(row["category"]), [])
            title = str(row["title"])
            if title not in titles:
                titles.append(title)
        return catalogue

    async def _prefetch(self, session: SessionContext) -> tuple[Any, Any, list[Any]]:
        preferences, cart, facts = await asyncio.gather(
            fetched(self.services.port.get_preferences(session)),
            fetched(self.services.port.get_cart(session) if self.config.enable_cart else None),
            fetched(self.memory.tier_one(session.customer_id)),
        )
        return preferences, cart, list(facts or [])
