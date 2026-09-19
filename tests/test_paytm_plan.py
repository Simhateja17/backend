"""A payments-only Paytm merchant (QR / Soundbox) has no catalogue or stock, so the
stock tools are held with a POS upsell; a POS merchant reaches them as before."""

from __future__ import annotations

import pytest

from cartisan_agent import MerchantToolExecutor, Outcome
from cartisan_agent.merchant_executor import POS_GATE, POS_ONLY_TOOLS
from cartisan_agent.merchant_prompts import build_merchant_context
from cartisan_agent.merchant_types import MerchantSessionContext
from commerce_common.skills import SkillRegistry
from tests.conftest_merchant import OPERATOR, build_merchant, build_merchant_store, merchant_state
from cartisan_agent.outcomes import classify


def _executor(tmp_path, plan: str) -> MerchantToolExecutor:
    world = build_merchant(build_merchant_store(tmp_path))
    return MerchantToolExecutor(
        backend=world.services, config=world.config, skills=SkillRegistry([]),
        session=MerchantSessionContext(conversation_id="p", customer_id=OPERATOR, paytm_plan=plan),
        state=merchant_state(),
    )


@pytest.mark.parametrize("name", sorted(POS_ONLY_TOOLS))
async def test_payments_only_merchant_is_held_at_stock_tools(tmp_path, name):
    outcome = await _executor(tmp_path, "payments").execute(name, {})
    assert outcome.blocked == POS_GATE
    assert classify(outcome) is Outcome.BLOCKED
    assert "Paytm POS" in outcome.result_text


async def test_payments_only_merchant_still_reads_payments(tmp_path):
    outcome = await _executor(tmp_path, "payments").execute("get_business_snapshot", {})
    assert outcome.blocked is None and not outcome.is_error


async def test_pos_merchant_reads_stock(tmp_path):
    outcome = await _executor(tmp_path, "pos").execute("get_inventory_alerts", {})
    assert outcome.blocked is None


def test_context_names_the_plan():
    assert '"plan": "payments"' in build_merchant_context(
        operator_name=None, paytm_plan="payments", store_context=None, memory_facts=[])
