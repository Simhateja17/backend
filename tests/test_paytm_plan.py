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


# -- payment health and restock financing ---------------------------------------------

from marketplace_backend import merchant_finance  # noqa: E402
from marketplace_backend.merchant_changes import MerchantChangeRepository, PolicyViolation  # noqa: E402


async def test_payment_health_counts_only_verified_money(tmp_path):
    outcome = await _executor(tmp_path, "payments").execute("get_payment_health", {"window_days": 30})
    assert outcome.blocked is None and not outcome.is_error
    assert "verified_collections" in outcome.result_text


async def test_loan_cannot_be_staged_before_it_is_sized(tmp_path):
    outcome = await _executor(tmp_path, "pos").execute(
        "stage_loan_request", {"amount_minor": 100, "tenure_months": 6, "purpose": "x",
                               "rationale": "x"})
    assert outcome.blocked == "loan_provenance"


async def test_payments_only_merchant_cannot_size_a_restock_loan(tmp_path):
    outcome = await _executor(tmp_path, "payments").execute("check_restock_financing", {})
    assert outcome.blocked == POS_GATE


async def test_loan_is_staged_against_the_computed_limit(tmp_path):
    executor = _executor(tmp_path, "pos")
    executor._state.sized_loan = {
        "restock_cost_minor": 500_000_00, "cash_last_7_days_minor": 100_000_00,
        "loan": {"suggested_amount_minor": 400_000_00, "eligible_limit_minor": 600_000_00}}
    outcome = await executor.execute(
        "stage_loan_request", {"amount_minor": 400_000_00, "tenure_months": 6,
                               "purpose": "Diwali restock", "rationale": "Shortfall of ₹4,00,000"})
    assert not outcome.refused, outcome.result_text
    assert "Queued for approval" in outcome.result_text


def test_loan_above_limit_or_bad_tenure_is_refused():
    before = {"eligible_limit_minor": 1_000}
    with pytest.raises(PolicyViolation):
        MerchantChangeRepository.check_policy("loan_request", before,
                                              {"amount_minor": 2_000, "tenure_months": 6})
    with pytest.raises(PolicyViolation):
        MerchantChangeRepository.check_policy("loan_request", before,
                                              {"amount_minor": 500, "tenure_months": 5})
    MerchantChangeRepository.check_policy("loan_request", before,
                                          {"amount_minor": 1_000, "tenure_months": 12})


def test_repayment_estimate_is_positive_and_rounds_up():
    assert merchant_finance.monthly_repayment(600_00, 6) == 109_00
    assert merchant_finance._inr(1_23_45_600) == "₹1,23,456"


async def test_recovery_policy_can_be_staged_by_the_agent(tmp_path):
    outcome = await _executor(tmp_path, "pos").execute("stage_recovery_policy", {
        "min_cart_minor": 150_000, "discount_percentage": 10, "max_discount_minor": 30_000,
        "monthly_budget_minor": 1_000_000, "rationale": "Abandoned carts worth ₹48,000 last week"})
    assert not outcome.refused, outcome.result_text


async def test_a_later_check_without_shortfall_keeps_the_sized_loan(tmp_path):
    """The failure seen in the portal: 2x demand sized a loan, a 1x re-check found no
    shortfall, and staging was refused because the sized loan had been overwritten."""
    executor = _executor(tmp_path, "pos")
    executor._state.sized_loan = {
        "restock_cost_minor": 630_000_00, "cash_last_7_days_minor": 494_000_00,
        "loan": {"suggested_amount_minor": 140_000_00, "eligible_limit_minor": 2_400_000_00}}
    await executor.execute("check_restock_financing", {"horizon_days": 21})  # no loan at 1x
    outcome = await executor.execute(
        "stage_loan_request", {"amount_minor": 140_000_00, "tenure_months": 3,
                               "purpose": "Diwali restock", "rationale": "2x demand shortfall"})
    assert not outcome.refused, outcome.result_text
