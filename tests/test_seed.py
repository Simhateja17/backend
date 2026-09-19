"""Phase 3 acceptance: the generated merchant.

The acceptance criterion is that an isolated reset reproduces expected record
counts, scenarios, metrics, inventory reconciliation, and attribution lineage.
"""

import json

import pytest

from marketplace_backend.metrics import MetricsRepository
from marketplace_backend.seed import (
    GENERATOR_VERSION,
    SCENARIOS,
    CommerceGenerator,
    install_scenarios,
    validate_all,
)
from marketplace_backend.seed.generator import SEED_PREFIX
from marketplace_backend.seed.validators import (
    validate_attributed_revenue,
    validate_attribution,
    validate_inventory,
    validate_lineage,
    validate_origin_labelling,
    validate_payments,
    validate_totals,
)
from marketplace_backend.store import Store


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """One generated world, shared by the read-only checks below."""
    store = Store(tmp_path_factory.mktemp("seed") / "world.db")
    generator = CommerceGenerator(store)
    generated = generator.generate()
    before = generator.snapshot_ids()
    results = install_scenarios(store, generated)
    generator.capture_scenario_rows(before, generated.seed_run_id)
    return {"store": store, "generator": generator, "world": generated, "scenarios": results}


@pytest.fixture
def fresh(tmp_path):
    store = Store(tmp_path / "fresh.db")
    return store, CommerceGenerator(store)


# ============================================================ determinism


def test_the_same_seed_reproduces_the_same_rows(tmp_path):
    """Byte-for-byte reproducibility is the whole point of a versioned generator."""
    def fingerprint(path):
        store = Store(path)
        CommerceGenerator(store, seed=4242).generate()
        digest = {}
        for table in ("catalog_products", "catalog_variants", "variant_prices", "variant_specs",
                      "commerce_orders", "commerce_order_lines", "payment_attempts",
                      "inventory_levels", "commerce_events", "recommendations"):
            rows = store.rows(f"SELECT * FROM {table}")
            digest[table] = json.dumps(rows, sort_keys=True, default=str)
        return digest

    first = fingerprint(tmp_path / "a.db")
    second = fingerprint(tmp_path / "b.db")

    assert first.keys() == second.keys()
    for table in first:
        assert first[table] == second[table], f"{table} differs between identical runs"


def test_a_different_seed_produces_a_different_business(tmp_path):
    def orders(path, seed):
        store = Store(path)
        CommerceGenerator(store, seed=seed).generate()
        return store.rows("SELECT COALESCE(SUM(total_minor),0) AS total FROM commerce_orders")[0]["total"]

    assert orders(tmp_path / "a.db", 1) != orders(tmp_path / "b.db", 2)


def test_the_catalogue_is_identical_across_seeds(tmp_path):
    """The catalogue is a fixed domain; only the history is stochastic."""
    def skus(path, seed):
        store = Store(path)
        CommerceGenerator(store, seed=seed).generate()
        return [row["sku"] for row in store.rows("SELECT sku FROM catalog_variants ORDER BY sku")]

    assert skus(tmp_path / "a.db", 1) == skus(tmp_path / "b.db", 2)


def test_history_is_anchored_to_as_of_not_to_the_clock(tmp_path):
    from datetime import UTC, datetime

    store = Store(tmp_path / "anchored.db")
    CommerceGenerator(store, as_of=datetime(2026, 3, 15, tzinfo=UTC)).generate()

    latest = store.rows("SELECT MAX(created_at) AS newest FROM commerce_orders")[0]["newest"]
    assert latest.startswith("2026-03-1")


# ============================================================ catalogue


def test_the_catalogue_is_in_the_expected_size_band(world):
    """300-500 SKUs of genuine apparel and fashion depth."""
    counts = world["world"].counts

    assert 300 <= counts["catalog_variants"] <= 500
    assert counts["catalog_products"] >= 100
    assert counts["catalog_categories"] == 9


def test_every_variant_is_priced_and_stocked(world):
    store = world["store"]

    unpriced = store.rows(
        "SELECT v.id FROM catalog_variants v LEFT JOIN variant_prices p ON p.variant_id=v.id "
        "WHERE p.id IS NULL")
    unstocked = store.rows(
        "SELECT v.id FROM catalog_variants v LEFT JOIN inventory_levels l ON l.variant_id=v.id "
        "WHERE l.variant_id IS NULL")

    assert unpriced == []
    assert unstocked == []


def test_specifications_are_typed_not_free_text(world):
    """A numeric spec is queryable as a number, which is what makes filtering real."""
    store = world["store"]

    heavyweight = store.rows(
        "SELECT variant_id FROM variant_specs WHERE spec_key='gsm' AND value_numeric >= 200")
    mistyped = store.rows(
        "SELECT variant_id FROM variant_specs WHERE spec_key='gsm' AND value_numeric IS NULL")

    assert heavyweight, "no numeric spec rows to filter on"
    assert mistyped == [], "gsm must always be numeric"


def test_compatibility_rules_are_structured_and_explained(world):
    store = world["store"]

    requirements = store.rows(
        "SELECT r.operator, r.explanation, c.value_kind FROM variant_requirements r "
        "JOIN capabilities c ON c.id = r.capability_id")

    assert requirements
    for requirement in requirements:
        assert requirement["operator"] in {"eq", "neq", "gte", "lte", "in", "is_true"}
        assert requirement["explanation"].strip(), "every rule must be able to explain itself"


def test_a_requirement_can_be_satisfied_by_a_real_variant(world):
    """The rules describe this catalogue, not a hypothetical one."""
    store = world["store"]

    satisfiable = store.rows(
        "SELECT r.id FROM variant_requirements r "
        "JOIN variant_capabilities vc ON vc.capability_id = r.capability_id "
        "WHERE (r.operator='gte' AND vc.value_numeric >= r.value_numeric) "
        "   OR (r.operator='eq' AND vc.value_text = r.value_text) LIMIT 5")

    assert satisfiable, "no requirement in the catalogue can be met by anything in it"


# ============================================================ reset


def test_reset_removes_every_seeded_row(fresh):
    store, generator = fresh
    generated = generator.generate()
    before = generator.snapshot_ids()
    install_scenarios(store, generated)
    generator.capture_scenario_rows(before, generated.seed_run_id)
    assert generator.seeded_row_count() > 0

    generator.reset()

    assert generator.seeded_row_count() == 0
    for table in ("catalog_products", "catalog_variants", "commerce_orders", "commerce_events",
                  "inventory_levels", "payment_attempts", "checkout_stages", "recommendations"):
        assert store.rows(f"SELECT id FROM {table}" if table != "inventory_levels"
                          else "SELECT variant_id AS id FROM inventory_levels") == [], table


def test_reset_reproduces_the_same_counts(fresh):
    """The acceptance criterion: an isolated reset returns the same business."""
    store, generator = fresh
    first = generator.generate().counts

    generator.reset()
    second = CommerceGenerator(store).generate().counts

    assert first == second
    assert first["seed_customers"] == 60


def test_reset_protects_live_records(fresh):
    """A reseed must never delete a real customer's order (ADR 0008)."""
    store, generator = fresh
    generator.generate()
    store.execute(
        "INSERT INTO catalog_categories (id,name) VALUES ('live_cat','Live')")
    store.execute(
        "INSERT INTO catalog_products (id,sku_root,title,brand,category_id,description,status,origin,"
        "created_at) VALUES ('live_prd','LIVE-1','Live Product','Aster','live_cat','d','active',"
        "'live_app','2026-09-04T00:00:00+00:00')")
    store.execute(
        "INSERT INTO commerce_events (id,occurred_at,event_type,subject_type,subject_id,"
        "amount_minor,origin) VALUES ('live_evt','2026-09-04T00:00:00+00:00','order_paid','order',"
        "'live_ord',500000,'live_app')")

    generator.reset()

    assert store.rows("SELECT id FROM catalog_products") == [{"id": "live_prd"}]
    assert store.rows("SELECT id FROM commerce_events") == [{"id": "live_evt"}]


def test_scenario_rows_are_removed_by_reset(fresh):
    """Scenarios run through the production repositories, so they carry no seed
    prefix — the capture is what makes them removable."""
    store, generator = fresh
    generated = generator.generate()
    before = generator.snapshot_ids()
    install_scenarios(store, generated)
    captured = generator.capture_scenario_rows(before, generated.seed_run_id)
    assert captured > 0
    assert store.rows("SELECT id FROM commerce_orders WHERE id NOT LIKE 'sd_%'")

    generator.reset()

    assert store.rows("SELECT id FROM commerce_orders") == []
    assert store.rows("SELECT id FROM seed_scenario_rows") == []


def test_the_seed_run_records_its_version_and_counts(fresh):
    store, generator = fresh
    world = generator.generate()

    row = store.rows("SELECT * FROM seed_runs")[0]

    assert row["generator_version"] == GENERATOR_VERSION
    assert row["seed"] == generator.seed
    assert json.loads(row["record_counts"]) == world.counts


# ============================================================ invariants


@pytest.mark.parametrize("validator", [
    validate_totals, validate_inventory, validate_payments, validate_attribution,
    validate_attributed_revenue, validate_lineage, validate_origin_labelling,
], ids=lambda v: v.__name__)
def test_generated_data_satisfies_every_invariant(world, validator):
    report = validator(world["store"])
    assert report.ok, f"{report}: {report.problems[:5]}"


def test_validators_actually_catch_a_broken_business(fresh):
    """A validator that cannot fail proves nothing."""
    store, generator = fresh
    generator.generate()
    assert validate_totals(store).ok

    store.execute("UPDATE commerce_orders SET total_minor = total_minor + 1")

    report = validate_totals(store)
    assert not report.ok
    assert "!= computed" in report.problems[0]


def test_inventory_validator_catches_a_drifted_level(fresh):
    store, generator = fresh
    generator.generate()
    assert validate_inventory(store).ok

    store.execute("UPDATE inventory_levels SET on_hand = on_hand + 5")

    assert not validate_inventory(store).ok


def test_attribution_validator_catches_an_unaccepted_recommendation(fresh):
    store, generator = fresh
    generator.generate()
    assert validate_attribution(store).ok

    attributed = store.rows(
        "SELECT recommendation_id FROM commerce_order_lines WHERE recommendation_id IS NOT NULL "
        "LIMIT 1")[0]["recommendation_id"]
    store.execute("UPDATE recommendations SET accepted_at=NULL WHERE id=?", (attributed,))

    report = validate_attribution(store)
    assert not report.ok
    assert "nobody accepted" in report.problems[0]


# ============================================================ scenarios


def test_every_named_scenario_installs(world):
    assert set(world["scenarios"]) == {pack.key for pack in SCENARIOS}


def test_golden_purchase_reaches_paid(world):
    assert world["scenarios"]["golden_purchase"]["status"] == "paid"


def test_a_declined_card_does_not_lose_the_order(world):
    result = world["scenarios"]["declined_then_retry"]

    assert result["status"] == "paid"
    assert result["attempts"] == 2


def test_an_expired_preview_is_refused(world):
    result = world["scenarios"]["expired_stage"]

    assert result["refused"] is True
    assert result["state"] == "expired"


def test_abandoning_a_checkout_returns_the_stock(world):
    result = world["scenarios"]["abandoned_then_released"]

    assert result["sellable_while_held"] == result["sellable_before"] - 1
    assert result["sellable_after"] == result["sellable_before"]


def test_the_last_unit_goes_to_exactly_one_shopper(world):
    result = world["scenarios"]["last_unit_contention"]

    assert result["second_refused"] is True
    assert result["sellable"] == 0


def test_a_mismatched_provider_payload_leaves_the_order_unpaid(world):
    result = world["scenarios"]["provider_mismatch_quarantined"]

    assert result["quarantined"] is True
    assert result["order_status"] != "paid"


def test_a_replayed_webhook_is_stored_once(world):
    result = world["scenarios"]["webhook_replay_deduplicated"]

    assert result["deliveries"] == 3
    assert result["newly_stored"] == 1
    assert result["rows"] == 1


def test_a_declined_recommendation_is_not_revenue(world):
    result = world["scenarios"]["cross_sell_presented_not_taken"]

    assert result["presented_not_accepted"] > 0
    assert result["attributed_order_lines"] <= result["accepted"]


def test_scenarios_leave_evidence_a_judge_can_follow(world):
    """Each scenario's correlation id walks its whole journey."""
    store = world["store"]
    correlation = world["scenarios"]["golden_purchase"]["correlation_id"]

    lineage = store.rows(
        "SELECT action, outcome FROM evidence_records WHERE correlation_id=? ORDER BY recorded_at",
        (correlation,))

    assert [row["action"] for row in lineage] == [
        "stage_checkout", "confirm_checkout", "open_payment_attempt", "settle_payment_attempt"]
    assert {row["outcome"] for row in lineage} == {"applied"}


# ============================================================== metrics


def test_metrics_are_derived_from_the_event_log(world):
    metrics = MetricsRepository(world["store"])
    store = world["store"]

    revenue = metrics.revenue(origin="seeded")
    direct = store.rows(
        "SELECT COALESCE(SUM(amount_minor),0) AS total FROM commerce_events "
        "WHERE event_type='order_paid' AND origin='seeded'")[0]["total"]

    assert revenue.inputs["gross_minor"] == direct
    assert revenue.basis, "a metric must say what it was computed from"


def test_metrics_separate_seeded_from_live(fresh):
    """A live demo purchase must never be added into the seeded history (ADR 0032)."""
    store, generator = fresh
    generator.generate()
    metrics = MetricsRepository(store)
    seeded_before = metrics.revenue(origin="seeded").value

    store.execute(
        "INSERT INTO commerce_events (id,occurred_at,event_type,subject_type,subject_id,"
        "amount_minor,origin) VALUES ('live_evt','2026-09-04T00:00:00+00:00','order_paid','order',"
        "'live_ord',777700,'live_app')")

    assert metrics.revenue(origin="seeded").value == seeded_before
    assert metrics.revenue(origin="live_app").value == 777700
    assert metrics.revenue(origin=None).value == seeded_before + 777700


def test_an_unknown_origin_is_refused(world):
    with pytest.raises(ValueError, match="unknown origin"):
        MetricsRepository(world["store"]).revenue(origin="production")


def test_every_metric_carries_its_basis_and_limits(world):
    snapshot = MetricsRepository(world["store"]).snapshot(origin="seeded")

    assert snapshot["metrics"]
    for metric in snapshot["metrics"]:
        assert metric["basis"], f"{metric['key']} does not say how it was computed"
        assert isinstance(metric["inputs"], dict) and metric["inputs"]


def test_conversion_states_what_it_is_not(world):
    """A rate that could be misread must say so itself."""
    conversion = MetricsRepository(world["store"]).conversion(origin="seeded")

    assert 0 < conversion.value < 1
    assert any("not visit-to-purchase" in note for note in conversion.limitations)


def test_agent_assisted_revenue_is_descriptive_not_causal(world):
    metric = MetricsRepository(world["store"]).agent_assisted_revenue(origin="seeded")

    assert metric.value > 0
    assert metric.inputs["attributed_lines"] > 0
    assert metric.inputs["recommendations_presented"] > metric.inputs["recommendations_accepted"]
    assert any("not causal" in note.lower() or "descriptive" in note.lower()
               for note in metric.limitations)


def test_agent_assisted_revenue_excludes_unpaid_orders(fresh):
    store, generator = fresh
    generator.generate()
    metrics = MetricsRepository(store)
    before = metrics.agent_assisted_revenue(origin="seeded").value

    # Un-pay every attributed order: the figure must fall to zero, not stay put.
    store.execute(
        "UPDATE commerce_orders SET status='cancelled', amount_paid_minor=0 WHERE id IN "
        "(SELECT order_id FROM commerce_order_lines WHERE recommendation_id IS NOT NULL)")

    assert before > 0
    assert metrics.agent_assisted_revenue(origin="seeded").value == 0


def test_inventory_alerts_surface_the_thin_stock(world):
    alerts = MetricsRepository(world["store"]).inventory_alerts(threshold=3)

    assert alerts
    assert all(row["sellable"] <= 3 for row in alerts)
    assert all(row["title"] for row in alerts)


def test_daily_revenue_returns_a_series(world):
    series = MetricsRepository(world["store"]).daily_revenue(origin="seeded", limit=10)

    assert 0 < len(series) <= 10
    assert all(row["orders"] > 0 for row in series)


def test_checkout_health_shows_failures_not_just_successes(world):
    health = MetricsRepository(world["store"]).checkout_health(origin="seeded")

    assert health["orders_by_status"]["paid"] > 0
    assert set(health["orders_by_status"]) - {"paid"}, "a demo with no failures is not evidence"
    assert "failed" in health["attempts_by_status"]


# ============================================================== labelling


def test_seeded_rows_are_labelled_seeded(world):
    store = world["store"]

    for table in ("catalog_products", "commerce_orders"):
        mislabelled = store.rows(
            f"SELECT id FROM {table} WHERE id LIKE '{SEED_PREFIX}%' AND origin <> 'seeded'")
        assert mislabelled == [], table


def test_scenario_orders_are_labelled_as_test_mode(world):
    """Scenario orders go through the real checkout, so they carry its origin."""
    store = world["store"]

    origins = {row["origin"] for row in store.rows(
        "SELECT DISTINCT origin FROM commerce_orders WHERE id NOT LIKE 'sd_%'")}

    assert origins == {"razorpay_test"}


def test_demo_stories_have_the_stock_and_sales_they_promise(world):
    """Every merchant edge case the demo leans on is built, not left to chance."""
    from marketplace_backend.seed import domain
    from marketplace_backend.seed.generator import story_variant

    store = world["store"]

    def on_hand(variant_id):
        return [row["on_hand"] for row in store.rows(
            "SELECT on_hand FROM inventory_levels WHERE variant_id=? ORDER BY location_id",
            (variant_id,))]

    def sold(variant_id):
        rows = store.rows(
            "SELECT COALESCE(SUM(l.quantity),0) AS n FROM commerce_order_lines l "
            "JOIN commerce_orders o ON o.id=l.order_id "
            "WHERE l.variant_id=? AND o.status='paid' AND o.origin='seeded'", (variant_id,))
        return int(rows[0]["n"])

    for key, index, units, stock, _why in domain.DEMAND_STORIES:
        variant_id = story_variant(key, index)
        assert sorted(on_hand(variant_id)) == sorted(stock), key
        assert sold(variant_id) == units, key
    for key, index, stock in domain.DEAD_STOCK:
        variant_id = story_variant(key, index)
        assert sum(on_hand(variant_id)) == sum(stock)
        assert sold(variant_id) == 0
