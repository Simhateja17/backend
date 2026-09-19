"""The versioned, deterministic commerce generator.

Two properties matter more than anything else here:

  Determinism.  The same `(GENERATOR_VERSION, seed, as_of)` triple produces the
  same rows, every time, on either backend. Nothing calls `datetime.now()` or an
  unseeded RNG. `as_of` is an explicit parameter precisely so that "ninety days of
  history" does not shift under the tests.

  Separation.  Every generated row carries a `sd_` id prefix and, where the column
  exists, `origin='seeded'`. A reset deletes exactly those rows and nothing else,
  so live demo records survive a reseed (ADR 0008).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from ..store import Store
from . import domain
from .domain import Counts

GENERATOR_VERSION = "1.0.0"
DEFAULT_SEED = 20260904
HISTORY_DAYS = 90
SEED_PREFIX = "sd_"

# Numeric specs where a smaller number is the better product (quieter, lighter).
# Every other numeric spec in domain.LINES climbs with price.
_LOWER_IS_BETTER: frozenset[str] = frozenset({"noise_db", "weight_g"})


def _tier_scaled(line_key: str, spec_key: str, value: object, tier_fraction: float) -> object:
    """A line declares one representative spec value; the edition's price tier
    (0.0 cheapest, 1.0 dearest) spreads it into a real range, so two editions of the
    same line actually differ on more than price. Text specs (a hardness grade, a
    switch feel) are not on a continuum and are left as declared."""
    if isinstance(value, bool):
        # A different, deterministic threshold per (line, spec) so features unlock
        # at different price points instead of every boolean flipping together.
        threshold = (sum(ord(c) for c in f"{line_key}:{spec_key}") % 60) / 100
        return tier_fraction >= threshold
    if isinstance(value, (int, float)):
        factor = (1.2 - 0.4 * tier_fraction) if spec_key in _LOWER_IS_BETTER else (0.8 + 0.4 * tier_fraction)
        scaled = value * factor
        return round(scaled) if isinstance(value, int) else round(scaled, 1)
    return value

# Deletion runs children-before-parents so foreign keys hold at every step.
_RESET_ORDER: tuple[tuple[str, str], ...] = (
    ("evidence_records", "id"),
    ("commerce_events", "id"),
    ("tool_executions", "id"),
    ("turns", "id"),
    ("conversations", "id"),
    ("outbox_messages", "id"),
    ("inbox_events", "id"),
    ("refunds", "id"),
    ("fulfillment_lines", "fulfillment_id"),
    ("fulfillments", "id"),
    ("payment_attempts", "id"),
    ("recommendations", "id"),
    ("presentation_items", "id"),
    ("presentations", "id"),
    ("commerce_order_lines", "id"),
    ("commerce_orders", "id"),
    ("checkout_stage_lines", "stage_id"),
    ("checkout_stages", "id"),
    ("inventory_reservations", "id"),
    ("inventory_movements", "id"),
    ("inventory_levels", "variant_id"),
    ("merchant_approvals", "id"),
    ("merchant_changes", "id"),
    ("campaigns", "id"),
    ("promotions", "id"),
    ("variant_requirements", "id"),
    ("variant_capabilities", "variant_id"),
    ("variant_specs", "variant_id"),
    ("variant_prices", "id"),
    ("catalog_variants", "id"),
    ("catalog_products", "id"),
    ("capabilities", "id"),
    ("inventory_locations", "id"),
    ("catalog_categories", "id"),
    ("seed_customers", "id"),
    ("seed_scenario_rows", "id"),
    ("seed_runs", "id"),
)

# The tables a scenario pack can write to. Scenarios deliberately run through the
# real repositories, so their rows carry production id prefixes rather than `sd_`;
# `capture_scenario_rows` records exactly what appeared so a reset can remove it.
_SCENARIO_TABLES: tuple[tuple[str, str], ...] = tuple(
    entry for entry in _RESET_ORDER
    if entry[0] not in {"seed_runs", "seed_scenario_rows", "seed_customers", "catalog_categories",
                        "capabilities", "inventory_locations"})


@dataclass
class GeneratedWorld:
    """Handles onto what was generated, for scenarios and validators."""

    seed_run_id: str
    counts: dict[str, int]
    variant_ids: list[str]
    customer_ids: list[str]
    location_ids: list[str]
    line_variants: dict[str, list[str]]


def _iso(moment: datetime) -> str:
    return moment.isoformat()


class CommerceGenerator:
    def __init__(self, store: Store, *, seed: int = DEFAULT_SEED,
                 as_of: datetime | None = None) -> None:
        self.store = store
        self.seed = seed
        # A fixed default so a plain `reset()` is reproducible without arguments.
        self.as_of = as_of or datetime(2026, 9, 4, tzinfo=UTC)
        self.rng = random.Random(seed)
        self.counts = Counts()

    # --------------------------------------------------------------- reset

    def reset(self) -> None:
        """Destructively remove every seeded row. Live records are untouched.

        This is deliberately explicit: nothing calls it implicitly, and it matches
        on the `sd_` prefix rather than truncating tables, so a `live_app` order
        sitting in the same table survives.
        """
        recorded: dict[str, list[str]] = {}
        for row in self.store.rows("SELECT table_name,row_id FROM seed_scenario_rows"):
            recorded.setdefault(row["table_name"], []).append(row["row_id"])
        for table, column in _RESET_ORDER:
            # Scenario rows first: they reference seeded catalogue rows, so they must
            # go before the rows they point at.
            ids = recorded.get(table)
            if ids:
                placeholders = ",".join("?" for _ in ids)
                self.store.execute(
                    f"DELETE FROM {table} WHERE {column} IN ({placeholders})", tuple(ids))
            self.store.execute(f"DELETE FROM {table} WHERE {column} LIKE ?", (f"{SEED_PREFIX}%",))

    def seeded_row_count(self) -> int:
        total = 0
        for table, column in _RESET_ORDER:
            rows = self.store.rows(
                f"SELECT count(*) AS n FROM {table} WHERE {column} LIKE ?", (f"{SEED_PREFIX}%",))
            total += int(rows[0]["n"])
        scenario = self.store.rows("SELECT count(*) AS n FROM seed_scenario_rows")
        return total + int(scenario[0]["n"])

    def snapshot_ids(self) -> dict[str, set[str]]:
        """Every id currently in the scenario-reachable tables."""
        return {
            table: {str(row["row_id"]) for row in
                    self.store.rows(f"SELECT {column} AS row_id FROM {table}")}
            for table, column in _SCENARIO_TABLES}

    def capture_scenario_rows(self, before: dict[str, set[str]], seed_run_id: str) -> int:
        """Record rows the scenario packs created, so `reset` can remove them.

        Scenarios run through the production repositories on purpose — that is what
        makes them evidence of real behaviour — which means they do not carry the
        `sd_` prefix. Recording the difference is how they stay removable without
        teaching the production code about seeding.
        """
        after = self.snapshot_ids()
        rows = []
        for table, _ in _SCENARIO_TABLES:
            for row_id in sorted(after[table] - before.get(table, set())):
                rows.append((f"{SEED_PREFIX}scn_{len(rows):06d}", seed_run_id, table, row_id))
        if rows:
            self.store.executemany(
                "INSERT INTO seed_scenario_rows (id,seed_run_id,table_name,row_id) VALUES (?,?,?,?)",
                rows)
        return len(rows)

    # ------------------------------------------------------------ generate

    def generate(self) -> GeneratedWorld:
        """Build the whole business. Callers reset first if they want a clean slate."""
        self.rng = random.Random(self.seed)
        self.counts = Counts()

        self._categories()
        self._capabilities()
        self._locations()
        variant_ids, line_variants = self._catalog()
        self._inventory(variant_ids)
        customer_ids = self._customers()
        self._promotions_and_campaigns()
        self._history(variant_ids, customer_ids, line_variants)

        seed_run_id = f"{SEED_PREFIX}run_{self.seed}_{GENERATOR_VERSION.replace('.', '_')}"
        self.store.execute(
            "INSERT INTO seed_runs (id,generator_version,seed,as_of,record_counts,created_at) "
            "VALUES (?,?,?,?,?,?)",
            (seed_run_id, GENERATOR_VERSION, self.seed, _iso(self.as_of),
             json.dumps(self.counts.as_dict()), _iso(self.as_of)))

        return GeneratedWorld(
            seed_run_id=seed_run_id, counts=self.counts.as_dict(), variant_ids=variant_ids,
            customer_ids=customer_ids, location_ids=[loc[0] for loc in self._location_rows()],
            line_variants=line_variants)

    # ------------------------------------------------------------ catalog

    def _categories(self) -> None:
        rows = [(f"{SEED_PREFIX}{key}", name,
                 f"{SEED_PREFIX}{parent}" if parent else None, _iso(self.as_of))
                for key, name, parent in domain.CATEGORIES]
        # Parents must land before children, and CATEGORIES is already in that order.
        self.store.executemany(
            "INSERT INTO catalog_categories (id,name,parent_id,created_at) VALUES (?,?,?,?)", rows)
        self.counts.add("catalog_categories", len(rows))

    def _capabilities(self) -> None:
        rows = [(f"{SEED_PREFIX}{cap_id}", label, kind) for cap_id, label, kind in domain.CAPABILITIES]
        self.store.executemany(
            "INSERT INTO capabilities (id,label,value_kind) VALUES (?,?,?)", rows)
        self.counts.add("capabilities", len(rows))

    def _location_rows(self) -> list[tuple[str, str, str, str]]:
        return [(f"{SEED_PREFIX}{key}", code, name, region)
                for key, code, name, region in domain.LOCATIONS]

    def _locations(self) -> None:
        rows = self._location_rows()
        self.store.executemany(
            "INSERT INTO inventory_locations (id,code,name,region) VALUES (?,?,?,?)", rows)
        self.counts.add("inventory_locations", len(rows))

    def _catalog(self) -> tuple[list[str], dict[str, list[str]]]:
        """Walk the product lines into products, variants, specs, capabilities and prices."""
        products, variants, specs, caps, reqs, prices = [], [], [], [], [], []
        variant_ids: list[str] = []
        line_variants: dict[str, list[str]] = {}

        for line in domain.LINES:
            line_variants[line.key] = []
            # Each line yields several SKUs, one per edition, so the catalog has
            # genuine within-line choice rather than a single token product.
            editions = self._editions_for(line)
            for edition_index, edition in enumerate(editions):
                brand = domain.BRANDS[(hash(line.key) + edition_index) % len(domain.BRANDS)]
                product_id = f"{SEED_PREFIX}prd_{line.key}_{edition_index}"
                sku_root = f"{line.key.upper().replace('_', '-')}-{edition_index:02d}"
                title = f"{brand} {edition} {line.name}"
                products.append((
                    product_id, sku_root, title, brand, f"{SEED_PREFIX}{line.category}",
                    line.blurb, "active", "seeded", _iso(self.as_of)))
                self.counts.add("catalog_products")

                base_low, base_high = line.price_range
                # Editions climb in price across the line's declared range.
                span = max(1, len(editions) - 1)
                base_price = base_low + (base_high - base_low) * edition_index // span
                # Specs climb with price the same way, so a comparison has real feature
                # differences to point at instead of price being the only signal.
                tier_fraction = edition_index / span

                model = domain.OUTFIT_SETS[edition_index % len(domain.OUTFIT_SETS)]
                for value_index, value in enumerate(line.variant_values):
                    variant_id = f"{product_id}_v{value_index}"
                    variant_ids.append(variant_id)
                    line_variants[line.key].append(variant_id)
                    variants.append((
                        variant_id, product_id, f"{sku_root}-{value_index}",
                        f"{title} — {value}",
                        json.dumps({line.variant_axis: value}), "active", _iso(self.as_of)))
                    self.counts.add("catalog_variants")

                    # Variant-axis values shift price within the product.
                    amount = base_price + value_index * max(10000, base_price // 20)
                    prices.append((
                        f"{SEED_PREFIX}prc_{variant_id}", variant_id, domain.CURRENCY, amount,
                        None, "list", _iso(self.as_of - timedelta(days=HISTORY_DAYS)), None))
                    self.counts.add("variant_prices")

                    specs.extend(
                        self._spec_rows(variant_id, line, line.variant_axis, value, tier_fraction)
                    )
                    caps.extend(self._capability_rows(variant_id, line, model))
                    reqs.extend(self._requirement_rows(variant_id, line, model))

        self.store.executemany(
            "INSERT INTO catalog_products (id,sku_root,title,brand,category_id,description,status,"
            "origin,created_at) VALUES (?,?,?,?,?,?,?,?,?)", products)
        self.store.executemany(
            "INSERT INTO catalog_variants (id,product_id,sku,title,options,status,created_at) "
            "VALUES (?,?,?,?,?,?,?)", variants)
        self.store.executemany(
            "INSERT INTO variant_specs (variant_id,spec_key,value_text,value_numeric,value_unit,"
            "value_bool) VALUES (?,?,?,?,?,?)", specs)
        self.store.executemany(
            "INSERT INTO variant_capabilities (variant_id,capability_id,value_text,value_numeric,"
            "value_bool) VALUES (?,?,?,?,?)", caps)
        self.store.executemany(
            "INSERT INTO variant_requirements (id,variant_id,capability_id,operator,value_text,"
            "value_numeric,severity,explanation) VALUES (?,?,?,?,?,?,?,?)", reqs)
        self.store.executemany(
            "INSERT INTO variant_prices (id,variant_id,currency,amount_minor,compare_at_minor,"
            "price_kind,valid_from,valid_to) VALUES (?,?,?,?,?,?,?,?)", prices)
        self.counts.add("variant_specs", len(specs))
        self.counts.add("variant_capabilities", len(caps))
        self.counts.add("variant_requirements", len(reqs))
        return variant_ids, line_variants

    def _editions_for(self, line: domain.Line) -> list[str]:
        """How many SKUs this line carries. Deterministic in the line's own name.

        Five to eight editions across thirty-four lines, each with two or three
        variant values, puts the catalogue inside the 300-500 SKU band the
        architecture asks for — with the depth coming from real within-line choice
        rather than padding.
        """
        span = 5 + (sum(ord(c) for c in line.key) % 4)  # 5 to 8 editions
        start = sum(ord(c) for c in line.key) % len(domain.EDITIONS)
        return [domain.EDITIONS[(start + i) % len(domain.EDITIONS)] for i in range(span)]

    def _spec_rows(
        self, variant_id: str, line: domain.Line, axis: str, value: str, tier_fraction: float
    ) -> list[tuple]:
        rows = [self._typed_spec(variant_id, axis, value)]
        for key, spec_value, unit in line.specs:
            scaled = _tier_scaled(line.key, key, spec_value, tier_fraction)
            rows.append(self._typed_spec(variant_id, key, scaled, unit))
        return rows

    @staticmethod
    def _typed_spec(variant_id: str, key: str, value: object, unit: str | None = None) -> tuple:
        """Land the value in exactly one typed column — the schema check enforces it.

        Booleans are written as Python bools, not as 1/0: SQLite stores either,
        but Postgres has a real `boolean` column and rejects a smallint.
        """
        if isinstance(value, bool):
            return (variant_id, key, None, None, unit, value)
        if isinstance(value, (int, float)):
            return (variant_id, key, None, value, unit, None)
        return (variant_id, key, str(value), None, unit, None)

    def _capability_rows(self, variant_id: str, line: domain.Line, model: str) -> list[tuple]:
        rows = []
        for cap_id, value in line.provides:
            capability = f"{SEED_PREFIX}{cap_id}"
            if isinstance(value, bool):
                rows.append((variant_id, capability, None, None, value))
            elif isinstance(value, (int, float)):
                rows.append((variant_id, capability, None, value, None))
            else:
                rows.append((variant_id, capability, str(value), None, None))
        # Coordinates tailored to one outfit set advertise which set that is, so a
        # compatibility check has something concrete on both sides.
        if line.category == domain.MATCHED_CATEGORY:
            rows.append((variant_id, f"{SEED_PREFIX}cap_outfit_set", model, None, None))
        if line.key in domain.MOUNTS:
            rows.append((variant_id, f"{SEED_PREFIX}cap_mount", domain.MOUNTS[line.key], None, None))
        # De-duplicate: a line may both declare and be given the same capability.
        seen, unique = set(), []
        for row in rows:
            if row[1] in seen:
                continue
            seen.add(row[1])
            unique.append(row)
        return unique

    def _requirement_rows(self, variant_id: str, line: domain.Line, model: str) -> list[tuple]:
        rows = []
        for index, (cap_id, operator, value, explanation) in enumerate(line.requires):
            # A `None` value means "this model", filled in per variant.
            resolved = model if value is None else value
            numeric = resolved if isinstance(resolved, (int, float)) and not isinstance(resolved, bool) else None
            text = None if numeric is not None else str(resolved)
            rows.append((
                f"{SEED_PREFIX}req_{variant_id}_{index}", variant_id, f"{SEED_PREFIX}{cap_id}",
                operator, text, numeric, "blocking",
                explanation.replace("one outfit set", f"the {model} set") if value is None else explanation))
        return rows

    # ---------------------------------------------------------- inventory

    def _inventory(self, variant_ids: list[str]) -> None:
        locations = [row[0] for row in self._location_rows()]
        movements, levels = [], []
        received_at = _iso(self.as_of - timedelta(days=HISTORY_DAYS))
        for variant_id in variant_ids:
            # Roughly a fifth of the catalogue is thin or out of stock, so
            # availability questions and stock-out journeys have real subjects.
            profile = self.rng.random()
            for index, location_id in enumerate(locations):
                if profile < 0.06:
                    quantity = 0
                elif profile < 0.20:
                    quantity = self.rng.randint(0, 3)
                else:
                    quantity = self.rng.randint(4, 60)
                if quantity == 0:
                    levels.append((variant_id, location_id, 0, 0, received_at))
                    continue
                levels.append((variant_id, location_id, quantity, 0, received_at))
                movements.append((
                    f"{SEED_PREFIX}mv_{variant_id}_{index}", variant_id, location_id, quantity,
                    "receipt", "seed", None, received_at))
        self.store.executemany(
            "INSERT INTO inventory_levels (variant_id,location_id,on_hand,reserved,updated_at) "
            "VALUES (?,?,?,?,?)", levels)
        self.store.executemany(
            "INSERT INTO inventory_movements (id,variant_id,location_id,delta,reason,"
            "reference_type,reference_id,created_at) VALUES (?,?,?,?,?,?,?,?)", movements)
        self.counts.add("inventory_levels", len(levels))
        self.counts.add("inventory_movements", len(movements))

    # ---------------------------------------------------------- principals

    def _customers(self) -> list[str]:
        """Seeded shoppers.

        These have no Supabase account behind them, so they live in
        `seed_customers` rather than `customers` — the latter is keyed on
        `auth.users` and holds verified principals only. The commerce core keys
        ownership by text, so a seeded order and a live order sit side by side.
        """
        rows, ids = [], []
        for index in range(60):
            given = domain.GIVEN_NAMES[index % len(domain.GIVEN_NAMES)]
            surname = domain.SURNAMES[(index // len(domain.GIVEN_NAMES) + index) % len(domain.SURNAMES)]
            customer_id = f"{SEED_PREFIX}cust_{index:03d}"
            ids.append(customer_id)
            rows.append((customer_id, f"{given.lower()}.{surname.lower()}{index}@cartisan.seed",
                         f"{given} {surname}",
                         _iso(self.as_of - timedelta(days=HISTORY_DAYS + 30))))
        self.store.executemany(
            "INSERT INTO seed_customers (id,email,display_name,created_at) VALUES (?,?,?,?)", rows)
        self.counts.add("seed_customers", len(rows))
        return ids

    def _promotions_and_campaigns(self) -> None:
        starts = _iso(self.as_of - timedelta(days=HISTORY_DAYS))
        promotions = [
            (f"{SEED_PREFIX}promo_{code}", code, description, kind, value, minimum,
             f"{SEED_PREFIX}{category}" if category else None, "active", starts, None)
            for code, description, kind, value, minimum, category in domain.PROMOTIONS]
        self.store.executemany(
            "INSERT INTO promotions (id,code,description,discount_kind,discount_value,"
            "min_subtotal_minor,category_id,status,starts_at,ends_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)", promotions)
        campaigns = [
            (f"{SEED_PREFIX}camp_{index}", name, channel, f"{SEED_PREFIX}promo_{code}", "running",
             budget, int(budget * self.rng.uniform(0.35, 0.85)), starts, None)
            for index, (name, channel, code, budget) in enumerate(domain.CAMPAIGNS)]
        self.store.executemany(
            "INSERT INTO campaigns (id,name,channel,promotion_id,status,budget_minor,spend_minor,"
            "starts_at,ends_at) VALUES (?,?,?,?,?,?,?,?,?)", campaigns)
        self.counts.add("promotions", len(promotions))
        self.counts.add("campaigns", len(campaigns))

    # ------------------------------------------------------------ history

    def _history(self, variant_ids: list[str], customer_ids: list[str],
                 line_variants: dict[str, list[str]]) -> None:
        """Ninety days of linked activity, in one pass per simulated day.

        Every order is reachable from a conversation, through a presentation, to a
        payment attempt and a fulfilment — the lineage the audit surface walks.
        """
        prices = {row["variant_id"]: row["amount_minor"] for row in
                  self.store.rows("SELECT variant_id,amount_minor FROM variant_prices")}
        # Which aisle each variant sits in, so a category-scoped promotion can be
        # applied to the lines it actually covers.
        categories = {row["id"]: row["category_id"] for row in self.store.rows(
            "SELECT v.id, p.category_id FROM catalog_variants v "
            "JOIN catalog_products p ON p.id = v.product_id")}
        stock = {(row["variant_id"], row["location_id"]): row["on_hand"] for row in
                 self.store.rows("SELECT variant_id,location_id,on_hand FROM inventory_levels")}
        sellable = [v for v in variant_ids if any(
            stock.get((v, loc[0]), 0) > 0 for loc in self._location_rows())]
        pairings = self._pairings(line_variants)

        buckets: dict[str, list[tuple]] = {name: [] for name in (
            "conversations", "turns", "presentations", "presentation_items", "recommendations",
            "checkout_stages", "checkout_stage_lines", "commerce_orders", "commerce_order_lines",
            "payment_attempts", "fulfillments", "fulfillment_lines", "refunds",
            "commerce_events", "evidence_records", "inbox_events")}
        counter = {"n": 0}

        for day_offset in range(HISTORY_DAYS, 0, -1):
            day = self.as_of - timedelta(days=day_offset)
            # Weekends run hotter, which gives the merchant metrics a real shape.
            base = 9 if day.weekday() >= 5 else 6
            for _ in range(self.rng.randint(base - 2, base + 3)):
                counter["n"] += 1
                self._one_journey(day, counter["n"], sellable, customer_ids, prices,
                                  categories, pairings, buckets)

        inserts = (
            ("conversations", "INSERT INTO conversations (id,principal_id,surface,created_at) VALUES (?,?,?,?)"),
            ("turns", "INSERT INTO turns (id,conversation_id,sequence,state,user_message,agent_message,"
                      "prompt_version,tool_contract_version,skill_versions,started_at,completed_at) "
                      "VALUES (?,?,?,?,?,?,?,?,?,?,?)"),
            ("presentations", "INSERT INTO presentations (id,conversation_id,customer_id,kind,turn_id,"
                              "created_at,expires_at) VALUES (?,?,?,?,?,?,?)"),
            ("presentation_items", "INSERT INTO presentation_items (id,presentation_id,position,"
                                   "variant_id,unit_price_minor) VALUES (?,?,?,?,?)"),
            ("recommendations", "INSERT INTO recommendations (id,presentation_item_id,customer_id,"
                                "variant_id,rationale,source_variant_id,accepted_at,created_at) "
                                "VALUES (?,?,?,?,?,?,?,?)"),
            ("checkout_stages", "INSERT INTO checkout_stages (id,cart_id,customer_id,cart_state_version,"
                                "state,currency,subtotal_minor,shipping_minor,tax_minor,discount_minor,"
                                "total_minor,promotion_id,fulfillment_option,constraints_note,expires_at,"
                                "created_at,resolved_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"),
            ("checkout_stage_lines", "INSERT INTO checkout_stage_lines (stage_id,variant_id,quantity,"
                                     "unit_price_minor,amount_minor) VALUES (?,?,?,?,?)"),
            ("commerce_orders", "INSERT INTO commerce_orders (id,customer_id,stage_id,status,currency,"
                                "subtotal_minor,shipping_minor,tax_minor,discount_minor,total_minor,"
                                "amount_paid_minor,promotion_id,origin,state_version,created_at,paid_at,"
                                "cancelled_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"),
            ("commerce_order_lines", "INSERT INTO commerce_order_lines (id,order_id,variant_id,quantity,"
                                     "unit_price_minor,amount_minor,recommendation_id) VALUES (?,?,?,?,?,?,?)"),
            ("payment_attempts", "INSERT INTO payment_attempts (id,order_id,provider,provider_reference,"
                                 "provider_link_url,status,amount_minor,currency,provider_snapshot,"
                                 "failure_reason,created_at,resolved_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"),
            ("fulfillments", "INSERT INTO fulfillments (id,order_id,status,carrier,tracking_reference,"
                             "promised_at,shipped_at,delivered_at,created_at) VALUES (?,?,?,?,?,?,?,?,?)"),
            ("fulfillment_lines", "INSERT INTO fulfillment_lines (fulfillment_id,order_line_id,quantity) "
                                  "VALUES (?,?,?)"),
            ("refunds", "INSERT INTO refunds (id,order_id,payment_attempt_id,amount_minor,reason,status,"
                        "provider_reference,created_at,resolved_at) VALUES (?,?,?,?,?,?,?,?,?)"),
            ("inbox_events", "INSERT INTO inbox_events (id,provider,provider_event_id,event_type,payload,"
                             "status,quarantine_reason,received_at,processed_at) VALUES (?,?,?,?,?,?,?,?,?)"),
            ("commerce_events", "INSERT INTO commerce_events (id,occurred_at,event_type,subject_type,"
                                "subject_id,customer_id,amount_minor,quantity,origin,demo_run_id,"
                                "correlation_id,detail) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"),
            ("evidence_records", "INSERT INTO evidence_records (id,recorded_at,actor_type,actor_id,surface,"
                                 "action,target_type,target_id,reason,outcome,policy_checks,state_ref,"
                                 "prompt_version,skill_versions,data_origin,demo_run_id,correlation_id,"
                                 "turn_id,tool_execution_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"),
        )
        for name, sql in inserts:
            if buckets[name]:
                self.store.executemany(sql, buckets[name])
                self.counts.add(name, len(buckets[name]))

    @staticmethod
    def _redeemed_promotion(
        lines: list[tuple[str, int]], prices: dict[str, int], categories: dict[str, str]
    ) -> tuple[str | None, int]:
        """The promotion this basket qualifies for, and what it takes off.

        A scoped promotion is measured against the lines it covers, not the whole
        basket: "personal audio above ₹5,000" means ₹5,000 of personal audio. When
        more than one code qualifies the shopper uses the better one; ties break on
        code so the choice is reproducible. Returns (None, 0) when nothing qualifies,
        which is why an order can carry no promotion at all.
        """
        best: tuple[str | None, int] = (None, 0)
        for code, _description, kind, value, minimum, category in domain.PROMOTIONS:
            covered = sum(
                prices.get(variant_id, 99900) * quantity
                for variant_id, quantity in lines
                if category is None or categories.get(variant_id) == f"{SEED_PREFIX}{category}"
            )
            if covered < minimum:
                continue
            discount = value if kind == "fixed_minor" else round(covered * value / 100)
            discount = min(discount, covered)
            if discount > best[1]:
                best = (f"{SEED_PREFIX}promo_{code}", discount)
        return best

    def _pairings(self, line_variants: dict[str, list[str]]) -> dict[str, list[str]]:
        """Which variants genuinely complement which, from the domain's own pairings."""
        pairings: dict[str, list[str]] = {}
        for line in domain.LINES:
            partners: list[str] = []
            for partner_key in line.pairs_with:
                partners.extend(line_variants.get(partner_key, []))
            if not partners:
                continue
            for variant_id in line_variants.get(line.key, []):
                pairings[variant_id] = partners
        return pairings

    def _one_journey(self, day: datetime, index: int, sellable: list[str],
                     customer_ids: list[str], prices: dict[str, int],
                     categories: dict[str, str], pairings: dict[str, list[str]],
                     buckets: dict[str, list[tuple]]) -> None:
        """One shopper's session: browse, maybe buy, maybe fail, maybe return."""
        rng = self.rng
        customer_id = rng.choice(customer_ids)
        suffix = f"{index:05d}"
        correlation = f"{SEED_PREFIX}corr_{suffix}"
        moment = day + timedelta(hours=rng.randint(8, 22), minutes=rng.randint(0, 59))

        conversation_id = f"{SEED_PREFIX}conv_{suffix}"
        turn_id = f"{SEED_PREFIX}turn_{suffix}"
        buckets["conversations"].append((conversation_id, customer_id, "shopping", _iso(moment)))
        buckets["turns"].append((
            turn_id, conversation_id, 0, "completed", "I'm looking for something",
            "Here are the closest matches from the catalogue.", "shopping@1.0.0",
            "tools@1.0.0", json.dumps([]), _iso(moment), _iso(moment + timedelta(seconds=4))))

        # What they were shown.
        shown = rng.sample(sellable, min(4, len(sellable)))
        presentation_id = f"{SEED_PREFIX}pres_{suffix}"
        buckets["presentations"].append((
            presentation_id, conversation_id, customer_id, "products", turn_id,
            _iso(moment), _iso(moment + timedelta(hours=1))))
        for position, variant_id in enumerate(shown):
            buckets["presentation_items"].append((
                f"{presentation_id}_i{position}", presentation_id, position, variant_id,
                prices.get(variant_id, 99900)))

        chosen = shown[0]
        lines = [(chosen, 1)]

        # One bounded cross-sell, presented separately, sometimes accepted. The
        # split is what makes attribution honest: a presented recommendation is
        # not revenue until the customer actually adds it (ADR 0019).
        recommendation_id = None
        accepted = False
        partners = [p for p in pairings.get(chosen, []) if p in sellable]
        if partners:
            partner = rng.choice(partners)
            rec_presentation = f"{SEED_PREFIX}pres_{suffix}_x"
            rec_item = f"{rec_presentation}_i0"
            buckets["presentations"].append((
                rec_presentation, conversation_id, customer_id, "suggestions", turn_id,
                _iso(moment), _iso(moment + timedelta(hours=1))))
            buckets["presentation_items"].append((
                rec_item, rec_presentation, 0, partner, prices.get(partner, 49900)))
            accepted = rng.random() < 0.34
            recommendation_id = f"{SEED_PREFIX}rec_{suffix}"
            buckets["recommendations"].append((
                recommendation_id, rec_item, customer_id, partner,
                "Pairs with the item already in the cart.", chosen,
                _iso(moment + timedelta(minutes=1)) if accepted else None, _iso(moment)))
            if accepted:
                lines.append((partner, 1))

        if rng.random() < 0.28:
            # Browsed and left. Recorded, because a session that produced nothing
            # is still evidence — and it is what conversion is measured against.
            buckets["commerce_events"].append((
                f"{SEED_PREFIX}evt_{suffix}_b", _iso(moment), "session_browsed", "conversation",
                conversation_id, customer_id, None, None, "seeded", None, correlation, None))
            return

        subtotal = sum(prices.get(variant_id, 99900) * quantity for variant_id, quantity in lines)
        option, shipping, promised_days = domain.FULFILMENT_OPTIONS[
            0 if subtotal >= 200000 else rng.randint(0, 1)]
        tax = round(subtotal * 0.18)
        # Not every shopper arrives with a code, but the ones who do get a discount
        # their order genuinely qualifies for, computed from that promotion's own
        # rule. A discount no promotion explains is a number nothing can attribute.
        promotion_id, discount = (
            self._redeemed_promotion(lines, prices, categories)
            if rng.random() < 0.25 else (None, 0))
        total = subtotal + shipping + tax - discount

        stage_id = f"{SEED_PREFIX}stage_{suffix}"
        order_id = f"{SEED_PREFIX}ord_{suffix}"
        buckets["checkout_stages"].append((
            stage_id, f"{SEED_PREFIX}cart_{suffix}", customer_id, 1, "confirmed", domain.CURRENCY,
            subtotal, shipping, tax, discount, total, promotion_id, option, None,
            _iso(moment + timedelta(minutes=15)), _iso(moment), _iso(moment + timedelta(minutes=2))))
        for variant_id, quantity in lines:
            unit = prices.get(variant_id, 99900)
            buckets["checkout_stage_lines"].append((stage_id, variant_id, quantity, unit, unit * quantity))

        outcome = rng.random()
        paid = outcome < 0.74
        cancelled = not paid and outcome < 0.90  # the rest stay pending_payment

        status = "paid" if paid else ("cancelled" if cancelled else "pending_payment")
        paid_at = _iso(moment + timedelta(minutes=6)) if paid else None
        cancelled_at = _iso(moment + timedelta(minutes=30)) if cancelled else None
        buckets["commerce_orders"].append((
            order_id, customer_id, stage_id, status, domain.CURRENCY, subtotal, shipping, tax,
            discount, total, total if paid else 0, promotion_id, "seeded", 1, _iso(moment),
            paid_at, cancelled_at))
        buckets["commerce_events"].append((
            f"{SEED_PREFIX}evt_{suffix}_c", _iso(moment), "order_created", "order", order_id,
            customer_id, total, None, "seeded", None, correlation, None))

        order_line_ids = []
        for position, (variant_id, quantity) in enumerate(lines):
            unit = prices.get(variant_id, 99900)
            line_id = f"{order_id}_l{position}"
            order_line_ids.append((line_id, quantity))
            # Attribution is attached only to the accepted cross-sell line.
            attributed = recommendation_id if (accepted and position == len(lines) - 1) else None
            buckets["commerce_order_lines"].append((
                line_id, order_id, variant_id, quantity, unit, unit * quantity, attributed))

        # A failed first attempt before a successful retry, sometimes.
        attempt_index = 0
        if paid and rng.random() < 0.18:
            buckets["payment_attempts"].append((
                f"{SEED_PREFIX}pay_{suffix}_{attempt_index}", order_id, "razorpay",
                f"plink_{suffix}_{attempt_index}", f"https://rzp.io/{suffix}{attempt_index}",
                "failed", total, domain.CURRENCY, json.dumps({"seeded": True}), "card_declined",
                _iso(moment + timedelta(minutes=3)), _iso(moment + timedelta(minutes=4))))
            attempt_index += 1

        final_status = "succeeded" if paid else ("cancelled" if cancelled else "pending")
        attempt_id = f"{SEED_PREFIX}pay_{suffix}_{attempt_index}"
        buckets["payment_attempts"].append((
            attempt_id, order_id, "razorpay", f"plink_{suffix}_{attempt_index}",
            f"https://rzp.io/{suffix}{attempt_index}", final_status, total, domain.CURRENCY,
            json.dumps({"seeded": True}), None if paid else "abandoned",
            _iso(moment + timedelta(minutes=5)),
            _iso(moment + timedelta(minutes=6)) if final_status != "pending" else None))

        if not paid:
            buckets["evidence_records"].append(self._evidence(
                suffix, "b", moment, customer_id, "confirm_payment",
                "order", order_id, "Payment was not completed",
                "failed" if cancelled else "blocked", correlation, turn_id))
            return

        # A verified provider event is what made it paid.
        buckets["inbox_events"].append((
            f"{SEED_PREFIX}in_{suffix}", "razorpay", f"evt_{suffix}", "payment_link.paid",
            json.dumps({"amount": total, "currency": "INR", "reference": f"plink_{suffix}_{attempt_index}"}),
            "processed", None, _iso(moment + timedelta(minutes=6)),
            _iso(moment + timedelta(minutes=6, seconds=2))))
        buckets["commerce_events"].append((
            f"{SEED_PREFIX}evt_{suffix}_p", paid_at, "order_paid", "order", order_id,
            customer_id, total, None, "seeded", None, correlation, None))
        buckets["evidence_records"].append(self._evidence(
            suffix, "p", moment, customer_id, "settle_payment_attempt", "order", order_id,
            "Verified provider outcome for this attempt", "applied", correlation, turn_id))
        if accepted:
            buckets["commerce_events"].append((
                f"{SEED_PREFIX}evt_{suffix}_r", paid_at, "recommendation_converted",
                "recommendation", recommendation_id, customer_id,
                prices.get(lines[-1][0], 49900), 1, "seeded", None, correlation, None))

        # Fulfilment, and occasionally a refund after delivery.
        fulfillment_id = f"{SEED_PREFIX}ful_{suffix}"
        shipped = moment + timedelta(days=1)
        delivered = moment + timedelta(days=promised_days)
        delivered_yet = delivered <= self.as_of
        buckets["fulfillments"].append((
            fulfillment_id, order_id, "delivered" if delivered_yet else "shipped",
            "BlueDart", f"BD{suffix}", _iso(delivered), _iso(shipped),
            _iso(delivered) if delivered_yet else None, _iso(moment + timedelta(hours=6))))
        for line_id, quantity in order_line_ids:
            buckets["fulfillment_lines"].append((fulfillment_id, line_id, quantity))

        if delivered_yet and rng.random() < 0.07:
            buckets["refunds"].append((
                f"{SEED_PREFIX}ref_{suffix}", order_id, attempt_id, total,
                "Customer returned the item within the window", "completed",
                f"rfnd_{suffix}", _iso(delivered + timedelta(days=1)),
                _iso(delivered + timedelta(days=2))))
            buckets["commerce_events"].append((
                f"{SEED_PREFIX}evt_{suffix}_f", _iso(delivered + timedelta(days=2)),
                "order_refunded", "order", order_id, customer_id, total, None, "seeded",
                None, correlation, None))

    @staticmethod
    def _evidence(suffix: str, tag: str, moment: datetime, customer_id: str, action: str,
                  target_type: str, target_id: str, reason: str, outcome: str,
                  correlation: str, turn_id: str) -> tuple:
        return (
            f"{SEED_PREFIX}ev_{suffix}_{tag}", _iso(moment), "customer", customer_id, "shopping",
            action, target_type, target_id, reason, outcome, None, None, "shopping@1.0.0",
            None, "seeded", None, correlation, turn_id, None)
