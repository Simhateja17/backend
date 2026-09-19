"""The normalized commerce core (Phase 2), defined once.

The same DDL text produces the Supabase Postgres migration and the SQLite schema
the tests run against, so the two can never drift. It is written in Postgres SQL
restricted to constructs `to_sqlite` can translate: no dialect-specific functions
beyond `now()`, no partial-index syntax SQLite lacks, and no schema qualification
(the Store adds `cartisan.` for Postgres).

Design rules from CARTISAN_COMMERCE_ARCHITECTURE.md and ADR 0027:
  * Searchable or enforceable facts get typed, indexed columns.
  * JSONB is limited to genuinely category-specific attributes and to immutable
    provider snapshots, both marked `-- governed jsonb` below.
  * Transactional tables are current truth; `commerce_events` is append-only and
    derived metrics come from it; `evidence_records` explains operations without
    ever replacing authoritative state.
  * Money is stored in minor units (paise) as integers. Never floats.
"""

from __future__ import annotations

import re

CORE_DDL = """
-- ========================================================== seed runs

-- One row per generator run. `seed` and `generator_version` together determine
-- every generated row, so a reset with the same pair reproduces the same catalog
-- and the same ninety days of history, byte for byte.
create table if not exists seed_runs (
  id text primary key,
  generator_version text not null,
  seed integer not null,
  as_of text not null,
  record_counts text not null,
  created_at timestamptz not null default now()
);

-- Rows created by the named scenario packs. They run through the production
-- repositories, so they carry production id prefixes; this records what appeared
-- so a reset can remove them without the production code knowing about seeding.
create table if not exists seed_scenario_rows (
  id text primary key,
  seed_run_id text not null,
  table_name text not null,
  row_id text not null
);

create index if not exists seed_scenario_rows_run_idx on seed_scenario_rows (seed_run_id);

-- Seeded shoppers. Deliberately NOT `cartisan.customers`: that table mirrors
-- verified Supabase accounts and is uuid-keyed against auth.users, whereas these
-- exist only to give ninety days of history plausible owners. The commerce core
-- stores `customer_id` as text precisely so both kinds of principal can own rows.
create table if not exists seed_customers (
  id text primary key,
  email text not null,
  display_name text not null,
  created_at timestamptz not null default now()
);

-- ============================================================ catalog

create table if not exists catalog_categories (
  id text primary key,
  name text not null,
  parent_id text references catalog_categories (id),
  created_at timestamptz not null default now()
);

create table if not exists catalog_products (
  id text primary key,
  sku_root text not null unique,
  title text not null,
  brand text not null,
  category_id text not null references catalog_categories (id),
  description text not null,
  status text not null default 'active' check (status in ('draft', 'active', 'discontinued')),
  origin text not null default 'seeded' check (origin in ('seeded', 'live_app', 'razorpay_test')),
  created_at timestamptz not null default now()
);

create table if not exists catalog_variants (
  id text primary key,
  product_id text not null references catalog_products (id) on delete cascade,
  sku text not null unique,
  title text not null,
  -- governed jsonb: category-specific option values only (for example
  -- {"colour": "graphite", "capacity_gb": 512}). Anything searchable or
  -- enforceable is duplicated into variant_specs as a typed row.
  options text,
  status text not null default 'active' check (status in ('draft', 'active', 'discontinued')),
  created_at timestamptz not null default now()
);

create index if not exists catalog_variants_product_idx on catalog_variants (product_id);

-- Typed specifications. A value lands in exactly one of the typed columns, so a
-- numeric filter ("at least 30W") is an indexed comparison, not a JSON scan.
create table if not exists variant_specs (
  variant_id text not null references catalog_variants (id) on delete cascade,
  spec_key text not null,
  value_text text,
  value_numeric numeric,
  value_unit text,
  value_bool boolean,
  primary key (variant_id, spec_key),
  check (
    (case when value_text is null then 0 else 1 end)
    + (case when value_numeric is null then 0 else 1 end)
    + (case when value_bool is null then 0 else 1 end) = 1
  )
);

create index if not exists variant_specs_key_numeric_idx on variant_specs (spec_key, value_numeric);
create index if not exists variant_specs_key_text_idx on variant_specs (spec_key, value_text);

-- ==================================================== compatibility

-- Compatibility is decided from these rows alone (ADR 0006). A variant offers
-- capabilities; a variant requires capabilities. Nothing here is free text the
-- model can reinterpret.
create table if not exists capabilities (
  id text primary key,
  label text not null,
  value_kind text not null check (value_kind in ('text', 'numeric', 'bool'))
);

create table if not exists variant_capabilities (
  variant_id text not null references catalog_variants (id) on delete cascade,
  capability_id text not null references capabilities (id),
  value_text text,
  value_numeric numeric,
  value_bool boolean,
  primary key (variant_id, capability_id)
);

create table if not exists variant_requirements (
  id text primary key,
  variant_id text not null references catalog_variants (id) on delete cascade,
  capability_id text not null references capabilities (id),
  operator text not null check (operator in ('eq', 'neq', 'gte', 'lte', 'in', 'is_true')),
  value_text text,
  value_numeric numeric,
  severity text not null default 'blocking' check (severity in ('blocking', 'advisory')),
  explanation text not null
);

create index if not exists variant_requirements_variant_idx on variant_requirements (variant_id);

-- ============================================================= prices

create table if not exists variant_prices (
  id text primary key,
  variant_id text not null references catalog_variants (id) on delete cascade,
  currency text not null default 'INR',
  amount_minor integer not null check (amount_minor > 0),
  compare_at_minor integer check (compare_at_minor is null or compare_at_minor > 0),
  price_kind text not null default 'list' check (price_kind in ('list', 'promotional')),
  valid_from timestamptz not null default now(),
  valid_to timestamptz
);

create index if not exists variant_prices_lookup_idx on variant_prices (variant_id, valid_from);

-- ========================================================== inventory

create table if not exists inventory_locations (
  id text primary key,
  code text not null unique,
  name text not null,
  region text not null
);

-- on_hand is physical stock; reserved is the part already committed to confirmed
-- orders. Sellable is (on_hand - reserved) and is never stored, so it cannot drift.
create table if not exists inventory_levels (
  variant_id text not null references catalog_variants (id) on delete cascade,
  location_id text not null references inventory_locations (id),
  on_hand integer not null default 0 check (on_hand >= 0),
  reserved integer not null default 0 check (reserved >= 0),
  updated_at timestamptz not null default now(),
  primary key (variant_id, location_id),
  check (reserved <= on_hand)
);

-- Append-only. Every change to on_hand is explained by exactly one movement.
create table if not exists inventory_movements (
  id text primary key,
  variant_id text not null references catalog_variants (id),
  location_id text not null references inventory_locations (id),
  delta integer not null check (delta <> 0),
  reason text not null check (reason in
    ('receipt', 'sale', 'return', 'adjustment', 'damage', 'reservation_release')),
  reference_type text,
  reference_id text,
  created_at timestamptz not null default now()
);

create index if not exists inventory_movements_variant_idx on inventory_movements (variant_id, created_at);

-- Stock is reserved on checkout confirmation, never on cart addition (ADR 0012).
create table if not exists inventory_reservations (
  id text primary key,
  order_id text not null,
  variant_id text not null references catalog_variants (id),
  location_id text not null references inventory_locations (id),
  quantity integer not null check (quantity > 0),
  status text not null default 'held' check (status in ('held', 'consumed', 'released', 'expired')),
  expires_at timestamptz not null,
  created_at timestamptz not null default now(),
  resolved_at timestamptz
);

create index if not exists inventory_reservations_order_idx on inventory_reservations (order_id);
create index if not exists inventory_reservations_expiry_idx on inventory_reservations (status, expires_at);

-- ============================================ promotions and campaigns

-- Declared before checkout because a stage and an order each name the promotion
-- that explains their discount.
create table if not exists promotions (
  id text primary key,
  code text not null unique,
  description text not null,
  discount_kind text not null check (discount_kind in ('percentage', 'fixed_minor')),
  discount_value integer not null check (discount_value > 0),
  min_subtotal_minor integer not null default 0 check (min_subtotal_minor >= 0),
  -- The category the promotion is confined to; null is storewide. A promotion that
  -- says "personal audio above ₹5,000" has to be enforceable as written, or the
  -- description is decoration and the discount it explains is unexplained.
  category_id text references catalog_categories (id),
  status text not null default 'draft' check (status in ('draft', 'active', 'paused', 'ended')),
  starts_at timestamptz not null,
  ends_at timestamptz
);

create table if not exists campaigns (
  id text primary key,
  name text not null,
  channel text not null,
  promotion_id text references promotions (id),
  status text not null default 'draft' check (status in ('draft', 'running', 'paused', 'ended')),
  budget_minor integer not null default 0 check (budget_minor >= 0),
  spend_minor integer not null default 0 check (spend_minor >= 0),
  starts_at timestamptz not null,
  ends_at timestamptz
);

-- =========================================================== checkout

-- An immutable, expiring preview. Staging moves no money and reserves no stock.
create table if not exists checkout_stages (
  id text primary key,
  cart_id text not null,
  customer_id text not null,
  cart_state_version integer not null,
  state text not null default 'staged' check (state in ('staged', 'confirmed', 'expired', 'superseded')),
  currency text not null default 'INR',
  subtotal_minor integer not null check (subtotal_minor >= 0),
  shipping_minor integer not null default 0 check (shipping_minor >= 0),
  tax_minor integer not null default 0 check (tax_minor >= 0),
  discount_minor integer not null default 0 check (discount_minor >= 0),
  total_minor integer not null check (total_minor >= 0),
  -- Which promotion produced `discount_minor`, so the discount is explained by a
  -- rule rather than appearing as a bare number. Null means no promotion applied.
  promotion_id text references promotions (id),
  fulfillment_option text not null,
  constraints_note text,
  expires_at timestamptz not null,
  created_at timestamptz not null default now(),
  resolved_at timestamptz
);

create index if not exists checkout_stages_customer_idx on checkout_stages (customer_id, created_at);

create table if not exists checkout_stage_lines (
  stage_id text not null references checkout_stages (id) on delete cascade,
  variant_id text not null references catalog_variants (id),
  quantity integer not null check (quantity > 0),
  unit_price_minor integer not null check (unit_price_minor > 0),
  amount_minor integer not null check (amount_minor > 0),
  primary key (stage_id, variant_id)
);

-- ============================================================= orders

create table if not exists commerce_orders (
  id text primary key,
  customer_id text not null,
  stage_id text references checkout_stages (id),
  status text not null default 'pending_payment' check (status in
    ('pending_payment', 'payment_verification_pending', 'paid', 'cancelled', 'expired', 'refunded')),
  currency text not null default 'INR',
  subtotal_minor integer not null check (subtotal_minor >= 0),
  shipping_minor integer not null default 0 check (shipping_minor >= 0),
  tax_minor integer not null default 0 check (tax_minor >= 0),
  discount_minor integer not null default 0 check (discount_minor >= 0),
  total_minor integer not null check (total_minor >= 0),
  amount_paid_minor integer not null default 0 check (amount_paid_minor >= 0),
  -- The promotion redeemed on this order, carried over from its stage. It is what
  -- makes campaign attribution a recorded fact rather than an inference: an order
  -- is attributed to a campaign only through the promotion it actually redeemed.
  -- Nullable: no promotion, or an order created before this column existed.
  promotion_id text references promotions (id),
  origin text not null default 'live_app' check (origin in ('seeded', 'live_app', 'razorpay_test')),
  state_version integer not null default 0 check (state_version >= 0),
  -- The lineage the order was created under, so the journey that produced it can be
  -- followed from the browser request through to the provider event (ADR 0032).
  -- Nullable: an order created before this column existed still has to read.
  correlation_id text,
  demo_run_id text,
  created_at timestamptz not null default now(),
  paid_at timestamptz,
  cancelled_at timestamptz
);

create index if not exists commerce_orders_correlation_idx on commerce_orders (correlation_id);
create index if not exists commerce_orders_promotion_idx on commerce_orders (promotion_id, created_at);

create index if not exists commerce_orders_customer_idx on commerce_orders (customer_id, created_at);
create index if not exists commerce_orders_status_idx on commerce_orders (status, created_at);

create table if not exists commerce_order_lines (
  id text primary key,
  order_id text not null references commerce_orders (id) on delete cascade,
  variant_id text not null references catalog_variants (id),
  quantity integer not null check (quantity > 0),
  unit_price_minor integer not null check (unit_price_minor > 0),
  amount_minor integer not null check (amount_minor > 0),
  -- Set only when this line traces to a recommendation the customer explicitly
  -- accepted; it is the sole basis for agent-assisted revenue (ADR 0019).
  recommendation_id text
);

create index if not exists commerce_order_lines_order_idx on commerce_order_lines (order_id);

-- ==================================================== payment attempts

-- Many attempts belong to one order. An attempt is never the order's truth: the
-- order becomes paid only from a verified provider outcome (ADR 0013).
create table if not exists payment_attempts (
  id text primary key,
  order_id text not null references commerce_orders (id) on delete cascade,
  provider text not null default 'razorpay',
  provider_reference text,
  provider_link_url text,
  status text not null default 'created' check (status in
    ('created', 'pending', 'succeeded', 'failed', 'cancelled', 'expired')),
  amount_minor integer not null check (amount_minor > 0),
  currency text not null default 'INR',
  -- governed jsonb: an immutable snapshot of the provider payload, kept verbatim
  -- for evidence. Never read as authority for order state.
  provider_snapshot text,
  failure_reason text,
  -- Carried so the provider's answer rejoins the journey that asked for the link.
  -- A webhook knows only a provider reference; this is how that reference leads
  -- back to the turn the customer started.
  correlation_id text,
  demo_run_id text,
  created_at timestamptz not null default now(),
  resolved_at timestamptz
);

create index if not exists payment_attempts_order_idx on payment_attempts (order_id, created_at);
create index if not exists payment_attempts_reference_idx on payment_attempts (provider_reference);

-- ======================================================= fulfillment

create table if not exists fulfillments (
  id text primary key,
  order_id text not null references commerce_orders (id) on delete cascade,
  status text not null default 'pending' check (status in
    ('pending', 'packed', 'shipped', 'delivered', 'cancelled')),
  carrier text,
  tracking_reference text,
  promised_at timestamptz,
  shipped_at timestamptz,
  delivered_at timestamptz,
  created_at timestamptz not null default now()
);

create table if not exists fulfillment_lines (
  fulfillment_id text not null references fulfillments (id) on delete cascade,
  order_line_id text not null references commerce_order_lines (id),
  quantity integer not null check (quantity > 0),
  primary key (fulfillment_id, order_line_id)
);

create table if not exists refunds (
  id text primary key,
  order_id text not null references commerce_orders (id) on delete cascade,
  payment_attempt_id text references payment_attempts (id),
  amount_minor integer not null check (amount_minor > 0),
  reason text not null,
  status text not null default 'requested' check (status in
    ('requested', 'processing', 'completed', 'failed')),
  provider_reference text,
  created_at timestamptz not null default now(),
  resolved_at timestamptz
);

-- ======================================== presentations and lineage

-- Server-issued item references. A conversational phrase such as "the second one"
-- resolves to a presentation_items row before any mutation (ADR 0020).
create table if not exists presentations (
  id text primary key,
  conversation_id text not null,
  customer_id text not null,
  kind text not null check (kind in
    ('products', 'comparison', 'cart', 'checkout', 'order_status', 'guide', 'suggestions')),
  turn_id text,
  created_at timestamptz not null default now(),
  expires_at timestamptz not null
);

create index if not exists presentations_conversation_idx on presentations (conversation_id, created_at);

create table if not exists presentation_items (
  id text primary key,
  presentation_id text not null references presentations (id) on delete cascade,
  position integer not null check (position >= 0),
  variant_id text not null references catalog_variants (id),
  unit_price_minor integer not null check (unit_price_minor > 0)
);

create index if not exists presentation_items_presentation_idx on presentation_items (presentation_id, position);

create table if not exists recommendations (
  id text primary key,
  presentation_item_id text not null references presentation_items (id) on delete cascade,
  customer_id text not null,
  variant_id text not null references catalog_variants (id),
  rationale text not null,
  source_variant_id text references catalog_variants (id),
  accepted_at timestamptz,
  created_at timestamptz not null default now()
);

create index if not exists recommendations_customer_idx on recommendations (customer_id, created_at);

-- =================================================== commerce events

-- Append-only. Derived metrics are computed from these rows, never by mutating a
-- running total (ADR 0018).
create table if not exists commerce_events (
  id text primary key,
  occurred_at timestamptz not null default now(),
  event_type text not null,
  subject_type text not null,
  subject_id text not null,
  customer_id text,
  amount_minor integer,
  quantity integer,
  origin text not null default 'live_app' check (origin in ('seeded', 'live_app', 'razorpay_test')),
  demo_run_id text,
  correlation_id text,
  -- governed jsonb: event-shape-specific detail. Every metric-bearing field above
  -- is a typed column, so metrics never parse this.
  detail text
);

create index if not exists commerce_events_type_time_idx on commerce_events (event_type, occurred_at);
create index if not exists commerce_events_subject_idx on commerce_events (subject_type, subject_id);
create index if not exists commerce_events_origin_idx on commerce_events (origin, occurred_at);

-- ============================================ staged merchant changes

create table if not exists merchant_changes (
  id text primary key,
  operator_id text not null,
  -- The conversation the proposal was staged in. A decided change is part of that
  -- conversation's history and is shown with it; a pending one is outstanding work
  -- and is shown to the operator wherever they are. Nullable because a change staged
  -- outside a conversation (a script, a scenario pack) still belongs in the queue.
  conversation_id text,
  kind text not null check (kind in
    ('inventory_action', 'price_update', 'promotion', 'campaign', 'listing_update',
     'recovery_policy', 'loan_request')),
  target_type text not null,
  target_id text,
  -- governed jsonb: the exact before/after documents shown on the approval surface.
  before_doc text not null,
  after_doc text not null,
  rationale text not null,
  status text not null default 'pending' check (status in
    ('pending', 'approved', 'rejected', 'applied', 'failed', 'superseded')),
  created_at timestamptz not null default now(),
  decided_at timestamptz,
  applied_at timestamptz
);

create index if not exists merchant_changes_status_idx on merchant_changes (status, created_at);
create index if not exists merchant_changes_conversation_idx on merchant_changes (conversation_id, created_at);

create table if not exists merchant_approvals (
  id text primary key,
  change_id text not null references merchant_changes (id) on delete cascade,
  operator_id text not null,
  decision text not null check (decision in ('approved', 'rejected')),
  note text,
  -- Why, in a word the lessons can count (merchant memory). Optional.
  reason_code text check (reason_code is null or reason_code in
    ('margin_too_low', 'bad_timing', 'brand_policy', 'stock_risk', 'other')),
  policy_checks text,
  decided_at timestamptz not null default now()
);

-- ================================================ agent conversations

create table if not exists conversations (
  id text primary key,
  principal_id text not null,
  surface text not null check (surface in ('shopping', 'merchant')),
  created_at timestamptz not null default now()
);

create table if not exists turns (
  id text primary key,
  conversation_id text not null references conversations (id) on delete cascade,
  sequence integer not null check (sequence >= 0),
  state text not null default 'received' check (state in
    ('received', 'running', 'awaiting_tool', 'completed', 'failed', 'abandoned')),
  user_message text,
  agent_message text,
  prompt_version text not null,
  tool_contract_version text not null,
  skill_versions text,
  input_tokens integer,
  output_tokens integer,
  cache_read_tokens integer,
  -- The turn's own lineage id, minted with the turn and carried by everything the
  -- turn causes. Stored rather than held in memory, so a journey is followable
  -- after the process that served it has gone.
  correlation_id text,
  demo_run_id text,
  started_at timestamptz not null default now(),
  completed_at timestamptz
);

create index if not exists turns_correlation_idx on turns (correlation_id);

create index if not exists turns_conversation_idx on turns (conversation_id, sequence);

create table if not exists tool_executions (
  id text primary key,
  turn_id text not null references turns (id) on delete cascade,
  tool_name text not null,
  -- governed jsonb: the exact arguments and result envelope, for replay and audit.
  arguments text not null,
  outcome text not null check (outcome in
    ('applied', 'blocked', 'unavailable', 'failed', 'conflict')),
  result text,
  latency_ms integer,
  created_at timestamptz not null default now()
);

create index if not exists tool_executions_turn_idx on tool_executions (turn_id, created_at);

-- ==================================================== outbox / inbox

-- External effects leave through here, in the same transaction as the internal
-- write that caused them (ADR 0024).
create table if not exists outbox_messages (
  id text primary key,
  topic text not null,
  -- governed jsonb: the message body handed to the external effect.
  payload text not null,
  status text not null default 'pending' check (status in
    ('pending', 'in_flight', 'delivered', 'failed', 'dead_letter')),
  attempts integer not null default 0 check (attempts >= 0),
  correlation_id text,
  last_error text,
  available_at timestamptz not null default now(),
  created_at timestamptz not null default now(),
  delivered_at timestamptz
);

create index if not exists outbox_ready_idx on outbox_messages (status, available_at);

-- Provider callbacks land here first and are deduplicated by provider event id,
-- so a redelivered webhook is recorded once and applied once.
create table if not exists inbox_events (
  id text primary key,
  provider text not null default 'razorpay',
  provider_event_id text not null,
  event_type text not null,
  -- governed jsonb: the verbatim provider payload. Immutable evidence.
  payload text not null,
  status text not null default 'received' check (status in
    ('received', 'processed', 'ignored', 'quarantined')),
  quarantine_reason text,
  -- Set once the event is matched to an attempt, so a quarantined callback reads as
  -- part of the journey it failed rather than as a loose row a human has to join.
  correlation_id text,
  received_at timestamptz not null default now(),
  processed_at timestamptz
);

create unique index if not exists inbox_events_provider_event_idx
  on inbox_events (provider, provider_event_id);

-- ================================================== evidence ledger

-- Append-only. One row per agent tool invocation and per meaningful commerce
-- transition, including blocked, unavailable and failed outcomes (ADR 0023).
create table if not exists evidence_records (
  id text primary key,
  recorded_at timestamptz not null default now(),
  actor_type text not null check (actor_type in ('customer', 'merchant_operator', 'agent', 'system', 'provider')),
  actor_id text,
  surface text,
  action text not null,
  target_type text,
  target_id text,
  reason text not null,
  outcome text not null check (outcome in
    ('applied', 'blocked', 'unavailable', 'failed', 'conflict')),
  -- governed jsonb: the policy checks evaluated and the state references read.
  policy_checks text,
  state_ref text,
  prompt_version text,
  skill_versions text,
  data_origin text not null default 'live_app' check (data_origin in ('seeded', 'live_app', 'razorpay_test')),
  demo_run_id text,
  correlation_id text,
  turn_id text,
  tool_execution_id text
);

create index if not exists evidence_actor_idx on evidence_records (actor_id, recorded_at);
create index if not exists evidence_correlation_idx on evidence_records (correlation_id);
create index if not exists evidence_target_idx on evidence_records (target_type, target_id);
create index if not exists evidence_origin_idx on evidence_records (data_origin, recorded_at);
create index if not exists evidence_demo_run_idx on evidence_records (demo_run_id, recorded_at);
create index if not exists evidence_surface_idx on evidence_records (surface, recorded_at);
"""

# The Telegram channel (n8n is the transport; this is the host's own state). Kept as
# its own fragment so it ships as its own migration, and folded into CORE_DDL so
# the SQLite schema the tests run against includes it.
TELEGRAM_DDL = """
-- ================================================== telegram channel

-- Which conversation a chat is on. `/new` bumps the sequence; the transcript
-- itself lives in `conversations`/`turns` like every other surface.
create table if not exists telegram_chats (
  bot_kind text not null check (bot_kind in ('shopping', 'merchant')),
  chat_id text not null,
  conversation_seq integer not null default 0,
  primary key (bot_kind, chat_id)
);

-- A Telegram account bound to a verified Cartisan principal. Written only by the
-- link-redemption path, which reads the principal from a Supabase session.
create table if not exists telegram_links (
  bot_kind text not null check (bot_kind in ('shopping', 'merchant')),
  telegram_user_id text not null,
  chat_id text not null,
  principal_id text not null,
  role text not null,
  linked_at timestamptz not null default now(),
  primary key (bot_kind, telegram_user_id)
);

-- Single-use, short-lived link tokens. Only the hash is stored.
create table if not exists telegram_link_tokens (
  token_hash text primary key,
  bot_kind text not null,
  telegram_user_id text not null,
  chat_id text not null,
  expires_at text not null,
  redeemed_at text,
  redeemed_by text
);

-- Telegram redelivers; each update is acted on once.
create table if not exists telegram_updates (
  bot_kind text not null,
  update_id text not null,
  received_at timestamptz not null default now(),
  primary key (bot_kind, update_id)
);

-- Inline-button references. `callback_data` carries only this id; what the button
-- does, and for whom, is read back from here.
create table if not exists telegram_actions (
  id text primary key,
  bot_kind text not null,
  chat_id text not null,
  action text not null,
  args text not null,
  created_at timestamptz not null default now()
);

-- Proactive messages already handed to the transport, so each goes out once.
create table if not exists telegram_notifications (
  kind text not null,
  ref_id text not null,
  chat_id text not null,
  created_at timestamptz not null default now(),
  primary key (kind, ref_id, chat_id)
);
"""

CORE_DDL += TELEGRAM_DDL

from .memory_schema import MEMORY_DDL, RECOVERY_DDL  # noqa: E402  (fragments, folded in like Telegram's)

CORE_DDL += MEMORY_DDL + RECOVERY_DDL


def table_names() -> tuple[str, ...]:
    """Every table the core defines, for the Store's schema-qualification pass."""
    return tuple(re.findall(r"create table if not exists (\w+)", CORE_DDL))


def to_sqlite(ddl: str = CORE_DDL) -> str:
    """Translate the core DDL to the SQLite dialect the tests run against.

    Only type spellings and `now()` differ; the constraints, checks and indexes
    are identical, so a constraint proven in a test is the constraint Postgres
    enforces in production.
    """
    substitutions = (
        (r"\btimestamptz\b", "text"),
        (r"\bnumeric\b", "real"),
        (r"\bboolean\b", "integer"),
        (r"\bnow\(\)", "(datetime('now'))"),
    )

    def translate(fragment: str) -> str:
        for pattern, replacement in substitutions:
            fragment = re.sub(pattern, replacement, fragment)
        return fragment

    # Translate code only. A comment may contain an apostrophe, and a quoted literal
    # may contain a type name — a CHECK list such as
    # `value_kind in ('text', 'numeric', 'bool')` must keep the domain values it
    # constrains rather than having them renamed to SQLite types. So each line is
    # split at its comment marker first, then the code half is split on literals.
    output = []
    for line in ddl.split("\n"):
        code, marker, comment = line.partition("--")
        parts = re.split(r"('(?:[^']|'')*')", code)
        translated_code = "".join(
            part if index % 2 else translate(part) for index, part in enumerate(parts))
        output.append(translated_code + marker + comment)
    return "\n".join(output)


def to_postgres(ddl: str = CORE_DDL) -> str:
    """Qualify the core DDL into the `cartisan` schema for the migration file."""
    qualified = ddl
    for table in table_names():
        qualified = re.sub(rf"\b{table}\b", f"cartisan.{table}", qualified)
    return qualified
