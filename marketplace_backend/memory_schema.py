"""Customer and merchant memory (Cognee integration), the host's own state.

Postgres is the system of record for memory: what the "What Cartisan remembers"
panel lists, what a person can delete, and what the agent's brief is built from.
Cognee Cloud is the semantic layer fed from here through the outbox. Kept as its own
fragment so it ships as its own migration, and folded into CORE_DDL by
`core_schema` so the SQLite schema the tests run against includes it.
"""

MEMORY_DDL = """
-- ============================================================ memory

-- Whose memory this is. `id` is the verified customer id, `guest:<anon id>` for a
-- signed guest cookie, or `merchant_ops`. `dataset_name` is a keyed hash of the id,
-- so the Cognee dataset never carries the principal id itself.
create table if not exists memory_subjects (
  id text primary key,
  kind text not null check (kind in ('customer', 'guest', 'merchant')),
  dataset_name text not null unique,
  cognee_dataset_id text,
  purge_generation integer not null default 0,
  merged_into text,
  last_active_at timestamptz not null default now(),
  created_at timestamptz not null default now()
);

-- Typed facts. A superseded fact is closed (`valid_to` set), not deleted, so the
-- history stays explainable; a person deleting a fact removes its rows outright.
create table if not exists memory_facts (
  id text primary key,
  subject_id text not null,
  fact_key text not null,
  fact_type text not null check (fact_type in (
    'device_owned', 'brand_affinity', 'budget', 'use_case', 'gift_context',
    'rejected_item', 'current_project', 'lesson', 'outcome')),
  value text not null,
  category text not null default 'preference' check (category in
    ('preference', 'constraint', 'context')),
  source_session text,
  valid_from timestamptz not null default now(),
  valid_to timestamptz,
  synced_at timestamptz,
  updated_at timestamptz not null default now()
);

create index if not exists memory_facts_subject_idx on memory_facts (subject_id, fact_key);

-- The raw behaviour log. Only high-intent events are accepted; the weight is set
-- by the host from the event type, never taken from the client.
create table if not exists behavior_events (
  id text primary key,
  subject_id text not null,
  event_type text not null check (event_type in (
    'product_view', 'add_to_cart', 'remove_from_cart', 'purchase',
    'rejected_recommendation', 'suggestion_click')),
  variant_id text not null,
  product_id text,
  weight numeric not null,
  dwell_ms integer,
  correlation_id text,
  created_at timestamptz not null default now(),
  synced_at timestamptz
);

create index if not exists behavior_events_subject_idx on behavior_events (subject_id, created_at);
create index if not exists behavior_events_unsynced_idx on behavior_events (synced_at);

-- The precomputed brief the agent and the storefront read. A Cognee outage leaves
-- the last one in place.
create table if not exists memory_briefs (
  subject_id text primary key,
  brief text not null,
  refreshed_at timestamptz not null default now()
);

-- One row per batch handed to Cognee. The key is the dataset plus a hash of the
-- batch body, so a replayed outbox message is a no-op instead of a duplicate memory.
create table if not exists memory_sync_batches (
  idempotency_key text primary key,
  subject_id text not null,
  item_count integer not null,
  cognee_status text not null,
  created_at timestamptz not null default now()
);

-- Cognee calls per day, per subject and in total (`subject_id = '*'`).
create table if not exists memory_usage (
  day text not null,
  subject_id text not null,
  calls integer not null default 0,
  primary key (day, subject_id)
);

-- Ratings on agent answers, joined to the evidence ledger by correlation id.
create table if not exists memory_feedback (
  id text primary key,
  subject_id text not null,
  conversation_id text,
  turn_id text,
  correlation_id text,
  rating text not null check (rating in ('up', 'down')),
  reason text check (reason is null or reason in (
    'not_relevant', 'wrong_info', 'too_pushy', 'helpful', 'other')),
  note text,
  synced_at timestamptz,
  created_at timestamptz not null default now()
);

create index if not exists memory_feedback_subject_idx on memory_feedback (subject_id, created_at);
"""

# Cart recovery (Q3, Q4). The merchant approves a policy once, through the same
# maker-checker queue as every other change; a deterministic worker issues offers
# inside it. Each coupon is its own single-use promotion row, so attribution works
# exactly as it does for any other promotion.
RECOVERY_DDL = """
-- ==================================================== cart recovery

-- One active policy at a time; approving a new one retires the old.
create table if not exists recovery_policies (
  id text primary key,
  status text not null default 'active' check (status in ('active', 'retired')),
  abandon_after_minutes integer not null check (abandon_after_minutes >= 30),
  min_cart_minor integer not null check (min_cart_minor >= 0),
  discount_percentage integer not null check (discount_percentage between 1 and 20),
  max_discount_minor integer not null check (max_discount_minor > 0),
  cooldown_days integer not null check (cooldown_days >= 7),
  monthly_budget_minor integer not null check (monthly_budget_minor >= 0),
  offer_valid_hours integer not null check (offer_valid_hours between 1 and 168),
  change_id text,
  created_at timestamptz not null default now(),
  retired_at timestamptz
);

-- One offer per abandoned cart version. `kind = 'reminder'` is the no-discount
-- nudge for a shopper whose history says they buy without one.
create table if not exists recovery_offers (
  id text primary key,
  customer_id text not null,
  cart_id text not null,
  cart_state_version integer not null,
  policy_id text not null references recovery_policies (id),
  kind text not null check (kind in ('coupon', 'reminder')),
  promotion_id text references promotions (id),
  code text,
  discount_percentage integer,
  max_discount_minor integer,
  headline_variant_id text,
  status text not null default 'issued' check (status in ('issued', 'redeemed', 'expired')),
  email_status text not null default 'pending' check (email_status in
    ('pending', 'sent', 'skipped', 'failed')),
  redeemed_order_id text,
  created_at timestamptz not null default now(),
  expires_at timestamptz not null,
  redeemed_at timestamptz,
  unique (cart_id, cart_state_version)
);

create index if not exists recovery_offers_customer_idx on recovery_offers (customer_id, created_at);

-- Marketing email consent, separate from memory (Q4, Q8). No row means no email.
create table if not exists marketing_consents (
  customer_id text primary key,
  email_opt_in boolean not null default false,
  unsubscribe_token text not null unique,
  updated_at timestamptz not null default now()
);
"""
