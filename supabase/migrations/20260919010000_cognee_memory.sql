-- Cognee memory. Generated from marketplace_backend/memory_schema.py (MEMORY_DDL).

-- ============================================================ memory

-- Whose memory this is. `id` is the verified customer id, `guest:<anon id>` for a
-- signed guest cookie, or `merchant_ops`. `dataset_name` is a keyed hash of the id,
-- so the Cognee dataset never carries the principal id itself.
create table if not exists cartisan.memory_subjects (
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
create table if not exists cartisan.memory_facts (
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

create index if not exists memory_facts_subject_idx on cartisan.memory_facts (subject_id, fact_key);

-- The raw behaviour log. Only high-intent events are accepted; the weight is set
-- by the host from the event type, never taken from the client.
create table if not exists cartisan.behavior_events (
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

create index if not exists behavior_events_subject_idx on cartisan.behavior_events (subject_id, created_at);
create index if not exists behavior_events_unsynced_idx on cartisan.behavior_events (synced_at);

-- The precomputed brief the agent and the storefront read. A Cognee outage leaves
-- the last one in place.
create table if not exists cartisan.memory_briefs (
  subject_id text primary key,
  brief text not null,
  refreshed_at timestamptz not null default now()
);

-- One row per batch handed to Cognee. The key is the dataset plus a hash of the
-- batch body, so a replayed outbox message is a no-op instead of a duplicate memory.
create table if not exists cartisan.memory_sync_batches (
  idempotency_key text primary key,
  subject_id text not null,
  item_count integer not null,
  cognee_status text not null,
  created_at timestamptz not null default now()
);

-- Cognee calls per day, per subject and in total (`subject_id = '*'`).
create table if not exists cartisan.memory_usage (
  day text not null,
  subject_id text not null,
  calls integer not null default 0,
  primary key (day, subject_id)
);

-- Ratings on agent answers, joined to the evidence ledger by correlation id.
create table if not exists cartisan.memory_feedback (
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

create index if not exists memory_feedback_subject_idx on cartisan.memory_feedback (subject_id, created_at);

-- Personal data, backend-only: RLS on with no policies closes these tables to the
-- anon and authenticated roles; the backend connects as the owner role.
alter table cartisan.memory_subjects enable row level security;
alter table cartisan.memory_facts enable row level security;
alter table cartisan.behavior_events enable row level security;
alter table cartisan.memory_briefs enable row level security;
alter table cartisan.memory_sync_batches enable row level security;
alter table cartisan.memory_usage enable row level security;
alter table cartisan.memory_feedback enable row level security;
