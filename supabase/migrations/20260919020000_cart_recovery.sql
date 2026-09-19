-- Cart recovery and merchant memory. Tables generated from marketplace_backend/memory_schema.py (RECOVERY_DDL).

-- ==================================================== cart recovery

-- One active policy at a time; approving a new one retires the old.
create table if not exists cartisan.recovery_policies (
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
create table if not exists cartisan.recovery_offers (
  id text primary key,
  customer_id text not null,
  cart_id text not null,
  cart_state_version integer not null,
  policy_id text not null references cartisan.recovery_policies (id),
  kind text not null check (kind in ('coupon', 'reminder')),
  promotion_id text references cartisan.promotions (id),
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

create index if not exists recovery_offers_customer_idx on cartisan.recovery_offers (customer_id, created_at);

-- Marketing email consent, separate from memory (Q4, Q8). No row means no email.
create table if not exists cartisan.marketing_consents (
  customer_id text primary key,
  email_opt_in boolean not null default false,
  unsubscribe_token text not null unique,
  updated_at timestamptz not null default now()
);

-- A recovery policy is a merchant change like any other: staged, then approved.
alter table cartisan.merchant_changes drop constraint if exists merchant_changes_kind_check;
alter table cartisan.merchant_changes add constraint merchant_changes_kind_check check (kind in
  ('inventory_action', 'price_update', 'promotion', 'campaign', 'listing_update', 'recovery_policy'));

-- Why an operator decided, in a word merchant memory can count.
alter table cartisan.merchant_approvals add column if not exists reason_code text
  check (reason_code is null or reason_code in
    ('margin_too_low', 'bad_timing', 'brand_policy', 'stock_risk', 'other'));

-- Offers and consent are personal data, backend-only.
alter table cartisan.recovery_policies enable row level security;
alter table cartisan.recovery_offers enable row level security;
alter table cartisan.marketing_consents enable row level security;
