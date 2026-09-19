-- A Paytm merchant loan request is a merchant change like any other: staged by the
-- agent, applied only on an operator's approval. Applying records the request;
-- the submission to a lender is simulated and no money moves.
alter table cartisan.merchant_changes drop constraint if exists merchant_changes_kind_check;
alter table cartisan.merchant_changes add constraint merchant_changes_kind_check check (kind in
  ('inventory_action', 'price_update', 'promotion', 'campaign', 'listing_update',
   'recovery_policy', 'loan_request'));

create table if not exists cartisan.loan_requests (
  id text primary key,
  change_id text not null,
  amount_minor integer not null check (amount_minor > 0),
  tenure_months integer not null check (tenure_months in (3, 6, 9, 12)),
  purpose text not null,
  eligible_limit_minor integer not null,
  status text not null default 'submitted' check (status in ('submitted', 'approved', 'declined')),
  created_at timestamptz not null default now()
);
alter table cartisan.loan_requests enable row level security;
