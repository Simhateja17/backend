-- Which Paytm products a merchant runs. 'pos' (Paytm POS / Smart Retail) supplies the
-- catalogue and stock; 'payments' (QR / Soundbox only) has payment data alone.
alter table cartisan.merchant_operators
  add column if not exists paytm_plan text not null default 'pos'
  check (paytm_plan in ('pos', 'payments'));
