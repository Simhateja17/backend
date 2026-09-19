-- Telegram channel via n8n. Generated from marketplace_backend/core_schema.py (TELEGRAM_DDL).

-- ================================================== telegram channel

-- Which conversation a chat is on. `/new` bumps the sequence; the transcript
-- itself lives in `cartisan.conversations`/`cartisan.turns` like every other surface.
create table if not exists cartisan.telegram_chats (
  bot_kind text not null check (bot_kind in ('shopping', 'merchant')),
  chat_id text not null,
  conversation_seq integer not null default 0,
  primary key (bot_kind, chat_id)
);

-- A Telegram account bound to a verified Cartisan principal. Written only by the
-- link-redemption path, which reads the principal from a Supabase session.
create table if not exists cartisan.telegram_links (
  bot_kind text not null check (bot_kind in ('shopping', 'merchant')),
  telegram_user_id text not null,
  chat_id text not null,
  principal_id text not null,
  role text not null,
  linked_at timestamptz not null default now(),
  primary key (bot_kind, telegram_user_id)
);

-- Single-use, short-lived link tokens. Only the hash is stored.
create table if not exists cartisan.telegram_link_tokens (
  token_hash text primary key,
  bot_kind text not null,
  telegram_user_id text not null,
  chat_id text not null,
  expires_at text not null,
  redeemed_at text,
  redeemed_by text
);

-- Telegram redelivers; each update is acted on once.
create table if not exists cartisan.telegram_updates (
  bot_kind text not null,
  update_id text not null,
  received_at timestamptz not null default now(),
  primary key (bot_kind, update_id)
);

-- Inline-button references. `callback_data` carries only this id; what the button
-- does, and for whom, is read back from here.
create table if not exists cartisan.telegram_actions (
  id text primary key,
  bot_kind text not null,
  chat_id text not null,
  action text not null,
  args text not null,
  created_at timestamptz not null default now()
);

-- Proactive messages already handed to the transport, so each goes out once.
create table if not exists cartisan.telegram_notifications (
  kind text not null,
  ref_id text not null,
  chat_id text not null,
  created_at timestamptz not null default now(),
  primary key (kind, ref_id, chat_id)
);

-- Backend-only tables: no client role may read link tokens or bindings.
alter table cartisan.telegram_chats enable row level security;
alter table cartisan.telegram_links enable row level security;
alter table cartisan.telegram_link_tokens enable row level security;
alter table cartisan.telegram_updates enable row level security;
alter table cartisan.telegram_actions enable row level security;
alter table cartisan.telegram_notifications enable row level security;
