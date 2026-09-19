# Telegram channel via n8n

n8n is only the transport. Cartisan decides everything: who the user is, what a button does, and when an order counts as paid.

## Setup
1. Create two bots with @BotFather: one for shopping, one for merchants.
2. Apply `supabase/migrations/20260919000000_telegram_channel.sql`.
3. Set the backend env (see `.env.example`): `TELEGRAM_CHANNEL_SECRET` (a long random string), `TELEGRAM_LINK_BASE_URL` (the frontend origin that serves `/telegram/link`), `CARTISAN_MERCHANT_ID` (default `cartisan`).
4. In n8n, add two Telegram credentials named **Cartisan Shopping Bot** and **Cartisan Merchant Bot**.
5. Import `telegram_inbound.json` and `telegram_relay.json`. In the *context* Set nodes, fill in `backend_url`, `secret` (the same value as `TELEGRAM_CHANNEL_SECRET`) and the bot tokens. Activate both workflows.

## Flow
- **Inbound:** a Telegram update fires a trigger, which sends `typing…`, signs the update, and POSTs it to `/channels/telegram/{shopping|merchant}/cartisan`. n8n then makes each Bot API call the backend returns.
- **Relay:** every 15 seconds n8n makes a signed POST to `/channels/telegram/relay` and sends each delivery. The relay covers "account linked", "payment verified" (only after the Razorpay webhook has moved the order to paid), and "change awaiting decision" (with Approve/Reject buttons).
- **Linking:** `/link` (or any action that needs an account) returns a single-use link that expires after 10 minutes and opens `/telegram/link` on the frontend. The user signs in, confirms, and the frontend calls `POST /channels/telegram/link` with their Supabase session.

## Signing
The signature is `X-Cartisan-Signature = hex(HMAC_SHA256(secret, "{timestamp}.{raw body}"))` and the timestamp is sent as `X-Cartisan-Timestamp` (unix seconds). The backend rejects a missing or wrong signature, a timestamp more than 5 minutes off, or an empty secret.

## Commands
`/start`, `/link`, `/new` (start a fresh conversation).
