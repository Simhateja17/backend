"""Telegram as a Cartisan channel, with n8n as the transport.

n8n receives Telegram updates, signs them, and posts them here; it sends back to
Telegram whatever this module returns. It holds no state and makes no decisions.
Everything with authority stays on this side of the line:

  * Who is speaking comes from `telegram_links`, written only when a verified
    Supabase session redeems a single-use link token. Nothing n8n or the model sends
    names a principal.
  * Buttons carry an opaque id into `telegram_actions`; what the button does, and in
    which chat, is read back from there, so a forged `callback_data` addresses nothing.
  * Each Telegram `update_id` is acted on once, so a redelivery cannot add a second
    cart line or confirm a checkout twice.
  * Payment is never announced from here. "Paid" is pushed by the relay only after
    the verified payment (Paytm simulator) has moved the order (ADR 0013).
"""

from __future__ import annotations

import hashlib
import hmac
import html
import re
import secrets
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from commerce_common.streaming import AgentEvent
from marketplace_backend.carts import ConflictError
from marketplace_backend.evidence import Correlation
from marketplace_backend.identity import paytm_plan_for
from marketplace_backend.merchant import DecisionRefused
from marketplace_backend.shopping import CheckoutRefused
from marketplace_backend.store import Store

BOT_KINDS = ("shopping", "merchant")
SIGNATURE_HEADER = "X-Cartisan-Signature"
TIMESTAMP_HEADER = "X-Cartisan-Timestamp"
SIGNATURE_MAX_SKEW_SECONDS = 300
LINK_TOKEN_TTL = timedelta(minutes=10)
ROLE_FOR_BOT = {"shopping": "customer", "merchant": "merchant_operator"}
_MESSAGE_LIMIT = 4000  # Telegram's cap is 4096; leave room for the joiner.
_MAX_CARDS = 6


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(moment: datetime) -> str:
    return moment.isoformat()


# ------------------------------------------------------------------ signing


def sign(secret: str, timestamp: str, raw: bytes) -> str:
    """HMAC-SHA256 over `timestamp.body`, hex. Binding the timestamp into the MAC is
    what makes the skew check mean anything."""
    message = timestamp.encode() + b"." + raw
    return hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


def verify(secret: str, timestamp: str, signature: str, raw: bytes,
           *, now: float | None = None) -> bool:
    """With no secret configured the channel is closed, not open."""
    if not secret or not timestamp or not signature:
        return False
    try:
        sent = int(timestamp)
    except ValueError:
        return False
    if abs((now if now is not None else time.time()) - sent) > SIGNATURE_MAX_SKEW_SECONDS:
        return False
    return hmac.compare_digest(sign(secret, timestamp, raw), signature)


# ---------------------------------------------------------------- messages


def _send(chat_id: str, text: str, buttons: list[list[dict]] | None = None,
          *, rich: bool = False) -> dict:
    body: dict[str, Any] = {"chat_id": chat_id, "disable_web_page_preview": True}
    if rich:  # agent prose is Markdown; Telegram renders a small HTML subset
        body["text"], body["parse_mode"] = markdown_to_telegram_html(text), "HTML"
    else:
        body["text"] = text[:_MESSAGE_LIMIT]
    if buttons:
        body["reply_markup"] = {"inline_keyboard": buttons}
    return {"method": "sendMessage", "body": body}


def _answer(callback_query_id: str, text: str | None = None) -> dict:
    body: dict[str, Any] = {"callback_query_id": callback_query_id}
    if text:
        body["text"] = text[:200]
    return {"method": "answerCallbackQuery", "body": body}


def _url_button(text: str, url: str) -> dict:
    return {"text": text, "url": url}


def _chunks(text: str) -> list[str]:
    """Split on paragraph boundaries, so no chunk cuts a bold span or code block in half."""
    chunks, current = [], ""
    for paragraph in text.strip().split("\n\n"):
        while len(paragraph) > _MESSAGE_LIMIT:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(paragraph[:_MESSAGE_LIMIT])
            paragraph = paragraph[_MESSAGE_LIMIT:]
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) > _MESSAGE_LIMIT:
            chunks.append(current)
            candidate = paragraph
        current = candidate
    return [chunk for chunk in chunks + [current] if chunk.strip()]


_CODE_BLOCK = re.compile(r"```[\w-]*\n?(.*?)```", re.DOTALL)
_INLINE = [
    (re.compile(r"`([^`\n]+)`"), r"<code>\1</code>"),
    (re.compile(r"\*\*(.+?)\*\*"), r"<b>\1</b>"),
    (re.compile(r"__(.+?)__"), r"<b>\1</b>"),
    (re.compile(r"(?<![\w*])\*(?!\s)([^*\n]+?)\*(?![\w*])"), r"<i>\1</i>"),
    (re.compile(r"(?<!\w)_(?!\s)([^_\n]+?)_(?!\w)"), r"<i>\1</i>"),
    (re.compile(r"~~(.+?)~~"), r"<s>\1</s>"),
    (re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)"), r'<a href="\2">\1</a>'),
]


def markdown_to_telegram_html(text: str) -> str:
    """The Markdown a model writes, as the HTML subset Telegram's `parse_mode=HTML`
    accepts. Everything is escaped first, so model text can never inject markup."""
    blocks: list[str] = []

    def stash(match: re.Match) -> str:
        blocks.append(f"<pre>{html.escape(match.group(1).rstrip())}</pre>")
        return f"\x00{len(blocks) - 1}\x00"

    text = _CODE_BLOCK.sub(stash, text)
    lines = []
    for line in html.escape(text, quote=False).split("\n"):
        heading = re.match(r"^\s{0,3}#{1,6}\s+(.*)$", line)
        if heading:
            line = f"**{heading.group(1).strip()}**"
        line = re.sub(r"^(\s*)[-*+]\s+", r"\1• ", line)
        if re.fullmatch(r"\s*([-*_])\1{2,}\s*", line):
            line = "──────────"
        for pattern, replacement in _INLINE:
            line = pattern.sub(replacement, line)
        lines.append(line)
    rendered = "\n".join(lines)
    return re.sub(r"\x00(\d+)\x00", lambda m: blocks[int(m.group(1))], rendered)


class TelegramChannel:
    """The host for the Telegram surface. `handle` turns one update into the Bot API
    calls n8n should make; `relay` yields proactive messages exactly once."""

    def __init__(self, store: Store, *, shopping_runtime, merchant_runtime,
                 shopping_service, merchant_service,
                 shopping_sessions: tuple[dict, dict], merchant_sessions: tuple[dict, dict],
                 session_types: dict[str, tuple[type, type]],
                 link_base_url: str, require_second_approver: bool = True) -> None:
        self.store = store
        self.runtimes = {"shopping": shopping_runtime, "merchant": merchant_runtime}
        self.shopping = shopping_service
        self.merchant = merchant_service
        self.sessions = {"shopping": shopping_sessions, "merchant": merchant_sessions}
        self.session_types = session_types
        self.link_base_url = link_base_url.rstrip("/")
        self.require_second_approver = require_second_approver

    # ------------------------------------------------------------  identity

    def linked_principal(self, bot_kind: str, telegram_user_id: str) -> str | None:
        rows = self.store.rows(
            "SELECT principal_id FROM telegram_links WHERE bot_kind=? AND telegram_user_id=?",
            (bot_kind, telegram_user_id))
        return rows[0]["principal_id"] if rows else None

    def issue_link_token(self, bot_kind: str, telegram_user_id: str, chat_id: str) -> str:
        token = secrets.token_urlsafe(24)
        self.store.execute(
            "INSERT INTO telegram_link_tokens (token_hash,bot_kind,telegram_user_id,chat_id,"
            "expires_at) VALUES (?,?,?,?,?)",
            (_hash(token), bot_kind, telegram_user_id, chat_id,
             _iso(_utcnow() + LINK_TOKEN_TTL)))
        return token

    def link_url(self, bot_kind: str, token: str) -> str:
        return f"{self.link_base_url}/telegram/link?bot={bot_kind}&token={token}"

    def redeem_link_token(self, token: str, principal) -> dict:
        """Bind the Telegram account that asked for `token` to the verified principal.

        Single use, short-lived, and role-checked: a shopper's session cannot link
        the merchant bot. A guest cart is carried over when the account has none.
        """
        rows = self.store.rows(
            "SELECT * FROM telegram_link_tokens WHERE token_hash=?", (_hash(token),))
        if not rows:
            raise LinkRefused("That link is not valid. Ask the bot for a new one with /link.")
        row = rows[0]
        if row["redeemed_at"]:
            raise LinkRefused("That link was already used. Ask the bot for a new one with /link.")
        if datetime.fromisoformat(row["expires_at"]) < _utcnow():
            raise LinkRefused("That link expired. Ask the bot for a new one with /link.")
        bot_kind = row["bot_kind"]
        if principal.role != ROLE_FOR_BOT[bot_kind]:
            raise LinkRefused(
                "The merchant bot needs an operator account." if bot_kind == "merchant"
                else "The shopping bot needs a customer account.")
        claimed = self.store.execute(
            "UPDATE telegram_link_tokens SET redeemed_at=?, redeemed_by=? "
            "WHERE token_hash=? AND redeemed_at IS NULL",
            (_iso(_utcnow()), principal.id, row["token_hash"]))
        if getattr(claimed, "rowcount", 1) == 0:  # lost a race with a second redemption
            raise LinkRefused("That link was already used. Ask the bot for a new one with /link.")
        self.store.execute("DELETE FROM telegram_links WHERE bot_kind=? AND telegram_user_id=?",
                           (bot_kind, row["telegram_user_id"]))
        self.store.execute(
            "INSERT INTO telegram_links (bot_kind,telegram_user_id,chat_id,principal_id,role,"
            "linked_at) VALUES (?,?,?,?,?,?)",
            (bot_kind, row["telegram_user_id"], row["chat_id"], principal.id, principal.role,
             _iso(_utcnow())))
        if bot_kind == "shopping":
            self.store.execute(
                "UPDATE customer_carts SET customer_id=? WHERE customer_id=? AND status='active' "
                "AND NOT EXISTS (SELECT 1 FROM customer_carts c2 WHERE c2.customer_id=? "
                "AND c2.status='active')",
                (principal.id, _guest_id(row["telegram_user_id"]), principal.id))
        return {"linked": True, "bot_kind": bot_kind}

    # ------------------------------------------------------------ updates

    def claim_update(self, bot_kind: str, update_id: Any) -> bool:
        """True the first time an update is seen; Telegram and n8n both retry."""
        cursor = self.store.execute(
            "INSERT INTO telegram_updates (bot_kind,update_id) VALUES (?,?) "
            "ON CONFLICT DO NOTHING", (bot_kind, str(update_id)))
        return getattr(cursor, "rowcount", 1) != 0

    async def handle(self, bot_kind: str, update: dict) -> list[dict]:
        if "update_id" not in update or not self.claim_update(bot_kind, update["update_id"]):
            return []
        if update.get("callback_query"):
            return await self._callback(bot_kind, update["callback_query"])
        message = update.get("message") or {}
        text = (message.get("text") or "").strip()
        chat_id = str((message.get("chat") or {}).get("id") or "")
        user_id = str((message.get("from") or {}).get("id") or "")
        if not (text and chat_id and user_id):
            return []
        return await self._message(bot_kind, chat_id, user_id, text)

    async def _message(self, bot_kind: str, chat_id: str, user_id: str, text: str) -> list[dict]:
        command = text.split()[0].split("@")[0].lower() if text.startswith("/") else None
        principal_id = self.linked_principal(bot_kind, user_id)
        if command == "/link":
            return [self._link_prompt(bot_kind, chat_id, user_id)]
        if command == "/new":
            self._bump_conversation(bot_kind, chat_id)
            return [_send(chat_id, "Started a new conversation.")]
        if command == "/start":
            greeting = ("Hi! I'm the Cartisan shopping assistant. Ask me about products, "
                        "compatibility or your cart." if bot_kind == "shopping" else
                        "Hi! I'm the Cartisan merchant assistant. Ask about sales, stock "
                        "or staged changes.")
            messages = [_send(chat_id, greeting)]
            if principal_id is None and bot_kind == "merchant":
                messages.append(self._link_prompt(bot_kind, chat_id, user_id))
            return messages
        if bot_kind == "merchant" and principal_id is None:
            return [self._link_prompt(bot_kind, chat_id, user_id)]
        return await self._turn(bot_kind, chat_id, principal_id or _guest_id(user_id), text)

    def _link_prompt(self, bot_kind: str, chat_id: str, user_id: str) -> dict:
        token = self.issue_link_token(bot_kind, user_id, chat_id)
        who = "operator" if bot_kind == "merchant" else "Cartisan"
        return _send(chat_id, f"Link your {who} account to continue. The link works once "
                     "and expires in 10 minutes.",
                     [[_url_button("Link account", self.link_url(bot_kind, token))]])

    # ------------------------------------------------------------ turns

    def _conversation_seq(self, bot_kind: str, chat_id: str) -> int:
        rows = self.store.rows(
            "SELECT conversation_seq FROM telegram_chats WHERE bot_kind=? AND chat_id=?",
            (bot_kind, chat_id))
        return int(rows[0]["conversation_seq"]) if rows else 0

    def _bump_conversation(self, bot_kind: str, chat_id: str) -> None:
        self.store.execute(
            "INSERT INTO telegram_chats AS c (bot_kind,chat_id,conversation_seq) VALUES (?,?,1) "
            "ON CONFLICT (bot_kind,chat_id) DO UPDATE SET conversation_seq=c.conversation_seq+1",
            (bot_kind, chat_id))

    def conversation_key(self, bot_kind: str, chat_id: str, principal_id: str) -> str:
        # Same `principal:conversation` shape the web surfaces use, so a linked
        # user's Telegram chats also appear in their conversation list.
        return f"{principal_id}:tg-{chat_id}-{self._conversation_seq(bot_kind, chat_id)}"

    async def _turn(self, bot_kind: str, chat_id: str, principal_id: str, text: str) -> list[dict]:
        key = self.conversation_key(bot_kind, chat_id, principal_id)
        transcripts, states = self.sessions[bot_kind]
        context_type, state_type = self.session_types[bot_kind]
        messages = transcripts.setdefault(key, [])
        state = states.setdefault(key, state_type())
        extra = ({"paytm_plan": paytm_plan_for(self.store, principal_id)}
                 if bot_kind == "merchant" else {})
        session = context_type(conversation_id=key, customer_id=principal_id,
                               correlation_id=Correlation().correlation_id,
                               channel="telegram", **extra)
        messages.append({"role": "user", "content": text})
        events: list[AgentEvent] = []
        try:
            async for event in self.runtimes[bot_kind].stream_turn(messages, session, state):
                events.append(event)
        except Exception:  # the turn is already marked failed; the chat gets one line
            events.append(AgentEvent.error("Something went wrong on that turn. Please try again."))
        return self.render(bot_kind, chat_id, events)

    # ------------------------------------------------------------ rendering

    def render(self, bot_kind: str, chat_id: str, events: list[AgentEvent]) -> list[dict]:
        text = "".join(e.data.get("text", "") for e in events if e.type == "text_delta")
        out = [_send(chat_id, chunk, rich=True) for chunk in _chunks(text)]
        for event in events:
            if event.type == "ui":
                out.extend(self._component(bot_kind, chat_id, event.data["component"],
                                           event.data.get("payload") or {}))
            elif event.type == "change_update":
                out.append(self._change(chat_id, event.data.get("change") or {}))
            elif event.type == "error":
                out.append(_send(chat_id, event.data.get("message", "Something went wrong.")))
        return out

    def _component(self, bot_kind: str, chat_id: str, component: str, p: dict) -> list[dict]:
        if component == "products":
            cards = p.get("items") or []
            out = [_send(chat_id, p["title"])] if p.get("title") else []
            for card in cards[:_MAX_CARDS]:
                lines = [card.get("title", ""), f"{card.get('brand') or ''} · {card.get('price', '')}".strip(" ·"),
                         "In stock" if card.get("in_stock") else "Out of stock"]
                if card.get("reason"):
                    lines.append(card["reason"])
                lines.extend(f"✓ {check}" for check in card.get("fit_checks") or [])
                buttons = ([[self._button(bot_kind, chat_id, "🛒 Add to cart", "add_to_cart",
                                          {"variant_id": card["variant_id"]})]]
                           if card.get("in_stock") else None)
                out.append(_send(chat_id, "\n".join(filter(None, lines)), buttons))
            return out
        if component == "comparison":
            parts = [p.get("title") or "Comparison"]
            for entry in p.get("entries") or []:
                star = " ⭐ recommended" if entry.get("variant_id") == p.get("recommended_variant_id") else ""
                parts.append(f"\n{entry.get('title')} — {entry.get('price')}{star}")
                parts.extend(f"  + {pro}" for pro in entry.get("pros") or [])
                parts.extend(f"  − {con}" for con in entry.get("cons") or [])
                if entry.get("best_for"):
                    parts.append(f"  Best for: {entry['best_for']}")
            return [_send(chat_id, "\n".join(parts))]
        if component == "cart":
            lines = p.get("lines") or []
            if not lines:
                return [_send(chat_id, "Your cart is empty.")]
            body = "\n".join(f"{l['quantity']} × {l['title']} — {l.get('amount', '')}" for l in lines)
            return [_send(chat_id, f"{p.get('title') or 'Your cart'}\n{body}\nSubtotal: {p.get('subtotal')}",
                          [[self._button(bot_kind, chat_id, "Checkout", "say",
                                         {"text": "Check out my cart"})]])]
        if component == "checkout":
            body = "\n".join(f"{l['quantity']} × {l['title']} — {l.get('amount', '')}"
                             for l in p.get("lines") or [])
            return [_send(chat_id, f"Checkout preview\n{body}\nTotal: {p.get('total')}",
                          [[self._button(bot_kind, chat_id, f"✅ Confirm & pay {p.get('total')}",
                                         "confirm_checkout", {"stage_id": p["stage_id"]})]])]
        if component == "order_status":
            return [_send(chat_id, f"Order {p.get('order_id')}: {p.get('status')} · "
                          f"{p.get('total')}\n{p.get('summary') or ''}")]
        if component == "guide":
            parts = [p.get("title", "")] + [f"\n{s['heading']}\n{s['body']}" for s in p.get("sections") or []]
            return [_send(chat_id, "\n".join(parts))]
        if component == "suggestions":
            chips = p.get("suggestions") or []
            return [_send(chat_id, "You could also ask:", [
                [self._button(bot_kind, chat_id, chip, "say", {"text": chip})] for chip in chips])]
        if component == "digest":
            parts = [p.get("title", "")] + [f"\n• {i['heading']}\n{i['body']}" for i in p.get("items") or []]
            return [_send(chat_id, "\n".join(parts))]
        if component == "metrics":
            points = p.get("points") or []
            rows = [f"{pt.get('label') or pt.get('key') or pt.get('date') or ''}: {pt.get('value')}"
                    for pt in points[:15]]
            return [_send(chat_id, "\n".join([p.get("title", ""), *rows, f"Total: {p.get('total')}"]))]
        if component == "change_preview":
            return [self._change(chat_id, p)]
        return []

    def _change(self, chat_id: str, change: dict) -> dict:
        """An approval card a shop owner can read at a glance: what, on which product,
        and why — no ids, field names or raw documents."""
        name = self._target_name(change.get("target_type"), change.get("target_id"))
        headline, details = _describe_change(change.get("kind") or "", name,
                                             change.get("before") or {}, change.get("after") or {})
        status = _STATUS_WORDS.get(change.get("status") or "", str(change.get("status") or ""))
        parts = [f"📝 {headline}", *details]
        if change.get("rationale"):
            parts.append(f"\nWhy: {change['rationale']}")
        parts.append(f"\n{status}")
        text = "\n".join(parts)
        buttons = None
        if change.get("status") == "pending" and change.get("change_id"):
            buttons = [[
                self._button("merchant", chat_id, "✅ Approve", "decide",
                             {"change_id": change["change_id"], "decision": "approved"}),
                self._button("merchant", chat_id, "❌ Reject", "decide",
                             {"change_id": change["change_id"], "decision": "rejected"}),
            ]]
        return _send(chat_id, text, buttons)

    def _target_name(self, target_type: str | None, target_id: str | None) -> str:
        if not target_id:
            return "your store"
        for table in ("catalog_variants", "catalog_products"):
            rows = self.store.rows(f"SELECT title FROM {table} WHERE id=?", (target_id,))
            if rows:
                return rows[0]["title"]
        return "this item" if target_type and "variant" in target_type else "your store"

    def _button(self, bot_kind: str, chat_id: str, text: str, action: str, args: dict) -> dict:
        action_id = f"tga_{secrets.token_hex(12)}"
        self.store.execute(
            "INSERT INTO telegram_actions (id,bot_kind,chat_id,action,args) VALUES (?,?,?,?,?)",
            (action_id, bot_kind, chat_id, action, self.store.dump(args)))
        return {"text": text[:64], "callback_data": action_id}

    # ------------------------------------------------------------ callbacks

    async def _callback(self, bot_kind: str, query: dict) -> list[dict]:
        query_id = str(query.get("id") or "")
        chat_id = str(((query.get("message") or {}).get("chat") or {}).get("id") or "")
        user_id = str((query.get("from") or {}).get("id") or "")
        rows = self.store.rows(
            "SELECT * FROM telegram_actions WHERE id=? AND bot_kind=? AND chat_id=?",
            (str(query.get("data") or ""), bot_kind, chat_id))
        if not rows:
            return [_answer(query_id, "That button is no longer valid.")]
        action, args = rows[0]["action"], self.store.load(rows[0]["args"])
        principal_id = self.linked_principal(bot_kind, user_id)
        correlation = Correlation()

        if action == "say":
            if bot_kind == "merchant" and principal_id is None:
                return [_answer(query_id), self._link_prompt(bot_kind, chat_id, user_id)]
            return [_answer(query_id)] + await self._turn(
                bot_kind, chat_id, principal_id or _guest_id(user_id), args["text"])

        if action == "add_to_cart":
            customer = principal_id or _guest_id(user_id)
            try:
                cart = await self.shopping.add(customer, args["variant_id"], 1,
                                               idempotency_key=f"tg:{query_id}",
                                               correlation=correlation)
            except (ConflictError, ValueError) as exc:
                return [_answer(query_id, str(exc))]
            count = sum(line["quantity"] for line in cart.get("lines", []))
            return [_answer(query_id, f"Added. {count} item(s) in your cart.")]

        if action == "confirm_checkout":
            if principal_id is None:
                return [_answer(query_id, "Link your account to pay."),
                        self._link_prompt(bot_kind, chat_id, user_id)]
            try:
                result = await self.shopping.confirm(
                    principal_id, args["stage_id"], idempotency_key=f"tg:{rows[0]['id']}",
                    correlation=correlation)
            except CheckoutRefused as exc:
                return [_answer(query_id), _send(chat_id, f"{exc} Ask me to check out again.")]
            payment, order = result.get("payment") or {}, result.get("order") or {}
            if not payment.get("pay_url"):
                return [_answer(query_id), _send(
                    chat_id, "Your order is placed and stock is held, but the payment link "
                    "isn't ready yet. I'll send it as soon as it is.")]
            return [_answer(query_id), _send(
                chat_id, f"Order placed. Pay securely with Paytm — I'll confirm here once "
                "the payment is verified.",
                [[_url_button("Pay now", payment["pay_url"])]])]

        if action == "decide":
            if principal_id is None:
                return [_answer(query_id), self._link_prompt(bot_kind, chat_id, user_id)]
            if self.require_second_approver:
                staged = self.store.rows("SELECT operator_id FROM merchant_changes WHERE id=?",
                                         (args["change_id"],))
                if staged and staged[0]["operator_id"] == principal_id:
                    return [_answer(query_id, "A different operator must decide this change.")]
            try:
                decided = self.merchant.decide(operator_id=principal_id,
                                               change_id=args["change_id"],
                                               decision=args["decision"],
                                               note="Decided from Telegram")
            except (DecisionRefused, LookupError) as exc:
                return [_answer(query_id, "Refused"), _send(chat_id, str(exc))]
            return [_answer(query_id, decided.get("status")),
                    _send(chat_id, _STATUS_WORDS.get(decided.get("status") or "", "Done."))]

        return [_answer(query_id, "That button is no longer valid.")]

    # ------------------------------------------------------------ relay

    def relay(self) -> list[dict]:
        """Proactive messages not yet handed to n8n, each claimed exactly once.

        At-most-once by design: a message lost after being claimed is a missed
        notification, whereas a message sent twice could read as a second payment.
        """
        out: list[dict] = []
        for link in self.store.rows("SELECT * FROM telegram_links"):
            if self._claim("linked", link["principal_id"], link["chat_id"]):
                out.append({"bot_kind": link["bot_kind"], **_send(
                    link["chat_id"], "✅ Your account is linked. You're all set.")})

        for row in self.store.rows(
                "SELECT o.id, o.total_minor, l.chat_id FROM commerce_orders o "
                "JOIN telegram_links l ON l.principal_id=o.customer_id AND l.bot_kind='shopping' "
                "WHERE o.status='paid' AND o.paid_at >= l.linked_at"):
            if self._claim("order_paid", row["id"], row["chat_id"]):
                out.append({"bot_kind": "shopping", **_send(
                    row["chat_id"], f"✅ Payment verified. Order {row['id']} is paid "
                    f"(₹{row['total_minor'] / 100:,.2f}). Thank you!")})

        pending = self.store.rows(
            "SELECT * FROM merchant_changes WHERE status='pending' ORDER BY created_at")
        operators = self.store.rows("SELECT * FROM telegram_links WHERE bot_kind='merchant'")
        for change in pending:
            for operator in operators:
                if self._claim("change_pending", change["id"], operator["chat_id"]):
                    record = {**change, "change_id": change["id"],
                              "before": self.store.load(change["before_doc"]),
                              "after": self.store.load(change["after_doc"])}
                    out.append({"bot_kind": "merchant",
                                **self._change(operator["chat_id"], record)})
        return out

    def _claim(self, kind: str, ref_id: str, chat_id: str) -> bool:
        cursor = self.store.execute(
            "INSERT INTO telegram_notifications (kind,ref_id,chat_id) VALUES (?,?,?) "
            "ON CONFLICT DO NOTHING", (kind, ref_id, chat_id))
        return getattr(cursor, "rowcount", 1) != 0


_STATUS_WORDS = {
    "pending": "⏳ Waiting for your approval",
    "approved": "✅ Approved",
    "applied": "✅ Approved and done",
    "rejected": "❌ Rejected",
    "failed": "⚠️ Approved, but it couldn't be applied — the numbers changed since it was proposed",
    "superseded": "Replaced by a newer proposal",
}


def _rupees(minor: Any) -> str:
    return f"₹{int(minor) / 100:,.0f}" if isinstance(minor, int) else "—"


def _describe_change(kind: str, name: str, before: dict, after: dict) -> tuple[str, list[str]]:
    if kind == "inventory_action":
        units = after.get("units")
        if isinstance(units, int) and units < 0:
            headline = f"Remove {abs(units)} units of {name}"
        else:
            headline = f"Restock {name} with {units} units"
        details = []
        if isinstance(before.get("on_hand"), int):
            details.append(f"In stock now: {before['on_hand']}")
            if isinstance(units, int):
                details.append(f"After this: {before['on_hand'] + units}")
        return headline, details
    if kind == "price_update":
        return (f"Change the price of {name}",
                [f"From {_rupees(before.get('amount_minor'))} to {_rupees(after.get('amount_minor'))}"])
    if kind == "promotion":
        value = after.get("discount_value")
        off = f"{value}% off" if after.get("discount_kind") == "percentage" else f"{_rupees(value)} off"
        details = []
        if after.get("min_subtotal_minor"):
            details.append(f"On orders above {_rupees(after['min_subtotal_minor'])}")
        return f"Run a promotion: {off} on {name}", details
    if kind == "campaign":
        return f"Start a campaign for {name}", [f"Budget: {_rupees(after.get('budget_minor'))}"]
    if kind == "listing_update":
        details = []
        if after.get("title"):
            details.append(f"New title: {after['title']}")
        if after.get("description"):
            details.append("New description written")
        if after.get("status"):
            details.append(f"Listing will be {str(after['status']).replace('_', ' ')}")
        return f"Update the listing for {name}", details
    if kind == "loan_request":
        return (f"Apply for a business loan of {_rupees(after.get('amount_minor'))}",
                [f"Repay over {after.get('tenure_months')} months"])
    if kind == "recovery_policy":
        return ("Win back abandoned carts automatically",
                [f"Offer {after.get('discount_percentage')}% off (up to "
                 f"{_rupees(after.get('max_discount_minor'))})",
                 f"Monthly budget: {_rupees(after.get('monthly_budget_minor'))}"])
    return f"Proposed change: {kind.replace('_', ' ')} for {name}", []


class LinkRefused(Exception):
    """A link token that cannot bind: unknown, used, expired, or the wrong role."""


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _guest_id(telegram_user_id: str) -> str:
    # A guest can browse and fill a cart; confirmation needs a linked account.
    return f"tg_guest_{telegram_user_id}"
