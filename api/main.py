from __future__ import annotations

import hmac, json, os
from uuid import uuid4
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from fastapi import BackgroundTasks, Cookie, Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from marketplace_backend.carts import ConflictError
from marketplace_backend.checkout import CheckoutRepository
from marketplace_backend.evidence import (
    ORIGINS,
    Actor,
    CommerceEventLog,
    Correlation,
    EvidenceLedger,
    Inbox,
    Outbox,
)
from marketplace_backend.health import HealthMetrics
from marketplace_backend.observability import EvidenceView
from marketplace_backend.identity import AuthenticationError, IdentityService, Principal
from marketplace_backend.inventory import InventoryRepository
from marketplace_backend.merchant import DecisionRefused, MerchantService
from marketplace_backend.merchant_changes import MerchantChangeRepository
from marketplace_backend.metrics import MetricsRepository
from marketplace_backend.payments import (
    PaymentLinkDispatcher,
    WebhookProcessor,
)
from marketplace_backend.recovery import (
    RecoveryRefused,
    RecoveryService,
    order_recovery_actions,
)
from marketplace_backend.shopping import CheckoutRefused, ShoppingService
from marketplace_backend.sim_gateway import (
    SimulatedCheckout,
    SimulatedCheckoutError,
    SimulatedPaytmGateway,
)
from cartisan_agent.outcomes import Unavailable
from cartisan_agent.types import PageContext
from marketplace_backend.store import Store
from marketplace_backend.cognee_client import CogneeClient
from marketplace_backend.customer_memory import (
    SYNC_TOPIC,
    BehaviorLog,
    EventRefused,
    MemoryBriefs,
    MemorySubjects,
    MemorySyncWorker,
    TypedFactStore,
    expire_guests,
    forget_locally,
    merge_guest,
    schedule_sync,
)
from marketplace_backend.timeutil import now as _now
from marketplace_backend.personalization import Personalizer
from marketplace_backend.cart_recovery import (
    MarketingConsent,
    RecoveryEmailWorker,
    RecoveryOffers,
    ResendSender,
    active_policy,
)
from marketplace_backend.merchant_memory import (
    REASON_CODES,
    MerchantLessonStore,
    record_reason,
    refresh_lessons,
)
from marketplace_backend.customer_memory import MERCHANT_SUBJECT
from marketplace_backend.merchant_changes import POLICY_BOUNDS, PolicyViolation
from cartisan_agent.merchant_executor import build_merchant_memory
from cartisan_agent.executor import build_memory

from cartisan_agent import (
    CartisanAgentConfig,
    CartisanMerchantRuntime,
    CartisanShoppingRuntime,
    CommerceServices,
    CoreCommercePort,
    CoreMerchantPort,
    MerchantAgentConfig,
    MerchantServices,
    MerchantSessionContext,
    MerchantSessionState,
    PresentationLedger,
    SessionContext,
    SessionState,
    TurnStore,
)
from commerce_common.streaming import AgentEvent, to_sse

from .lineage import CORRELATION_HEADER, DEMO_RUN_HEADER, request_correlation
from . import telegram as tg

db = Store(
    path=os.getenv("CARTISAN_DB_PATH"),
    database_url=os.getenv("SUPABASE_DATABASE_URL"),
)
identity = IdentityService(db)

# The Claude runtime (Phase 4). The shopping conversation runs on the Messages API loop
# in `cartisan_agent`; `marketplace_backend.routing` still decides checkout precedence,
# but it now steers that loop instead of standing in for it.
agent_config = CartisanAgentConfig()
ledger = EvidenceLedger(db)
inventory = InventoryRepository(db)
outbox, inbox = Outbox(db), Inbox(db)
checkout_repo = CheckoutRepository(db, inventory, ledger, outbox, CommerceEventLog(db))
core_port = CoreCommercePort(db, checkout=checkout_repo, config=agent_config)
commerce = CommerceServices(
    port=core_port,
    presentations=PresentationLedger(db, agent_config),
)

# Phase 5. The browser and the agent share `core_port`, so there is one cart, one
# price and one stock figure behind both. The payment half is host-only: the
# dispatcher asks a payment provider for a link, and the processor is the single
# path from a verified provider event to a paid order (ADR 0005, ADR 0011, ADR 0013).
#
# The provider is the simulated Paytm gateway: links point at the hosted checkout
# page (`/pay?link=<link id>`), and that page's answer goes through `webhooks` like any
# provider event would. No real money moves.
dispatcher = PaymentLinkDispatcher(db, checkout_repo, outbox, SimulatedPaytmGateway(), ledger)
webhooks = WebhookProcessor(db, checkout_repo, inbox, ledger)
shopping = ShoppingService(db, core_port, checkout_repo, dispatcher, ledger)
simulated_checkout = SimulatedCheckout(db, webhooks)

# Phase 7. Three readers and one set of controls, all on records that already
# existed and had no surface: the evidence ledger, the runtime's own counters, and
# the two stuck states the payment path can reach (ADR 0023, ADR 0030, ADR 0032).
evidence_view = EvidenceView(db)
# Not `health`: the /health route function below binds that name at import time
# and would shadow this, which is a 500 no test catches and one live call does.
health_metrics = HealthMetrics(db)
recovery = RecoveryService(db, checkout_repo, ledger)
turn_store = TurnStore(db, ledger)

# Memory (Cognee). Postgres holds the typed facts, the behaviour log and the brief;
# Cognee Cloud is fed from them through the outbox and never sits on the turn's
# critical path. With COGNEE_ENABLED off, memory still works locally.
cognee = CogneeClient.from_env()
memory_subjects = MemorySubjects(db)
memory_facts = TypedFactStore(db, memory_subjects, outbox)
memory_briefs = MemoryBriefs(db, memory_facts, price_of=core_port.current_price)
behavior = BehaviorLog(db, memory_subjects, outbox)
personalizer = Personalizer(db, core_port)
memory_sync = MemorySyncWorker(db, memory_subjects, memory_facts, memory_briefs, outbox,
                               ledger, cognee)
GUEST_COOKIE = "cartisan_anon_id"

shopping_agent = CartisanShoppingRuntime(
    services=commerce, store=db, config=agent_config, turns=turn_store,
    memory=build_memory(agent_config, memory_facts), brief_reader=memory_briefs.get,
)

# Phase 6. The merchant agent runs the same loop over the same commerce core, and
# stops one step earlier: its writes create `pending` rows in `merchant_changes` and
# nothing else. `merchant_service` is the other side of that line — operator-only,
# in no tool list, and the only thing that turns an approval into a write (ADR 0016).
merchant_config = MerchantAgentConfig()
merchant_changes = MerchantChangeRepository(db, ledger)
merchant_port = CoreMerchantPort(
    db, changes=merchant_changes, metrics=MetricsRepository(db), config=merchant_config
)
merchant_service = MerchantService(db, merchant_port, merchant_changes, ledger)
merchant_lessons = MerchantLessonStore(db, memory_subjects, outbox)
merchant_agent = CartisanMerchantRuntime(
    services=MerchantServices(port=merchant_port), store=db, config=merchant_config,
    turns=turn_store, memory=build_merchant_memory(merchant_config, merchant_lessons),
)

# Cart recovery (Q3, Q4): the approved policy, a deterministic scan, a coupon applied
# at staging from stored terms, and email only with consent.
recovery_offers = RecoveryOffers(db, outbox, ledger, price_of=core_port.current_price,
                                 brief_of=memory_briefs.get)
core_port.stage_discount = recovery_offers.terms_for_stage
marketing_consent = MarketingConsent(db)
recovery_email = RecoveryEmailWorker(
    db, outbox, recovery_offers, marketing_consent, ResendSender.from_env(),
    site_url=os.getenv("CARTISAN_SITE_URL", "http://localhost:3000"),
    api_url=os.getenv("CARTISAN_API_URL", "http://127.0.0.1:8000"))

# The model's message array lives in this process, and deliberately stays there: it
# holds tool_use/tool_result pairs that only the running turn can complete, and a
# half-written pair is exactly what makes the next request unanswerable.
#
# What a person needs back after a reload or a restart is not that array — it is
# what they asked and what they were told, and both are durable on `turns`. That is
# what `/chat/*/resume` returns, so a judge who restarts the backend mid-demo
# repaints the conversation instead of losing it.
_transcripts: dict[str, list[dict]] = {}
_states: dict[str, SessionState] = {}
_portal_transcripts: dict[str, list[dict]] = {}
_portal_states: dict[str, MerchantSessionState] = {}

app = FastAPI(title="Cartisan API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS","http://localhost:3000").split(","),
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Identity is never a request field. `conversation_id` groups a chat thread; it
# carries no authority, and the cart it reads is the authenticated customer's.
class ChatRequest(BaseModel):
    conversation_id: str = Field(default="default",min_length=1,max_length=100)
    message: str = Field(min_length=1,max_length=2000)
    variant_id: str | None = Field(default=None, min_length=1, max_length=100)

# Carts are variant-keyed. A variant is the thing that has a price, a stock level
# and an order line, so it is the only id the cart, the stage and the order share.
class CartRequest(BaseModel):
    variant_id: str
    quantity: int = Field(default=1,ge=0,le=10)
    reasoning: str = "Customer requested this cart change"
    expected_version: int | None = None
    idempotency_key: str | None = Field(default=None,max_length=200)

class StageRequest(BaseModel):
    fulfillment_option: str = Field(default="standard",max_length=40)
    note: str | None = Field(default=None,max_length=280)

class ConfirmRequest(BaseModel):
    stage_id: str
    idempotency_key: str | None = Field(default=None,max_length=200)

# A decision names only the change and the verdict. Who decided comes from the
# verified operator principal, and what is being decided comes from the stored row —
# neither is a request field, because both are authority (ADR 0010, ADR 0016).
class DecisionRequest(BaseModel):
    decision: str = Field(pattern="^(approved|rejected)$")
    note: str | None = Field(default=None, max_length=400)
    # Why, in a word merchant memory can count. Optional; never authority.
    reason_code: str | None = Field(default=None, pattern="^(" + "|".join(REASON_CODES) + ")$")

# A recovery action names the thing and the human's reason. Who acted comes from the
# operations token, not from the body.
class AcknowledgeRequest(BaseModel):
    note: str = Field(min_length=1, max_length=400)

class CancelOrderRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=400)


def require_customer(authorization: str | None = Header(default=None)) -> Principal:
    """Resolve the request's principal from a verified Supabase session."""
    try:
        principal = identity.principal(authorization)
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    if principal.role != "customer":
        raise HTTPException(status_code=403, detail="This action requires a customer account")
    return principal

def require_operator(authorization: str | None = Header(default=None)) -> Principal:
    """The merchant surfaces act on the whole store, so they need an operator, not a
    signed-in shopper. The role comes from Supabase app metadata, never the client."""
    try:
        principal = identity.principal(authorization)
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    if principal.role != "merchant_operator":
        raise HTTPException(status_code=403, detail="This action requires an operator account")
    return principal

def require_operations_token(x_cartisan_ops_token: str = Header(default="")) -> None:
    """The maintenance and recovery endpoints change commerce state, so they are not
    open. With no token configured they are closed rather than public, and none of
    them is reachable from a model (ADR 0005)."""
    expected = os.getenv("CARTISAN_OPS_TOKEN", "")
    if not expected or not hmac.compare_digest(expected, x_cartisan_ops_token):
        raise HTTPException(status_code=401, detail="Operations token required")


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data,ensure_ascii=False)}\n\n"


def _lineage_headers(correlation: Correlation) -> dict[str, str]:
    """A streamed response is constructed by the handler, so the header the
    dependency set on the injected `Response` never reaches the client. The client
    needs it to continue the journey on its next call, so it is set here too."""
    headers = {CORRELATION_HEADER: correlation.correlation_id}
    if correlation.demo_run_id:
        headers[DEMO_RUN_HEADER] = correlation.demo_run_id
    return headers


@app.get("/health")
def health(): return {"status":"ok"}

@app.get("/health/database")
def database_health():
    db.rows("SELECT 1 AS ready")
    return {"status":"ok", "database":db.backend}

def _conversation_key(principal: Principal, conversation_id: str) -> str:
    # Keyed by principal as well as conversation: a conversation id arriving from a
    # client carries no authority, and must never address another customer's transcript.
    return f"{principal.id}:{conversation_id}"


def _public_conversation_id(principal: Principal, conversation_key: str) -> str:
    """Return the client-side id without exposing the principal namespace."""
    prefix = f"{principal.id}:"
    return conversation_key.removeprefix(prefix)


def _conversation_summaries(principal: Principal, surface: str, limit: int) -> list[dict]:
    rows = db.rows(
        "SELECT c.id AS conversation_key, c.created_at, COUNT(t.id) AS turn_count, "
        "MAX(COALESCE(t.completed_at,t.started_at,c.created_at)) AS updated_at, "
        "(SELECT t2.user_message FROM turns t2 "
        "WHERE t2.conversation_id=c.id AND t2.user_message IS NOT NULL "
        "ORDER BY t2.sequence ASC LIMIT 1) AS title "
        "FROM conversations c LEFT JOIN turns t ON t.conversation_id=c.id "
        "WHERE c.principal_id=? AND c.surface=? "
        "GROUP BY c.id,c.created_at "
        "ORDER BY updated_at DESC,c.created_at DESC LIMIT ?",
        (principal.id, surface, max(1, min(limit, 100))),
    )
    return [
        {
            "conversation_id": _public_conversation_id(principal, row["conversation_key"]),
            "title": row["title"],
            "turn_count": int(row["turn_count"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        for row in rows
    ]


@app.post("/chat/storefront")
async def storefront_chat(body: ChatRequest, principal: Principal = Depends(require_customer),
                          correlation: Correlation = Depends(request_correlation)):
    """One agent turn, streamed. The events are `commerce_common.streaming.AgentEvent`
    types; a client renders the ones it knows and ignores the rest.

    The turn adopts the request's lineage, so the browser action that started it, the
    tools it calls and anything it stages are one journey rather than three (ADR 0032).
    """
    key = _conversation_key(principal, body.conversation_id)
    messages = _transcripts.setdefault(key, [])
    state = _states.setdefault(key, SessionState())
    session = SessionContext(conversation_id=key, customer_id=principal.id,
                             correlation_id=correlation.correlation_id,
                             demo_run_id=correlation.demo_run_id)
    if body.variant_id:
        details = await catalog_variant(body.variant_id)
        session.page = PageContext(page_type="product", variant_id=body.variant_id,
                                   extra={"product": details})
    messages.append({"role": "user", "content": body.message})

    async def stream() -> AsyncIterator[str]:
        completed = False
        try:
            async for event in shopping_agent.stream_turn(messages, session, state):
                yield to_sse(event)
            completed = True
        except Exception:  # the turn is already marked failed; the client gets one line
            yield to_sse(
                AgentEvent.error("Something went wrong on that turn. Please try again.")
            )
        yield sse("done", {"ok": True})
        if completed:
            # After `done`, so the customer never waits on it. Memory never fails a turn.
            await shopping_agent.update_memory(messages, session)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers=_lineage_headers(correlation))


@app.get("/chat/storefront/conversations")
def storefront_conversations(limit: int = 50,
                             principal: Principal = Depends(require_customer)):
    """List this customer's durable shopping conversations, newest first.

    Conversation ids are namespaced by the verified principal in storage. The
    response removes that internal namespace before returning them, while the
    principal filter ensures the client cannot discover another customer's chats.
    """
    return _conversation_summaries(principal, "shopping", limit)


@app.get("/chat/storefront/resume")
def storefront_resume(conversation_id: str, principal: Principal = Depends(require_customer)):
    """What a reconnecting client should show: the turn still running, or the reply it
    missed while it was away — plus the conversation so far (ADR 0029).

    `history` comes from the `turns` table, so it survives a restart of this process.
    It is what a person said and what they were told, not the model's message array:
    that array holds tool_use blocks only the running turn can pair with results, and
    it stays where it is being written.
    """
    key = _conversation_key(principal, conversation_id)
    resumed = turn_store.resume(key) or {"state": "idle", "turn_id": None, "agent_message": None}
    return {**resumed, "history": turn_store.history(key)}


@app.get("/chat/portal/conversations")
def portal_conversations(limit: int = 50,
                         principal: Principal = Depends(require_operator)):
    """List this operator's durable merchant conversations, newest first."""
    return _conversation_summaries(principal, "merchant", limit)


@app.post("/chat/portal")
async def portal_chat(body: ChatRequest, principal: Principal = Depends(require_operator),
                      correlation: Correlation = Depends(request_correlation)):
    """One merchant agent turn, streamed.

    The same `AgentEvent` stream the storefront speaks, so the portal renders tool
    calls, components and errors the same way. A turn may stage a change, which
    arrives as a `change_update` event and appears in the approval queue; it cannot
    approve or apply one, and there is no tool on this surface that could.
    """
    key = _conversation_key(principal, body.conversation_id)
    messages = _portal_transcripts.setdefault(key, [])
    state = _portal_states.setdefault(key, MerchantSessionState())
    session = MerchantSessionContext(conversation_id=key, customer_id=principal.id,
                                     correlation_id=correlation.correlation_id,
                                     demo_run_id=correlation.demo_run_id)
    messages.append({"role": "user", "content": body.message})

    async def stream() -> AsyncIterator[str]:
        try:
            async for event in merchant_agent.stream_turn(messages, session, state):
                yield to_sse(event)
        except Exception:  # the turn is already marked failed; the client gets one line
            yield to_sse(
                AgentEvent.error("Something went wrong on that turn. Please try again.")
            )
        yield sse("done", {"ok": True})

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers=_lineage_headers(correlation))


@app.get("/chat/portal/resume")
def portal_resume(conversation_id: str, principal: Principal = Depends(require_operator)):
    """What a reconnecting portal should show: the turn still running, or the reply it
    missed while it was away, plus the durable conversation so far (ADR 0029)."""
    key = _conversation_key(principal, conversation_id)
    resumed = turn_store.resume(key) or {"state": "idle", "turn_id": None, "agent_message": None}
    return {**resumed, "history": turn_store.history(key)}


@app.get("/catalog")
def catalog():
    """The normalized catalogue. Every buyable id here is a variant id."""
    return shopping.catalog()


@app.get("/catalog/variants/{variant_id}")
async def catalog_variant(variant_id: str):
    active = shopping.store.rows(
        "SELECT v.id FROM catalog_variants v JOIN catalog_products p ON p.id = v.product_id "
        "WHERE v.id = ? AND v.status = 'active' AND p.status = 'active'", (variant_id,))
    if not active:
        raise HTTPException(404, "This product is no longer available.")
    details = await shopping.port.get_product_details(shopping.session("catalog"), variant_id)
    if details is None:
        raise HTTPException(404, "Product not found.")
    return details.model_dump()

@app.get("/cart")
async def cart(principal: Principal = Depends(require_customer)):
    return await shopping.cart(principal.id)

@app.post("/cart/items")
async def add_cart(body: CartRequest, principal: Principal = Depends(require_customer),
                   correlation: Correlation = Depends(request_correlation)):
    return await _cart_write(shopping.add(
        principal.id, body.variant_id, body.quantity,
        expected_version=body.expected_version, idempotency_key=body.idempotency_key,
        correlation=correlation))

@app.patch("/cart/items")
async def update_cart(body: CartRequest, principal: Principal = Depends(require_customer),
                      correlation: Correlation = Depends(request_correlation)):
    return await _cart_write(shopping.update(
        principal.id, body.variant_id, body.quantity,
        expected_version=body.expected_version, idempotency_key=body.idempotency_key,
        correlation=correlation))

@app.delete("/cart/items/{variant_id}")
async def remove_cart(variant_id: str, principal: Principal = Depends(require_customer),
                      correlation: Correlation = Depends(request_correlation)):
    return await _cart_write(shopping.remove(principal.id, variant_id, correlation=correlation))

async def _cart_write(coro):
    """A stale version is a 409 the client can recover from by re-reading; an item
    that cannot be sold is a 400. Neither is a 500, because neither is a surprise."""
    try:
        return await coro
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

# -- checkout ----------------------------------------------------------------
# Three separate calls, because they have three different authorities behind them.
# Staging holds nothing; confirmation is the customer's act and the only thing that
# reserves stock; the payment link is requested by the host, never by the model.

@app.post("/checkout/stage")
async def stage_checkout(body: StageRequest, principal: Principal = Depends(require_customer),
                         correlation: Correlation = Depends(request_correlation)):
    try:
        return await shopping.stage(principal.id, fulfillment_option=body.fulfillment_option,
                                    note=body.note, correlation=correlation)
    except CheckoutRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

@app.post("/checkout/confirm")
async def confirm_checkout(body: ConfirmRequest, principal: Principal = Depends(require_customer),
                           correlation: Correlation = Depends(request_correlation)):
    try:
        return await shopping.confirm(principal.id, body.stage_id,
                                      idempotency_key=body.idempotency_key,
                                      correlation=correlation)
    except CheckoutRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

@app.get("/orders")
def order_list(principal: Principal = Depends(require_customer)):
    return shopping.orders(principal.id)

@app.get("/orders/{order_id}")
def order_status(order_id: str, principal: Principal = Depends(require_customer)):
    try:
        return shopping.order(principal.id, order_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

@app.post("/orders/{order_id}/payment")
async def retry_payment(order_id: str, principal: Principal = Depends(require_customer)):
    """Try again on the same internal order. A retry is a new attempt, never a new
    order, so the stock the customer already holds is not reserved twice (ADR 0030)."""
    try:
        shopping.order(principal.id, order_id)  # ownership first, before any effect
        return await shopping.open_payment(principal.id, order_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except CheckoutRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

@app.post("/orders/{order_id}/redirect")
def payment_redirect(order_id: str, principal: Principal = Depends(require_customer)):
    """The customer came back from Razorpay. That is not proof of payment: it moves
    the order to `payment_verification_pending` and waits for a verified event."""
    try:
        return shopping.redirect_returned(principal.id, order_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

@app.get("/me")
def me(principal: Principal = Depends(require_customer)):
    return {"id":principal.id,"email":principal.email,"role":principal.role,
            "display_name":principal.display_name}

# -- memory (Cognee) -----------------------------------------------------------
# Whose memory a request touches is decided here, from a verified Supabase session
# or a guest cookie this server signed — never from a body field (Q6, Q9).

class MemoryEventRequest(BaseModel):
    event_type: str = Field(pattern="^(product_view|add_to_cart|remove_from_cart|"
                                    "rejected_recommendation|suggestion_click)$")
    variant_id: str = Field(min_length=1, max_length=100)
    dwell_ms: int | None = Field(default=None, ge=0, le=3_600_000)

class FeedbackRequest(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=100)
    rating: str = Field(pattern="^(up|down)$")
    reason: str | None = Field(default=None, pattern="^(not_relevant|wrong_info|too_pushy|helpful|other)$")
    turn_id: str | None = Field(default=None, max_length=100)
    note: str | None = Field(default=None, max_length=280)


def memory_subject(response: Response, authorization: str | None = Header(default=None),
                   cartisan_anon_id: str | None = Cookie(default=None)) -> str:
    """A signed-in customer's own subject, otherwise the guest cookie's (issued here
    on first use). An operator or an invalid token is refused, not demoted."""
    if authorization:
        return require_customer(authorization).id
    anon_id = memory_subjects.verify_cookie(cartisan_anon_id)
    if anon_id is None:
        cookie = memory_subjects.new_guest_cookie()
        anon_id = memory_subjects.verify_cookie(cookie)
        response.set_cookie(GUEST_COOKIE, cookie, max_age=90 * 24 * 3600, httponly=True,
                            samesite="lax", secure=os.getenv("CARTISAN_SECURE_COOKIES") == "1")
    return memory_subjects.guest_subject(anon_id)


@app.post("/memory/events")
async def memory_event(body: MemoryEventRequest, background: BackgroundTasks,
                       subject: str = Depends(memory_subject),
                       correlation: Correlation = Depends(request_correlation)):
    """A browsing signal. `purchase` is not accepted from the browser: the paid-order
    path records it, because only a verified webhook knows an order was paid."""
    try:
        result = behavior.record(subject, body.event_type, body.variant_id,
                                 dwell_ms=body.dwell_ms, correlation_id=correlation.correlation_id)
    except EventRefused as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if result["counted"]:
        background.add_task(memory_sync.drain, 5)
    return result


@app.get("/memory/me")
def memory_panel(subject: str = Depends(memory_subject)):
    """What Cartisan remembers about the caller, and why (Q8)."""
    brief = memory_briefs.refresh(subject)
    return {
        "subject_kind": brief["subject_kind"],
        "facts": brief["facts"],
        "interests": {key: brief[key] for key in (
            "top_categories", "liked_brands", "avoided_brands", "price_band_minor")},
        "explored": brief["explored"],
        "notes": brief["cognee_notes"],
        "notice": "Cartisan remembers what you browse and tell the assistant so it can "
                  "personalise suggestions. Memory is processed by Cognee on our behalf. "
                  "You can delete any item, or everything, at any time.",
    }


@app.delete("/memory/me/facts/{fact_id}")
def memory_delete_fact(fact_id: str, subject: str = Depends(memory_subject)):
    if not memory_facts.delete_by_id(subject, fact_id):
        raise HTTPException(status_code=404, detail="Nothing to delete")
    memory_briefs.refresh(subject)
    return {"deleted": True}


@app.delete("/memory/me")
def memory_forget_everything(subject: str = Depends(memory_subject)):
    forget_locally(db, subject)
    if memory_subjects.get(subject):
        outbox.enqueue(topic=SYNC_TOPIC, payload={"subject_id": subject, "forget": True})
    return {"forgotten": True}


@app.post("/memory/merge")
def memory_merge(principal: Principal = Depends(require_customer),
                 cartisan_anon_id: str | None = Cookie(default=None)):
    """Call once after sign-in: the guest's signals move to the account, one way."""
    anon_id = memory_subjects.verify_cookie(cartisan_anon_id)
    if anon_id is None:
        return {"merged": False, "events": 0}
    return merge_guest(db, memory_subjects, memory_facts, outbox, anon_id, principal.id)


@app.get("/memory/suggestions")
def memory_suggestions(limit: int = 8, subject: str = Depends(memory_subject)):
    return personalizer.suggestions(memory_briefs.get(subject), limit=max(1, min(limit, 20)))


@app.get("/memory/welcome")
async def memory_welcome(subject: str = Depends(memory_subject)):
    cart_view = None
    if memory_subjects.kind_of(subject) == "customer":
        try:
            cart_view = await shopping.cart(subject)
        except Exception:
            cart_view = None
    return {"card": personalizer.welcome(memory_briefs.get(subject), cart_view)}


@app.post("/chat/feedback")
def chat_feedback(body: FeedbackRequest, principal: Principal = Depends(require_customer),
                  correlation: Correlation = Depends(request_correlation)):
    """A rating on an answer, joined to the turn's evidence by correlation id (Q12)."""
    feedback_id = f"fb_{uuid4().hex[:16]}"
    memory_subjects.ensure(principal.id)
    db.execute(
        "INSERT INTO memory_feedback (id,subject_id,conversation_id,turn_id,correlation_id,"
        "rating,reason,note,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (feedback_id, principal.id, _conversation_key(principal, body.conversation_id),
         body.turn_id, correlation.correlation_id, body.rating, body.reason, body.note,
         _now()))
    ledger.record(actor=Actor(type="customer", id=principal.id, surface="storefront"),
                  action="memory.feedback", reason=f"Customer rated an answer {body.rating}",
                  outcome="applied", target_type="turn", target_id=body.turn_id,
                  state_ref={"rating": body.rating, "reason": body.reason},
                  correlation=correlation)
    schedule_sync(db, outbox, principal.id)
    return {"id": feedback_id}


class ConsentRequest(BaseModel):
    email_opt_in: bool


@app.get("/memory/offer")
def memory_offer(principal: Principal = Depends(require_customer)):
    """The customer's live recovery offer, if any, for the on-site banner."""
    return {"offer": recovery_offers.view(recovery_offers.live_offer(principal.id))}


@app.get("/me/marketing-consent")
def get_marketing_consent(principal: Principal = Depends(require_customer)):
    return marketing_consent.get(principal.id)


@app.put("/me/marketing-consent")
def put_marketing_consent(body: ConsentRequest, principal: Principal = Depends(require_customer)):
    ledger.record(actor=Actor(type="customer", id=principal.id, surface="storefront"),
                  action="marketing.consent", outcome="applied",
                  reason=f"Customer {'opted in to' if body.email_opt_in else 'opted out of'} offer emails")
    return marketing_consent.set(principal.id, body.email_opt_in)


@app.get("/marketing/unsubscribe")
@app.post("/marketing/unsubscribe")
def marketing_unsubscribe(token: str):
    """One-click unsubscribe from the email footer (and the List-Unsubscribe header)."""
    marketing_consent.unsubscribe(token)
    # The same answer whether or not the token matched: it reveals nothing.
    return {"unsubscribed": True}


@app.post("/admin/recovery/scan", dependencies=[Depends(require_operations_token)])
async def admin_recovery_scan():
    """Expire stale offers, issue new ones under the active policy, send the emails."""
    expired = recovery_offers.expire()
    issued = recovery_offers.scan()
    emails = await recovery_email.drain()
    return {"expired": expired, "issued": [o["id"] for o in issued], "emails": emails}


async def _background_jobs() -> None:
    """Memory sync and cart recovery on a timer, when CARTISAN_BACKGROUND_JOBS=1.
    The same work stays callable through the /admin endpoints for an external cron."""
    import asyncio
    import logging

    interval = max(60, int(os.getenv("CARTISAN_JOB_INTERVAL_SECONDS", "300")))
    while True:
        for job in (lambda: memory_sync.drain(limit=50), admin_recovery_scan):
            try:
                await job()
            except Exception:
                logging.getLogger(__name__).warning("background job failed", exc_info=True)
        await asyncio.sleep(interval)


@app.on_event("startup")
async def _start_background_jobs() -> None:
    if os.getenv("CARTISAN_BACKGROUND_JOBS") == "1":
        import asyncio

        asyncio.get_running_loop().create_task(_background_jobs())


@app.post("/admin/memory/sync", dependencies=[Depends(require_operations_token)])
async def admin_memory_sync(limit: int = 20):
    return await memory_sync.drain(limit=max(1, min(limit, 100)))


@app.post("/admin/memory/expire-guests", dependencies=[Depends(require_operations_token)])
def admin_memory_expire_guests():
    return {"expired": expire_guests(db, outbox)}


# -- the merchant surface ----------------------------------------------------
# Reads are open to any operator; decisions are the operator's own act, recorded
# against their verified principal. Nothing below is reachable from a tool.

@app.get("/portal/snapshot")
async def snapshot(window_days: int = 7, principal: Principal = Depends(require_operator)):
    return await merchant_service.snapshot(principal.id, window_days)


@app.get("/portal/metrics")
async def portal_metrics(metric: str, window_days: int = 30, group_by: str | None = None,
                         principal: Principal = Depends(require_operator)):
    try:
        return await merchant_service.metrics(principal.id, metric, window_days, group_by)
    except Unavailable as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/portal/changes")
def portal_changes(limit: int = 50, conversation_id: str | None = None,
                   principal: Principal = Depends(require_operator)):
    """The approval queue: what is waiting, and what was decided, each with the exact
    before-and-after documents the agent staged.

    `conversation_id` narrows the decided half to proposals staged in that
    conversation. Pending ones are always returned: they are unanswered work, not
    conversation history."""
    return merchant_service.changes_list(
        limit=min(limit, 200), conversation_id=conversation_id)


@app.get("/portal/changes/{change_id}")
def portal_change(change_id: str, principal: Principal = Depends(require_operator)):
    try:
        return merchant_service.change(change_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/portal/changes/{change_id}/decision")
def portal_decide(change_id: str, body: DecisionRequest,
                  principal: Principal = Depends(require_operator)):
    """The operator's decision, and — on an approval — the application that follows it.

    Cartisan re-reads the record and re-checks the bounds here, before writing. A
    proposal whose target moved since it was staged, or whose bounds no longer hold
    against current figures, is refused with the reason: the approval stands in the
    ledger, the change is marked failed, and nothing was written (ADR 0016).
    """
    try:
        return merchant_service.decide(
            operator_id=principal.id, change_id=change_id, decision=body.decision,
            note=body.note)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DecisionRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        # The decision is recorded even when application is refused, so its reason
        # and the lessons it feeds are updated either way.
        _learn_from_decision(change_id, body.reason_code)


def _learn_from_decision(change_id: str, reason_code: str | None) -> None:
    try:
        record_reason(db, change_id, reason_code)
        refresh_lessons(db, merchant_lessons, recovery_offers.stats())
    except Exception:
        import logging
        logging.getLogger(__name__).warning("could not update merchant lessons", exc_info=True)


class RecoveryPolicyRequest(BaseModel):
    abandon_after_minutes: int = Field(ge=30, le=7 * 24 * 60)
    min_cart_minor: int = Field(ge=0)
    discount_percentage: int = Field(ge=1, le=20)
    max_discount_minor: int = Field(gt=0)
    cooldown_days: int = Field(ge=7, le=365)
    monthly_budget_minor: int = Field(ge=0)
    offer_valid_hours: int = Field(ge=1, le=168)
    rationale: str = Field(default="Operator proposed a cart recovery policy", max_length=400)


@app.get("/portal/recovery-policy")
def portal_recovery_policy(principal: Principal = Depends(require_operator)):
    return {"active": active_policy(db), "stats": recovery_offers.stats(),
            "bounds": POLICY_BOUNDS["recovery_policy"]}


@app.post("/portal/recovery-policy")
def portal_propose_recovery_policy(body: RecoveryPolicyRequest,
                                   principal: Principal = Depends(require_operator)):
    """Stage a policy. It lands in the approval queue as a pending change like any
    other, and only an approval there puts it in force (Q3)."""
    after = body.model_dump(exclude={"rationale"})
    current = active_policy(db)
    before = {key: current[key] for key in after} if current else {}
    try:
        return merchant_changes.stage(
            operator_id=principal.id, kind="recovery_policy", target_type="recovery_policy",
            target_id=None, before=before, after=after, rationale=body.rationale)
    except PolicyViolation as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/portal/memory")
def portal_memory(principal: Principal = Depends(require_operator)):
    """What the merchant assistant has learned, and from what."""
    refresh_lessons(db, merchant_lessons, recovery_offers.stats())
    return {"facts": [
        {"id": row["id"], "type": row["fact_type"], "key": row["fact_key"], "value": row["value"],
         "since": str(row["valid_from"])}
        for row in merchant_lessons.live_rows(MERCHANT_SUBJECT)]}


@app.delete("/portal/memory/facts/{fact_id}")
def portal_memory_delete(fact_id: str, principal: Principal = Depends(require_operator)):
    if not merchant_lessons.delete_by_id(MERCHANT_SUBJECT, fact_id):
        raise HTTPException(status_code=404, detail="Nothing to delete")
    return {"deleted": True}


# -- evidence -----------------------------------------------------------------
# The evidence ledger, which until now had no reader. These replace `/audit` and
# the flat `audit` table behind it: that table had one row per action with no
# principal filter, no correlation, no origin and no actor type, so every session's
# rows arrived in one undifferentiated list — the "unrelated-session noise" this
# phase has to rule out. `evidence_records` already held all four (ADR 0023).

@app.get("/evidence")
def my_evidence(demo_run_id: str | None = None, correlation_id: str | None = None,
                outcome: str | None = None, limit: int = 100,
                principal: Principal = Depends(require_customer)):
    """A customer's own evidence. The principal filter is applied from the verified
    token and is not a parameter, so this endpoint cannot be widened by asking."""
    try:
        return evidence_view.records(
            actor_id=principal.id, demo_run_id=demo_run_id, correlation_id=correlation_id,
            outcome=outcome, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/evidence/journeys/{correlation_id}")
def my_journey(correlation_id: str, principal: Principal = Depends(require_customer)):
    """One of the customer's own journeys, end to end.

    Ownership is checked against the ledger before the journey is assembled: a
    correlation id is a handle a client can guess at, so it grants nothing on its
    own. A journey the customer has no row in does not exist to them.
    """
    if not evidence_view.records(actor_id=principal.id, correlation_id=correlation_id, limit=1):
        raise HTTPException(status_code=404, detail="No such journey")
    return evidence_view.journey(correlation_id)


@app.get("/portal/evidence")
def portal_evidence(actor_id: str | None = None, demo_run_id: str | None = None,
                    correlation_id: str | None = None, origin: str | None = None,
                    surface: str | None = None, outcome: str | None = None,
                    actor_type: str | None = None, action: str | None = None,
                    target_id: str | None = None, since: str | None = None,
                    limit: int = 100, principal: Principal = Depends(require_operator)):
    """The store-wide ledger, filtered. An operator acts on the whole store, so this
    one takes a principal as a *filter* rather than forcing their own."""
    try:
        return evidence_view.records(
            actor_id=actor_id, demo_run_id=demo_run_id, correlation_id=correlation_id,
            origin=origin, surface=surface, outcome=outcome, actor_type=actor_type,
            action=action, target_id=target_id, since=since, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/portal/evidence/filters")
def portal_evidence_filters(demo_run_id: str | None = None,
                            principal: Principal = Depends(require_operator)):
    """What is actually in the ledger to filter by — the demo runs recorded and the
    action names present, so the UI offers what exists rather than a fixed list."""
    return {"demo_runs": evidence_view.demo_runs(), "origins": list(ORIGINS),
            "actions": evidence_view.actions(demo_run_id=demo_run_id)}


@app.get("/portal/evidence/journeys")
def portal_journeys(actor_id: str | None = None, demo_run_id: str | None = None,
                    origin: str | None = None, surface: str | None = None, limit: int = 40,
                    principal: Principal = Depends(require_operator)):
    """One row per lineage: who started it, what it produced, and how it ended."""
    try:
        return evidence_view.journeys(actor_id=actor_id, demo_run_id=demo_run_id,
                                      origin=origin, surface=surface, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/portal/evidence/journeys/{correlation_id}")
def portal_journey(correlation_id: str, principal: Principal = Depends(require_operator)):
    """One journey from the customer's request to the Razorpay evidence: the turns,
    the tool calls, the order, its payment attempts, and the provider's own answer —
    including a refused one — in the order they happened."""
    journey = evidence_view.journey(correlation_id)
    if not journey["found"]:
        raise HTTPException(status_code=404, detail="No such journey")
    return journey


@app.get("/portal/health")
def portal_health(hours: int = 24, demo_run_id: str | None = None,
                  principal: Principal = Depends(require_operator)):
    """Production health, every figure carrying the formula that produced it and the
    window it covers — the same `Claim` shape the merchant metrics use (ADR 0017)."""
    return health_metrics.report(hours=hours, demo_run_id=demo_run_id)


# -- payment recovery ---------------------------------------------------------
# Reading what is stuck needs an operator; changing it needs the operations token,
# exactly like `/admin/expire`. Nothing here is in any tool list, and no
# model-reachable path reaches it (ADR 0005, ADR 0030).

@app.get("/portal/recovery")
def recovery_queue(limit: int = 50, principal: Principal = Depends(require_operator)):
    """Dead-lettered effects, quarantined provider events, events never decided, and
    orders holding stock with nothing in flight — each with the reason and the
    actions still open to it."""
    return recovery.queue(limit=limit)


@app.post("/admin/recovery/messages/{message_id}/retry",
          dependencies=[Depends(require_operations_token)])
def recovery_retry_message(message_id: str):
    """Return a dead-lettered payment-link request to the queue. The effect is
    idempotent per attempt, so this recovers the existing link rather than making a
    second one."""
    try:
        return recovery.retry_message(message_id)
    except RecoveryRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/admin/recovery/events/{inbox_id}/acknowledge",
          dependencies=[Depends(require_operations_token)])
def recovery_acknowledge(inbox_id: str, body: AcknowledgeRequest):
    """Record that a human read a quarantined event and what they concluded.

    The event stays quarantined. A payload that failed verification is never
    re-applied, because a wrong `paid` is the worst thing this system can produce;
    the recovery is on the order, not on the payload (ADR 0013).
    """
    try:
        return recovery.acknowledge(inbox_id, note=body.note)
    except RecoveryRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/admin/recovery/events/{inbox_id}/reprocess",
          dependencies=[Depends(require_operations_token)])
def recovery_reprocess(inbox_id: str):
    """Re-run an event that was stored but never decided, through the same
    verification a live delivery gets. A payload that does not match is quarantined
    now rather than sitting undecided forever."""
    try:
        return recovery.reprocess_event(inbox_id, webhooks)
    except RecoveryRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/admin/recovery/orders/{order_id}/cancel",
          dependencies=[Depends(require_operations_token)])
def recovery_cancel_order(order_id: str, body: CancelOrderRequest):
    """Give up on an unpaid order and release the stock it holds."""
    try:
        return recovery.cancel_order(order_id, reason=body.reason)
    except RecoveryRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


class SimulatedPaymentRequest(BaseModel):
    method: str
    outcome: str = "success"


@app.get("/pay/sim/{link_id}")
def simulated_payment_page(link_id: str):
    """What the hosted checkout page shows for one payment link."""
    try:
        return simulated_checkout.summary(link_id)
    except SimulatedCheckoutError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/pay/sim/{link_id}")
def simulated_payment_submit(link_id: str, body: SimulatedPaymentRequest):
    """The customer paid (or the payment failed) on the simulated gateway. The
    outcome reaches the order only through the webhook processor."""
    if body.outcome not in {"success", "failure"}:
        raise HTTPException(400, "outcome must be 'success' or 'failure'")
    try:
        outcome = simulated_checkout.complete(
            link_id, method=body.method, succeed=body.outcome == "success")
    except SimulatedCheckoutError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if outcome.get("result") == "applied" and outcome.get("order_id") and body.outcome == "success":
        _remember_purchase(outcome["order_id"])
    return {"ok": True, **outcome}


def _remember_purchase(order_id: str) -> None:
    """A paid order is the strongest memory signal, and only the verified webhook
    knows one happened. Memory never fails a payment, so errors are logged only."""
    try:
        rows = db.rows("SELECT customer_id FROM commerce_orders WHERE id=?", (order_id,))
        customer_id = rows[0]["customer_id"] if rows else None
        if not customer_id or not db.rows("SELECT 1 AS known FROM customers WHERE id=?", (customer_id,)):
            return
        recovery_offers.mark_redeemed(order_id)
        for line in db.rows("SELECT variant_id FROM commerce_order_lines WHERE order_id=?", (order_id,)):
            behavior.record(customer_id, "purchase", line["variant_id"])
    except Exception:
        import logging
        logging.getLogger(__name__).warning("could not record purchase memory", exc_info=True)


# -- telegram (n8n transport) -------------------------------------------------
# n8n relays Telegram updates here and sends back what these return. Every call is
# HMAC-signed with a shared secret; the principal comes from `telegram_links`, which
# only a verified Supabase session can write (docs/TELEGRAM_N8N_PLAN.md).

telegram = tg.TelegramChannel(
    db,
    shopping_runtime=shopping_agent, merchant_runtime=merchant_agent,
    shopping_service=shopping, merchant_service=merchant_service,
    shopping_sessions=(_transcripts, _states),
    merchant_sessions=(_portal_transcripts, _portal_states),
    session_types={"shopping": (SessionContext, SessionState),
                   "merchant": (MerchantSessionContext, MerchantSessionState)},
    link_base_url=os.getenv("TELEGRAM_LINK_BASE_URL", "http://localhost:3000"),
    require_second_approver=os.getenv("TELEGRAM_REQUIRE_SECOND_APPROVER", "1") != "0",
)


class TelegramLinkRequest(BaseModel):
    token: str = Field(min_length=10, max_length=200)


async def _signed_channel_body(request: Request) -> dict:
    raw = await request.body()
    if not tg.verify(os.getenv("TELEGRAM_CHANNEL_SECRET", ""),
                     request.headers.get(tg.TIMESTAMP_HEADER, ""),
                     request.headers.get(tg.SIGNATURE_HEADER, ""), raw):
        raise HTTPException(401, "Invalid channel signature")
    try:
        body = json.loads(raw or b"{}")
    except ValueError as exc:
        raise HTTPException(400, "Malformed body") from exc
    if not isinstance(body, dict):
        raise HTTPException(400, "Malformed body")
    return body


@app.post("/channels/telegram/{bot_kind}/{merchant_id}")
async def telegram_update(bot_kind: str, merchant_id: str, request: Request):
    """One Telegram update in; the Bot API calls to make in reply out.

    Cartisan is a single store today, so `merchant_id` must name it; the path carries
    it so one n8n workflow can serve more stores once the core is multi-tenant.
    """
    update = await _signed_channel_body(request)
    if bot_kind not in tg.BOT_KINDS or merchant_id != os.getenv("CARTISAN_MERCHANT_ID", "cartisan"):
        raise HTTPException(404, "Unknown channel")
    return {"messages": await telegram.handle(bot_kind, update)}


@app.post("/channels/telegram/relay")
async def telegram_relay(request: Request):
    """Proactive messages (account linked, payment verified, change awaiting a
    decision), each returned once. n8n polls this on a schedule."""
    await _signed_channel_body(request)
    return {"deliveries": telegram.relay()}


@app.post("/channels/telegram/link")
def telegram_link(body: TelegramLinkRequest, authorization: str | None = Header(default=None)):
    """Redeem a link token from the bot with a verified Supabase session."""
    try:
        principal = identity.principal(authorization)
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    try:
        return telegram.redeem_link_token(body.token, principal)
    except tg.LinkRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/admin/expire", dependencies=[Depends(require_operations_token)])
def expire_abandoned():
    """Release what abandoned checkouts are holding. Idempotent, and host-triggered:
    no model-reachable path releases stock (ADR 0005)."""
    return checkout_repo.expire_unpaid()


@app.post("/admin/payments/drain", dependencies=[Depends(require_operations_token)])
async def drain_payment_outbox():
    """Deliver any payment-link request that an earlier provider failure left pending."""
    return await dispatcher.drain()
