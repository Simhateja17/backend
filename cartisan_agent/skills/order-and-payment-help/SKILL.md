---
name: order-and-payment-help
description: The customer asks where an order is, whether a payment went through, or what to do about a failed or abandoned payment.
---

1. Start from `get_orders`, or `get_order_status` when they named an order.
2. Read `payment_state`, not `status`, when answering whether they have paid. Only
   `paid` means paid. `payment_verification_pending` means Cartisan is waiting on
   Paytm and has not confirmed anything yet — say exactly that, and do not reassure
   them that it has gone through.
3. Answer with `present_order_status`, one card per order in flight.

A failed or abandoned payment is resumable while the order still holds its reservation:
tell them they can try again from the order in the app. You cannot retry a payment,
create a new link, cancel the order, or issue a refund; say who does and stop there.

If the customer says they paid but the order does not show it, do not take their word as
the answer and do not promise it will settle. Say what the record shows, that Cartisan
verifies payments against Paytm before marking an order paid, and offer to check the
order again.
