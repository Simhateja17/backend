---
name: checkout-and-payment
description: The customer wants to check out, pay, or complete a purchase, or asks what happens after they confirm.
---

This turn stages their cart and does nothing else. Do not search, do not add, do not
change a quantity, even if something in the conversation looks unfinished.

1. Call `stage_checkout`. It reads the authoritative cart itself; you pass it only the
   fulfillment option, when the customer chose one, and a note when there is something
   they should check.
2. Call `present_checkout` with the returned `stage_id`, in the same round as
   `present_suggestions`.
3. Say what the preview covers and that it is waiting for their confirmation. The
   preview expires; if they come back after it has, stage again rather than referring to
   the old one.

What staging is not: no order exists, no stock is held, no payment link has been
created, and no money has moved. If they ask what happens next, the answer is that
confirming in the app creates the order and hands off to Paytm — done by Cartisan,
not by you.

If staging is refused because the cart is empty or a line can no longer be sold, say
which line and why, and offer to look for a replacement. Do not stage a partial cart of
your own devising.
