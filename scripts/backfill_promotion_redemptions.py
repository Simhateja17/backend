"""Link existing discounts to the promotion that explains them. Idempotent.

    python scripts/backfill_promotion_redemptions.py --dry-run
    python scripts/backfill_promotion_redemptions.py

Campaign attribution needs an order to name the promotion it redeemed. Seeded history
predates that column: it applied a flat ₹200 off any basket over ₹2,000 without ever
recording a promotion, so the discount was a number nothing could attribute.

The rule the generator was already applying *is* a promotion, it was simply never
written down as one. So this names it — MONSOON10 becomes "₹200 off orders above
₹2,000", which is exactly what those orders received — and points every discounted
order and stage at it. No amount changes: not a discount, not a total, not a paid
amount. The rows are only being explained, never rewritten.

AUDIO500 and SMARTHOME15 gain the category scope their descriptions always claimed.
Neither was ever redeemed in the existing history, so both honestly report zero
attributed orders until a fresh seed run applies them.

Run `python scripts/seed_commerce.py --validate` afterwards: `promotion redemptions`
proves every link the backfill made.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from marketplace_backend.store import Store  # noqa: E402

MONSOON = "sd_promo_MONSOON10"

# The rule seeded history actually applied, now written down as the promotion it was.
MONSOON_RULE = {
    "description": "Monsoon sale: ₹200 off orders above ₹2,000",
    "discount_kind": "fixed_minor",
    "discount_value": 20000,
    "min_subtotal_minor": 200000,
}

# The scopes the descriptions have always named.
SCOPES = {
    "sd_promo_FESTIVE500": "sd_cat_ethnic",
    "sd_promo_STRIDE15": "sd_cat_footwear",
}


def open_store(sqlite_path: str | None) -> Store:
    if sqlite_path:
        return Store(sqlite_path)
    url = os.getenv("SUPABASE_DATABASE_URL")
    if not url:
        raise SystemExit("Set SUPABASE_DATABASE_URL, or pass --sqlite PATH for a local run.")
    return Store(database_url=url)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sqlite", help="Run against a local SQLite file instead of Supabase.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change and exit without writing.")
    args = parser.parse_args()
    store = open_store(args.sqlite)

    # Refuse to touch anything whose discount this promotion cannot account for. If a
    # single row disagrees, the rule is not the one that produced the data and naming
    # it would be a false explanation, not a backfill.
    unexplained = store.rows(
        "SELECT id, subtotal_minor, discount_minor FROM commerce_orders "
        "WHERE discount_minor > 0 AND promotion_id IS NULL AND NOT "
        "(discount_minor = ? AND subtotal_minor >= ?)",
        (MONSOON_RULE["discount_value"], MONSOON_RULE["min_subtotal_minor"]))
    if unexplained:
        for row in unexplained[:5]:
            print(f"  {row['id']}: {row['discount_minor']} off {row['subtotal_minor']}",
                  file=sys.stderr)
        raise SystemExit(
            f"{len(unexplained)} discounted orders do not match the ₹200-above-₹2,000 rule. "
            "Backfilling would attribute them to a promotion that did not produce them.")

    live = store.rows(
        "SELECT COUNT(*) AS n FROM commerce_orders "
        "WHERE discount_minor > 0 AND promotion_id IS NULL AND origin <> 'seeded'")[0]["n"]
    if int(live):
        raise SystemExit(
            f"{live} non-seeded discounted orders would be rewritten. This backfill is only "
            "for generated history; a real order's discount needs its own provenance.")

    orders = int(store.rows(
        "SELECT COUNT(*) AS n FROM commerce_orders "
        "WHERE discount_minor > 0 AND promotion_id IS NULL")[0]["n"])
    stages = int(store.rows(
        "SELECT COUNT(*) AS n FROM checkout_stages "
        "WHERE discount_minor > 0 AND promotion_id IS NULL")[0]["n"])
    print(f"orders to link={orders}  stages to link={stages}")
    if args.dry_run:
        return 0

    store.execute(
        "UPDATE promotions SET description=?, discount_kind=?, discount_value=?, "
        "min_subtotal_minor=? WHERE id=?",
        (MONSOON_RULE["description"], MONSOON_RULE["discount_kind"],
         MONSOON_RULE["discount_value"], MONSOON_RULE["min_subtotal_minor"], MONSOON))
    for promotion_id, category_id in SCOPES.items():
        store.execute("UPDATE promotions SET category_id=? WHERE id=?",
                      (category_id, promotion_id))

    store.execute(
        "UPDATE commerce_orders SET promotion_id=? "
        "WHERE discount_minor > 0 AND promotion_id IS NULL", (MONSOON,))
    store.execute(
        "UPDATE checkout_stages SET promotion_id=? "
        "WHERE discount_minor > 0 AND promotion_id IS NULL", (MONSOON,))

    linked = int(store.rows(
        "SELECT COUNT(*) AS n FROM commerce_orders WHERE promotion_id IS NOT NULL")[0]["n"])
    print(f"done: {linked} orders now name the promotion they redeemed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
