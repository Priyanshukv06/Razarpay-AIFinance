"""Tier 1 — Exact Match Engine.

Matches settlement records to order records by:
  - order_ref == order_id  (ID linkage)
  - amount_settled == order_amount  (exact amount, zero tolerance)

This catches perfect matches where no fee was deducted (e.g. UPI with
zero MDR). These are the highest-confidence matches possible — no
ambiguity, no AI, confidence = 1.0.

Design note: Tier 1 deliberately does NOT attempt fee-adjusted matching.
That belongs in Tier 2. Keeping tiers narrow and single-purpose makes
each one independently testable and the whole pipeline auditable.
"""

import pandas as pd


def run(settlements: pd.DataFrame, orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run Tier 1 exact matching.

    Args:
        settlements: Source A (settlement report), all columns.
        orders: Source B (order ledger), all columns.

    Returns:
        (results, unmatched_settlements, unmatched_orders):
        - results: DataFrame of matched records in unified result schema.
        - unmatched_settlements: Settlements not matched at this tier.
        - unmatched_orders: Orders not matched at this tier.
    """
    results = []
    matched_stl_ids = set()
    matched_ord_ids = set()

    # Build a lookup of orders by order_id. Multiple orders can share
    # the same order_id (duplicate_order mismatch type), so we use a
    # list-based index rather than assuming uniqueness.
    order_lookup = {}
    for idx, order in orders.iterrows():
        oid = order["order_id"]
        if oid not in order_lookup:
            order_lookup[oid] = []
        order_lookup[oid].append((idx, order))

    for stl_idx, stl_row in settlements.iterrows():
        order_ref = stl_row["order_ref"]
        amount_settled = stl_row["amount_settled"]

        # Skip if this order_ref has no matching order at all
        if order_ref not in order_lookup:
            continue

        candidates = order_lookup[order_ref]

        # If multiple orders share this order_id, the match is ambiguous
        # by definition (duplicate_order mismatch type). Leave these for
        # Tier 3 where the AI can reason about which entry is real.
        # Counting only unmatched candidates: if one of two duplicates was
        # already consumed, the remaining one is unambiguous.
        unmatched_candidates = [(i, o) for i, o in candidates if i not in matched_ord_ids]
        if len(unmatched_candidates) != 1 and len(candidates) > 1:
            continue

        for ord_idx, ord_row in candidates:
            # Skip if this order was already matched by a prior settlement
            if ord_idx in matched_ord_ids:
                continue

            order_amount = ord_row["order_amount"]

            # Exact amount match -- use a tiny epsilon for floating point
            # comparison rather than strict equality, because CSV
            # round-tripping can introduce float noise.
            if abs(amount_settled - order_amount) < 0.01:
                # Check settlement date vs order date. If the gap is > T+2,
                # this is a date_drift case even though amounts match exactly.
                # Let Tier 2 handle it so it gets the correct category label.
                stl_date = pd.to_datetime(stl_row["settlement_date"])
                ord_date = pd.to_datetime(ord_row["order_date"])
                date_diff = (stl_date - ord_date).days
                if date_diff > 2:
                    # Leave for Tier 2 to categorize as date_drift
                    continue

                results.append({
                    "order_id": order_ref,
                    "settlement_id": stl_row["settlement_id"],
                    "tier": 1,
                    "match_status": "matched",
                    "confidence": 1.0,
                    "category": "exact",
                    "explanation": (
                        f"Exact match: order {order_ref} amount "
                        f"{order_amount:.2f} equals settlement "
                        f"{amount_settled:.2f}. No fee deduction."
                    ),
                    "suggested_action": None,
                })
                matched_stl_ids.add(stl_idx)
                matched_ord_ids.add(ord_idx)
                break  # This settlement is matched, move to the next one

    results_df = pd.DataFrame(results)

    # Filter out matched records for downstream tiers
    unmatched_stl = settlements[~settlements.index.isin(matched_stl_ids)].copy()
    unmatched_ord = orders[~orders.index.isin(matched_ord_ids)].copy()

    return results_df, unmatched_stl, unmatched_ord
